from collections import defaultdict

import os
import sys
import time
import copy
import datetime

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from utils import *
from rdkit import Chem
from models_gan import Generator, Discriminator
from data.sparse_molecular_dataset import SparseMolecularDataset

CLIP_BOUND = 1.
SENSITIVITY = 2.


# =================================================================================== #
#                                     Hook functions                                  #
# =================================================================================== #
def master_hook_adder(module, grad_input, grad_output):
    """
    global hook
    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    """
    global dynamic_hook_function
    return dynamic_hook_function(module, grad_input, grad_output)


def dummy_hook(module, grad_input, grad_output):
    """
    dummy hook
    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    """
    pass


def modify_gradnorm_conv_hook(module, grad_input, grad_output):
    """
    gradient modification hook
    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    """
    # get grad wrt. input
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize  # account for the 'sum' operation in GP

    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    # clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image
    grad_input_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)


def dp_conv_hook(module, grad_input, grad_output):
    """
    gradient modification + noise hook
    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    """
    global noise_multiplier
    # get grad wrt. input
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize

    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    # clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image

    # add noise
    noise = clip_bound_ * noise_multiplier * SENSITIVITY * torch.randn_like(grad_wrt_image)
    grad_wrt_image = grad_wrt_image + noise
    grad_input_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)


class Solver(object):
    """Solver for training and testing MolGAN."""

    def __init__(self, config, log=None):
        """Initialize configurations."""

        # Log
        self.log = log

        # Data loader.
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir, num_subsets=config.num_subsets)

        # DP configurations
        self.dp_mode = config.dp_mode
        self.target_eps = config.target_eps
        self.delta = config.delta

        self.noise_multiplier = config.noise_multiplier
        if self.dp_mode == 'GS_WGAN':
            global noise_multiplier
            noise_multiplier = config.noise_multiplier

        # note, for the mode of selective sanitization, we need multiplier discriminators
        self.num_subsets = config.num_subsets
        self.load_dir = config.load_dir  # the pretrained dir

        # Model configurations.
        self.z_dim = config.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.wgan = config.wgan

        self.la_gp = config.lambda_gp
        self.post_method = config.post_method

        self.metric = 'validity,qed'

        # Training configurations.
        self.batch_size = config.batch_size
        self.iterations = config.iterations
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout

        # self.lr_decay = config.lr_decay_rate

        self.ema_d_steps = config.ema_d_steps
        self.d_acc = []
        self.n_critic = config.n_critic
        self.threshold = config.threshold
        self.beta = config.beta
        self.critic_index = 0
        self.g_step_cnt = 0

        self.resume_iterations = config.resume_iterations

        # Training or testing, or pretraining.
        self.mode = config.mode

        # the discriminators index used in pretrain mode
        self.net_ids = config.net_ids

        # Miscellaneous.
        self.num_gpus = config.num_gpus
        use_cuda = torch.cuda.is_available()
        self.devices = [torch.device("cuda:%d" % i if use_cuda else "cpu") for i in range(self.num_gpus)]
        self.device0 = self.devices[0]
        print('Note we will use the following devices: ')
        for _d in self.devices:
            print("\t", _d)

        # Directories.
        self.log_dir_path = config.log_dir_path
        self.log_score_step = config.log_score_step

        if self.mode != 'pretrain':
            self.model_dir_path = config.model_dir_path
            self.img_dir_path = config.img_dir_path

        # Step size.
        self.model_save_step = config.model_save_step
        self.sample_step = config.sample_step
        self.test_sample_num = config.test_sample_num

        self.G = None
        self.netGS = None
        self.g_optimizer = None
        self.optimizerD_list = []
        self.netD_list = []

        # self.d_lr_scheduler = []
        # self.g_lr_scheduler = None

        # Recordings for training
        self.losses = defaultdict(list)
        self.scores = defaultdict(list)
        self.score_sum = 0.

        # Build the model.
        self.build_model(config.optim)

    def build_model(self, optimizer):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.z_dim,
                           self.data.vertexes,
                           self.data.bond_num_types,
                           self.data.atom_num_types,
                           self.dropout)
        self.netGS = copy.deepcopy(self.G)

        num_discriminators = 1

        if self.mode == 'pretrain':
            num_discriminators = len(self.net_ids)
        elif self.dp_mode == 'GS_WGAN':
            # for the selective sanitization method, we may train different discriminator on different subset
            num_discriminators = self.num_subsets

        for _ in range(num_discriminators):
            netD = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim - 1, dropout_rate=0.)
            self.netD_list.append(netD)

        self.G.to(self.device0)
        self.netGS.to(self.device0)
        for netD_id, netD in enumerate(self.netD_list):
            device = self.devices[get_device_id(netD_id, num_discriminators, self.num_gpus)]
            netD.to(device)

        # Set up optimizers
        if optimizer == 'Adam':
            self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, betas=(0.5, 0.999))
            for i in range(len(self.netD_list)):
                netD = self.netD_list[i]
                optimizerD = torch.optim.Adam(netD.parameters(), self.d_lr, betas=(0.5, 0.999))
                self.optimizerD_list.append(optimizerD)
        else:
            self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), self.g_lr)
            for i in range(len(self.netD_list)):
                netD = self.netD_list[i]
                optimizerD = torch.optim.RMSprop(netD.parameters(), self.d_lr)
                self.optimizerD_list.append(optimizerD)

        # self.g_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=self.g_optimizer, gamma=self.lr_decay)
        # for i in range(len(self.netD_list)):
        #     d_scheduler_temp = lr_scheduler.ExponentialLR(optimizer=self.optimizerD_list[i], gamma=self.lr_decay)
        #     self.d_lr_scheduler.append(d_scheduler_temp)

        if self.dp_mode == 'DP_SGD':
            from opacus.accountants import RDPAccountant
            from opacus import GradSampleModule
            from opacus.optimizers import DPOptimizer

            # initialize privacy accountant
            self.dp_accountant = RDPAccountant()

            # then wrap the discriminator, note in dp_sgd we only entails one discriminators
            assert len(self.netD_list) == 1
            self.dp_dis = GradSampleModule(self.netD_list[0])

            # wrap optimizer
            self.dp_optimizer = DPOptimizer(
                optimizer=self.optimizerD_list[0],
                noise_multiplier=self.noise_multiplier,  # same as make_private arguments
                max_grad_norm=1.0,  # same as make_private arguments
                expected_batch_size=self.batch_size
            )

            # attach accountant to track privacy for an optimizer
            _sample_rate = self.batch_size / (self.data.train_count * self.num_subsets)
            # _sample_rate = self.batch_size / 10000
            assert _sample_rate < 1.
            self.dp_optimizer.attach_step_hook(
                self.dp_accountant.get_optimizer_hook_fn(
                    sample_rate=_sample_rate
                )
            )

        elif self.dp_mode == 'GS_WGAN' and self.mode == 'train':
            # Load pre-trained discriminators
            if self.mode != 'pretrain' and self.load_dir is not None:
                _dir_name = os.path.join(self.load_dir, 'num_dis_%d' % self.num_subsets)
                print('==> Load pretrained discriminators from ', _dir_name)

                if self.log is not None:
                    self.log.info(f'==> Load pretrained discriminators from {_dir_name}')

                for netD_id in range(self.num_subsets):
                    network_path = os.path.join(_dir_name, 'netD_%d' % netD_id, 'netD.pth')
                    netD = self.netD_list[netD_id]
                    netD.load_state_dict(torch.load(network_path))

            global dynamic_hook_function
            for netD in self.netD_list:
                netD.gcn_layer.multi_graph_convolution_layers.conv_nets[0].register_full_backward_hook(
                    master_hook_adder)
        else:
            pass

        # self.print_network(self.G, 'G', self.log)
        # self.print_network(self.netD_list[0], 'D_0', self.log)

    @staticmethod
    def print_network(model, name, log=None):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        if log is not None:
            log.info(name + " ==>")
            log.info(model)
            log.info("The number of parameters: {}".format(num_params))

        print(name + " ==>")
        print(model)

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_dir_path, '{}-netGs.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

        # D_path = os.path.join(self.model_dir_path, '{}-D.ckpt'.format(resume_iters))
        # V_path = os.path.join(self.model_dir_path, '{}-V.ckpt'.format(resume_iters))
        # self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        # self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        # for param_group in self.d_optimizer.param_groups:
        #     param_group['lr'] = d_lr

    def gradient_penalty(self, size, edges_hat, a_tensor, nodes_hat, x_tensor, device=None, netD=None):
        """Compute gradient penalty: """
        if device is None:
            device = self.device0

        if not netD:
            netD = self.netD_list[0]

        eps = torch.rand(size, 1, 1, 1).to(device)
        x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
        x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
        grad0, grad1 = netD(x_int0, None, x_int1)

        dydx_0 = torch.autograd.grad(outputs=grad0,
                                     inputs=x_int0,
                                     grad_outputs=torch.ones(grad0.size()).to(device),
                                     create_graph=True,
                                     retain_graph=True,
                                     only_inputs=True)[0]
        dydx_0 = dydx_0.view(dydx_0.size(0), -1)

        dydx_1 = torch.autograd.grad(outputs=grad1,
                                     inputs=x_int1,
                                     grad_outputs=torch.ones(grad1.size()).to(device),
                                     create_graph=True,
                                     only_inputs=True)[0]
        dydx_1 = dydx_0.view(dydx_1.size(0), -1)

        gp_0 = ((dydx_0.norm(2, dim=1) - 1) ** 2).mean()
        gp_1 = ((dydx_1.norm(2, dim=1) - 1) ** 2).mean()

        return self.la_gp * (gp_0 + gp_1)

    def label2onehot(self, labels, dim, device):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size()) + [dim]).to(device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

    def sample_z(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    @staticmethod
    def postprocess(inputs, method, temperature=1.):
        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

    def reward(self, mols):
        rr = 1.
        for m in ('logp,sas,qed,unique' if self.metric == 'all' else self.metric).split(','):

            if m == 'np':
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == 'logp':
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == 'sas':
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, self.data)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, self.data)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, self.data)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))

        return rr.reshape(-1, 1)

    def get_gen_mols(self, netG, noise_z, method):
        edges_logits, nodes_logits = netG(noise_z)
        (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        return mols

    def get_reward(self, n_hat, e_hat, method):
        (edges_hard, nodes_hard) = self.postprocess((e_hat, n_hat), method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        reward = torch.from_numpy(self.reward(mols)).to(self.device0)
        return reward

    def save_checkpoints(self, iterations_i):
        G_path = os.path.join(self.model_dir_path, '{}-netGs.ckpt'.format(iterations_i + 1))
        torch.save(self.netGS.state_dict(), G_path)
        print('Saved model checkpoints into {}...'.format(self.model_dir_path))
        if self.log is not None:
            self.log.info('Saved model checkpoints into {}...'.format(self.model_dir_path))

    # print the training information
    def show_losses(self, iterations_i, losses):
        # Print out training information.
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}], Iteration [{}/{}]:".format(et, iterations_i + 1, self.iterations)

        is_first = True
        for tag, value in losses.items():
            if is_first:
                log += "\n{}: {:.2f}".format(tag, np.mean(value))
                is_first = False
            else:
                log += ", {}: {:.2f}".format(tag, np.mean(value))

        # show the discriminator's accuracy if necessary
        if self.ema_d_steps:
            log += ", {}: {:.2f}".format('d_acc', self.d_acc[-1])

        print(log)

        if self.log is not None:
            self.log.info(log)

    # Get scores of the generated sample and show information on validating
    def get_scores(self, iterations_i, losses, scores):
        z = self.sample_z(self.test_sample_num)
        z = torch.from_numpy(z).to(self.device0).float()
        mols = self.get_gen_mols(self.netGS, z, self.post_method)
        # mols = self.get_gen_mols(self.G, z, self.post_method)

        m0, m1 = all_scores(mols, self.data, norm=True)  # 'mols' is output of Fake Reward
        for k, v in m1.items():
            scores[k].append(v)
        for k, v in m0.items():
            scores[k].append(np.array(v)[np.nonzero(v)].mean())

        # Print out training information.
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "[Validating...] ==> Iteration [{}/{}]:".format(iterations_i + 1, self.iterations)

        is_first = True
        for tag, value in losses.items():
            if is_first:
                log += "\n{}: {:.2f}".format(tag, value[-1])
                is_first = False
            else:
                log += ", {}: {:.2f}".format(tag, value[-1])

        # show the discriminator's accuracy if necessary
        if self.ema_d_steps:
            log += ", {}: {:.2f}".format('d_acc', self.d_acc[-1])

        is_first = True
        for tag, value in scores.items():
            if is_first:
                log += "\n{}: {:.2f}".format(tag, value[-1])
                is_first = False
            else:
                log += ", {}: {:.2f}".format(tag, value[-1])

        _status = True

        # DP log
        if self.dp_mode == 'DP_SGD' and self.noise_multiplier > 0.:
            epsilon = self.dp_accountant.get_epsilon(delta=self.delta)
            log += "\n(ε = {:.5f}, δ = {:.5f})".format(epsilon, self.delta)

            if epsilon >= self.target_eps:
                _status = False
                log += "\nTarget epsilon exceed and we terminate the training process now !"

        if self.mode == 'train':
            save_status = False
            if len(scores['valid']) > 1:
                new_score = scores['valid'][-1] + scores['unique'][-1] + scores['novel'][-1]

                if new_score > self.score_sum:
                    self.score_sum = new_score
                    log += "\n=======> Better Score for this iteration <=======\n"
                    save_status = True

            # Saving molecule images.
            if (iterations_i+1)  % self.sample_step == 0 or save_status:
                mol_f_name = os.path.join(self.img_dir_path,
                                          'mol_{}_{:.2f}.png'.format(iterations_i, scores['QED'][-1]))
                save_mol_img(mols, mol_f_name, is_test=self.mode == 'test')

            # Save checkpoints if necessary
            if (iterations_i+1) % self.model_save_step == 0 or save_status:
                self.save_checkpoints(iterations_i=iterations_i)

        print(log)
        if self.log is not None:
            self.log.info(log)

        if not _status:
            self.save_checkpoints(iterations_i)
            sys.exit()

    def train_and_validate(self):
        self.start_time = time.time()

        # Start training from scratch or resume training.
        if self.resume_iterations is not None:
            self.restore_model(self.resume_iterations)
            print("We restore the model from iteration {}".format(self.resume_iterations))

        # Start training.
        if self.mode == 'train':
            for i in range(0, self.iterations):

                if self.dp_mode == 'GS_WGAN':
                    self.train_or_valid_GSWGAN(_iter=i, train_val_test='train')
                    if (i + 1) % self.log_score_step == 0:
                        self.train_or_valid_GSWGAN(_iter=i, train_val_test='val')

                elif self.dp_mode == 'DP_SGD':
                    self.train_or_valid_DPSGD(_iter=i, train_val_test='train')
                    log_critic = self.n_critic[self.critic_index] if self.ema_d_steps else self.log_score_step
                    if (i + 1) % log_critic == 0:
                        self.dp_dis.disable_hooks()
                        self.train_or_valid_DPSGD(_iter=i, train_val_test='val')
                        self.dp_dis.enable_hooks()

                else:
                    self.train_or_valid(_iter=i, train_val_test='train')
                    if (i + 1) % self.log_score_step == 0:
                        self.train_or_valid(_iter=i, train_val_test='val')

                # if (i + 1) % 1000 == 0:
                #     self.g_lr_scheduler.step()
                #     for i in range(len(self.netD_list)):
                #         self.d_lr_scheduler[i].step()

        elif self.mode == 'test':
            assert self.resume_iterations is not None
            self.testing_model()
        else:
            raise NotImplementedError

    """ Conventional train function """

    def train_or_valid(self, _iter, train_val_test='val'):
        # =================================================================================== #
        #                              0. select the subset index                             #
        # =================================================================================== #
        # input_index = np.random.randint(self.num_subsets, size=1)[0]
        netD = self.netD_list[0]
        device = self.device0
        optimizerD = self.optimizerD_list[0]

        d_steps = 1
        if train_val_test == 'train' and self.wgan:
            d_steps = 5

        for p in netD.parameters():
            p.requires_grad = True
        # =================================================================================== #
        #                            1. Train the discriminator                               #
        # =================================================================================== #
        for _ in range(d_steps):
            if train_val_test == 'val':
                mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
                z = self.sample_z(a.shape[0])
            elif train_val_test == 'train':
                mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
                z = self.sample_z(self.batch_size)
            else:
                raise RuntimeError("wrong setting for 'train_val_test'")

            optimizerD.zero_grad()

            a = torch.from_numpy(a).to(device).long()  # Adjacency.
            x = torch.from_numpy(x).to(device).long()  # Nodes.
            a_tensor = self.label2onehot(a, self.b_dim, device)
            x_tensor = self.label2onehot(x, self.m_dim, device)

            # Compute losses with fake inputs.
            z = torch.from_numpy(z).to(self.device0).float()
            edges_logits, nodes_logits = self.G(z)
            # Post-process with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            edges_hat = edges_hat.to(device)
            nodes_hat = nodes_hat.to(device)

            if self.wgan:
                # Compute losses with real inputs.
                logits_real, _ = netD(a_tensor, None, x_tensor)
                d_loss_real = torch.mean(logits_real)

                logits_fake, _ = netD(edges_hat, None, nodes_hat)
                d_loss_fake = torch.mean(logits_fake)

                # Compute losses for gradient penalty.
                grad_penalty = self.gradient_penalty(logits_real.size(0), edges_hat, a_tensor, nodes_hat,
                                                     x_tensor, device, netD)

                loss_D = -d_loss_real + d_loss_fake + grad_penalty
                Wasserstein_D = d_loss_real - d_loss_fake

                self.losses['Wasserstein_D'].append(Wasserstein_D.item())
                del grad_penalty
            else:
                logits_real, _ = netD(a_tensor, None, x_tensor, activation=torch.sigmoid)
                true_label = torch.full((logits_real.size(0),), 1., dtype=torch.float, device=self.device0)
                d_loss_real = torch.nn.BCELoss()(logits_real.view(-1), true_label)

                logits_fake, _ = netD(edges_hat, None, nodes_hat, activation=torch.sigmoid)
                fake_label = torch.full((logits_fake.size(0),), 0., dtype=torch.float, device=self.device0)
                d_loss_fake = torch.nn.BCELoss()(logits_fake.view(-1), fake_label)
                loss_D = d_loss_real + d_loss_fake

            self.losses['l_D'].append(loss_D.item())
            self.losses['l_D/R'].append(d_loss_real.item())
            self.losses['l_D/F'].append(d_loss_fake.item())

            # Optimise discriminator.
            if train_val_test == 'train':
                loss_D.backward()
                optimizerD.step()

        del mols, a, x, logits_real, logits_fake, z, d_loss_real, d_loss_fake
        torch.cuda.empty_cache()

        # =================================================================================== #
        #                               2. Train the generator                                #
        # =================================================================================== #
        # Generator update
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation

        self.g_optimizer.zero_grad()

        z = self.sample_z(self.batch_size)
        z = torch.from_numpy(z).to(self.device0).float()
        edges_logits, nodes_logits = self.G(z)
        # Post-process with Gumbel softmax
        (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
        edges_hat = edges_hat.to(device)
        nodes_hat = nodes_hat.to(device)

        if self.wgan:
            logits_fake, _ = netD(edges_hat, None, nodes_hat)
        else:
            logits_fake, _ = netD(edges_hat, None, nodes_hat, activation=torch.sigmoid)

        if self.ema_d_steps:
            # calculate the discriminator's accuracy based on fake samples, we only log the dis_acc here
            with torch.no_grad():
                _odd_acc = self.d_acc[-1] if len(self.d_acc) > 0 else 0
                _logits = logits_fake.detach().cpu().view(-1)
                if self.wgan:
                    _logits = torch.sigmoid(logits_fake).view(-1)

                dis_acc = get_accuracy(0, _logits)
                _acc = discriminators_EMA(_odd_acc, dis_acc, self.beta, global_step=_iter)
                self.d_acc.append(_acc)

        if self.wgan:
            loss_G = -torch.mean(logits_fake)
        else:
            label = torch.full((self.batch_size,), 1., dtype=torch.float, device=self.device0)
            loss_G = torch.nn.BCELoss()(logits_fake.view(-1), label)

        self.losses['l_G'].append(loss_G.item())

        if train_val_test == 'train':
            loss_G.backward()
            self.g_optimizer.step()

            # update the exponential moving average
            exp_mov_avg(self.netGS, self.G, alpha=0.999, global_step=_iter)

        # =================================================================================== #
        #                                 4. Miscellaneous                                    #
        # =================================================================================== #
        if train_val_test == 'val':
            self.get_scores(_iter, self.losses, self.scores)
        else:
            # self.show_losses(_iter, losses)
            pass

    """  train function for dp_sgd """

    def train_or_valid_DPSGD(self, _iter, train_val_test='val'):
        # =================================================================================== #
        #                              0. select the subset index                             #
        # =================================================================================== #
        # input_index = np.random.randint(self.num_subsets, size=1)[0]
        netD = self.dp_dis
        device = self.device0
        optimizerD = self.dp_optimizer

        # =================================================================================== #
        #                            1. Train the discriminator                               #
        # =================================================================================== #
        if train_val_test == 'val':
            mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
            z = self.sample_z(a.shape[0])
        elif train_val_test == 'train':
            mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
            z = self.sample_z(self.batch_size)
        else:
            raise RuntimeError("wrong setting for 'train_val_test'")

        optimizerD.zero_grad(set_to_none=True)

        a = torch.from_numpy(a).to(device).long()  # Adjacency.
        x = torch.from_numpy(x).to(device).long()  # Nodes.
        a_tensor = self.label2onehot(a, self.b_dim, device)
        x_tensor = self.label2onehot(x, self.m_dim, device)

        # Compute losses with fake inputs.
        z = torch.from_numpy(z).to(device).float()
        edges_logits, nodes_logits = self.G(z)
        # Post-process with Gumbel softmax
        (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)

        if self.wgan:
            # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # # Since currently Opacus doesn't support multiple backward(), to overcome this, we need
            # # 1. split fwd-bkwd on fake and real images into wo explicit blocks.
            # # 2. no need to compute per_sample_gardients on fake data, disable hooks.
            # # fake fwd-bkwd
            # netD.disable_hooks()

            logits_fake, _ = netD(edges_hat, None, nodes_hat)
            d_loss_fake = torch.mean(logits_fake)

            # if train_val_test == 'train':
            #     d_loss_fake.backward()
            # # 3. re-enable hooks to obtain per_sample_gardients for real data.
            # netD.enable_hooks()

            # Compute loss with real inputs.
            logits_real, _ = netD(a_tensor, None, x_tensor)
            d_loss_real = torch.mean(logits_real)

            # Currently, opacus_modified doesn't support gradient penalty.
            # grad_penalty = self.gradient_penalty(logits_real.size(0), edges_hat, a_tensor, nodes_hat, x_tensor, device)
            # if train_val_test == 'train':
            #     grad_penalty.backward()
            # loss_D = d_loss_real + d_loss_fake + grad_penalty

            loss_D = -d_loss_real + d_loss_fake
            Wasserstein_D = d_loss_real - d_loss_fake

            # # Compute losses with real inputs.
            # logits_real, _ = netD(a_tensor, None, x_tensor)
            # d_loss_real = -torch.mean(logits_real)
            #
            # logits_fake, _ = netD(edges_hat, None, nodes_hat)
            # d_loss_fake = torch.mean(logits_fake)
            #
            # # Compute losses for gradient penalty.
            # grad_penalty = self.gradient_penalty(logits_real.size(0), edges_hat, a_tensor, nodes_hat,
            #                                      x_tensor, device, netD)
            #
            # loss_D = d_loss_real + d_loss_fake + grad_penalty
            # Wasserstein_D = -d_loss_real - d_loss_fake
            #
            # self.losses['Wasserstein_D'].append(Wasserstein_D.item())
            # del grad_penalty
            self.losses['Wasserstein_D'].append(Wasserstein_D.item())
        else:
            logits_real, _ = netD(a_tensor, None, x_tensor, activation=torch.sigmoid)
            true_label = torch.full((logits_real.size(0),), 1., dtype=torch.float, device=self.device0)
            d_loss_real = torch.nn.BCELoss()(logits_real.view(-1), true_label)

            logits_fake, _ = netD(edges_hat, None, nodes_hat, activation=torch.sigmoid)
            fake_label = torch.full((logits_fake.size(0),), 0., dtype=torch.float, device=self.device0)
            d_loss_fake = torch.nn.BCELoss()(logits_fake.view(-1), fake_label)
            loss_D = d_loss_real + d_loss_fake

        self.losses['l_D'].append(loss_D.item())
        self.losses['l_D/R'].append(d_loss_real.item())
        self.losses['l_D/F'].append(d_loss_fake.item())

        # Optimise discriminator.
        if train_val_test == 'train':
            loss_D.backward()
            optimizerD.step()

        del mols, a, x, logits_real, logits_fake, z, d_loss_real, d_loss_fake
        torch.cuda.empty_cache()

        _critic = 1
        if train_val_test == 'train' and self.ema_d_steps:
            _critic = self.n_critic[self.critic_index]

        # =================================================================================== #
        #                               2. Train the generator                                #
        # =================================================================================== #
        if _iter % _critic == 0:
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # no need to compute per sample gradients for G update, disable hooks and then re-enable after backward
            netD.disable_hooks()

            self.g_optimizer.zero_grad()

            z = self.sample_z(self.batch_size)
            z = torch.from_numpy(z).to(device).float()
            edges_logits, nodes_logits = self.G(z)
            # Post-process with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)

            if self.wgan:
                logits_fake, _ = netD(edges_hat, None, nodes_hat)
                loss_G = -torch.mean(logits_fake)
            else:
                logits_fake, _ = netD(edges_hat, None, nodes_hat, activation=torch.sigmoid)
                label = torch.full((self.batch_size,), 1., dtype=torch.float, device=self.device0)
                loss_G = torch.nn.BCELoss()(logits_fake.view(-1), label)

            self.losses['l_G'].append(loss_G.item())

            if train_val_test == 'train':
                loss_G.backward()
                self.g_optimizer.step()

                # update the exponential moving average
                exp_mov_avg(self.netGS, self.G, alpha=0.999, global_step=_iter)

                # calculate the discriminator's accuracy based on fake samples
                with torch.no_grad():
                    _odd_acc = self.d_acc[-1] if len(self.d_acc) > 0 else 0
                    _logits = logits_fake.detach().cpu().view(-1)
                    if self.wgan:
                        _logits = torch.sigmoid(logits_fake).view(-1)

                    dis_acc = get_accuracy(0, _logits)
                    _acc = discriminators_EMA(_odd_acc, dis_acc, self.beta, global_step=_iter)
                    self.d_acc.append(_acc)

                self.g_step_cnt += 1

            netD.enable_hooks()
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            if self.ema_d_steps and train_val_test == 'train':
                _acc = self.d_acc[-1]
                if _acc < self.threshold and self.critic_index < len(self.n_critic) - 1 and \
                        self.g_step_cnt >= (2 / (1 - self.beta)):
                    self.critic_index += 1
                    self.g_step_cnt = 0
                    _log = "-" * 10 + f"with dis_acc = {_acc}, we move to next discriminator step " \
                                      f"frequency {self.n_critic[self.critic_index]}" + "-" * 10
                    print(_log)

                    if self.log is not None:
                        self.log.info(_log)

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            if train_val_test == 'val':
                self.get_scores(_iter, self.losses, self.scores)

    """ Below are the functions that used when dp_mode == 'GS_WGAN' """

    def train_or_valid_GSWGAN(self, _iter, train_val_test='val'):
        # =================================================================================== #
        #                              0. select the subset index                             #
        # =================================================================================== #
        input_index = np.random.randint(self.num_subsets, size=1)[0]
        netD = self.netD_list[input_index]
        device = self.devices[get_device_id(input_index, self.num_subsets, self.num_gpus)]
        optimizerD = self.optimizerD_list[input_index]

        # =================================================================================== #
        #                            1. Train the discriminator                               #
        # =================================================================================== #
        for p in netD.parameters():
            p.requires_grad = True

        # For the discriminators, we add no sanitization mechanism
        global dynamic_hook_function
        dynamic_hook_function = dummy_hook

        _critic = 5 if train_val_test == 'train' else 1
        if train_val_test == 'train' and self.ema_d_steps:
            _critic = self.n_critic[self.critic_index]

        for _ in range(_critic):
            if train_val_test == 'val':
                mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
                z = self.sample_z(a.shape[0])
            elif train_val_test == 'train':
                mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size, input_index)
                z = self.sample_z(self.batch_size)
            else:
                raise RuntimeError("wrong setting for 'train_val_test'")

            optimizerD.zero_grad()

            a = torch.from_numpy(a).to(device).long()  # Adjacency.
            x = torch.from_numpy(x).to(device).long()  # Nodes.
            a_tensor = self.label2onehot(a, self.b_dim, device)
            x_tensor = self.label2onehot(x, self.m_dim, device)

            # Compute losses with fake inputs.
            z = torch.from_numpy(z).to(device).float()
            edges_logits, nodes_logits = self.G(z)
            # Post-process with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)

            logits_fake, _ = netD(edges_hat, None, nodes_hat)
            d_loss_fake = torch.mean(logits_fake)

            # Compute loss with real inputs.
            logits_real, _ = netD(a_tensor, None, x_tensor)
            d_loss_real = torch.mean(logits_real)

            # Compute losses for gradient penalty.
            grad_penalty = self.gradient_penalty(logits_real.size(0), edges_hat, a_tensor, nodes_hat,
                                                 x_tensor, device, netD)
            loss_D = -d_loss_real + d_loss_fake + grad_penalty

            Wasserstein_D = d_loss_real - d_loss_fake

            self.losses['Wasserstein_D'].append(Wasserstein_D.item())
            self.losses['l_D'].append(loss_D.item())
            self.losses['l_D/R'].append(d_loss_real.item())
            self.losses['l_D/F'].append(d_loss_fake.item())

            # Optimise discriminator.
            if train_val_test == 'train':
                loss_D.backward()
                optimizerD.step()

        del mols, a, x, logits_real, logits_fake, z, d_loss_real, d_loss_fake
        torch.cuda.empty_cache()

        # =================================================================================== #
        #                               2. Train the generator                                #
        # =================================================================================== #
        if train_val_test == 'train' and self.noise_multiplier is not None:
            # Sanitize the gradients passed to the Generator
            dynamic_hook_function = dp_conv_hook
        else:
            # Only modify the gradient norm, without adding noise
            dynamic_hook_function = modify_gradnorm_conv_hook

        for p in netD.parameters():
            p.requires_grad = False
        self.g_optimizer.zero_grad()

        z = self.sample_z(self.batch_size)
        z = torch.from_numpy(z).to(device).float()
        edges_logits, nodes_logits = self.G(z)
        # Post-process with Gumbel softmax
        (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)

        logits_fake, _ = netD(edges_hat, None, nodes_hat)
        loss_G = -torch.mean(logits_fake)
        self.losses['l_G'].append(loss_G.item())

        if train_val_test == 'train':
            loss_G.backward()
            self.g_optimizer.step()
            # update the exponential moving average
            exp_mov_avg(self.netGS, self.G, alpha=0.999, global_step=_iter)

            # calculate the discriminator's accuracy based on fake samples
            with torch.no_grad():
                _odd_acc = self.d_acc[-1] if len(self.d_acc) > 0 else 0
                _logits = logits_fake.detach().cpu().view(-1)
                if self.wgan:
                    _logits = torch.sigmoid(logits_fake).view(-1)

                dis_acc = get_accuracy(0, _logits)
                _acc = discriminators_EMA(_odd_acc, dis_acc, self.beta, global_step=_iter)
                self.d_acc.append(_acc)

            self.g_step_cnt += 1

        if self.ema_d_steps and train_val_test == 'train':
            _acc = self.d_acc[-1]
            if _acc < self.threshold and self.critic_index < len(self.n_critic) - 1 and \
                    self.g_step_cnt >= (2 / (1 - self.beta)):
                self.critic_index += 1
                self.g_step_cnt = 0
                _log = "-" * 10 + f"with dis_acc = {_acc}, we move to next discriminator step " \
                                  f"frequency {self.n_critic[self.critic_index]}" + "-" * 10
                print(_log)

                if self.log is not None:
                    self.log.info(_log)

        # =================================================================================== #
        #                                 4. Miscellaneous                                    #
        # =================================================================================== #
        if train_val_test == 'val':
            self.get_scores(_iter, self.losses, self.scores)

    def pretrain(self, pretrain_dir, pretrain_iterations):
        self.start_time = time.time()

        # Recordings
        losses = defaultdict(list)
        scores = defaultdict(list)

        # Train (non-private) Generator for each Discriminator
        for idx, netD_id in enumerate(self.net_ids):

            # stop the process if finished
            if netD_id >= self.num_subsets:
                print('ID {} exceeds the num of discriminators'.format(netD_id))
                sys.exit()

            # Discriminator
            netD = self.netD_list[idx]
            optimizerD = self.optimizerD_list[idx]
            device = self.devices[get_device_id(netD_id, self.num_subsets, self.num_gpus)]

            # Save dir for each discriminator
            save_subdir = os.path.join(pretrain_dir, 'netD_%d' % netD_id)

            if os.path.exists(os.path.join(save_subdir, 'netD.pth')):
                print("netD %d already pre-trained" % netD_id)
            else:
                os.makedirs(save_subdir, exist_ok=True)

                # create dirs for different discriminators
                self.img_dir_path = os.path.join(save_subdir, 'img_dir')
                if not os.path.exists(self.img_dir_path):
                    os.makedirs(self.img_dir_path)

                for iter in range(pretrain_iterations):
                    # =================================================================================== #
                    #                            1. Train the discriminator                               #
                    # =================================================================================== #
                    for p in netD.parameters():
                        p.requires_grad = True

                    n_steps = 5 if self.wgan else 1
                    if (iter + 1) % self.log_score_step == 0:
                        # print("[validating the model on this epoch]")
                        n_steps = 1

                    for _ in range(n_steps):
                        if (iter + 1) % self.log_score_step == 0:
                            mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
                            z = self.sample_z(a.shape[0])
                        else:
                            mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size, netD_id)
                            z = self.sample_z(self.batch_size)

                        a = torch.from_numpy(a).to(device).long()  # Adjacency.
                        x = torch.from_numpy(x).to(device).long()  # Nodes.
                        a_tensor = self.label2onehot(a, self.b_dim, device)
                        x_tensor = self.label2onehot(x, self.m_dim, device)

                        # Compute losses with real inputs.
                        optimizerD.zero_grad()
                        logits_real, features_real = netD(a_tensor, None, x_tensor)
                        d_loss_real = -torch.mean(logits_real)

                        # Compute losses with fake inputs.
                        z = torch.from_numpy(z).to(self.device0).float()
                        edges_logits, nodes_logits = self.G(z)
                        # Post-process with Gumbel softmax
                        (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
                        edges_hat = edges_hat.to(device)
                        nodes_hat = nodes_hat.to(device)
                        logits_fake, _ = netD(edges_hat, None, nodes_hat)
                        d_loss_fake = torch.mean(logits_fake)

                        # Compute losses for gradient penalty.
                        grad_penalty = self.gradient_penalty(logits_real.size(0), edges_hat, a_tensor, nodes_hat,
                                                             x_tensor, device, netD)

                        loss_D = d_loss_real + d_loss_fake + grad_penalty
                        Wasserstein_D = -d_loss_real - d_loss_fake

                        # Optimise discriminator.
                        # self.reset_grad(dis_index = netD_id)
                        if (iter + 1) % self.log_score_step != 0:
                            loss_D.backward()
                            optimizerD.step()

                        losses['Wasserstein_D'].append(Wasserstein_D.item())
                        losses['l_D'].append(loss_D.item())
                        losses['l_D/R'].append(d_loss_real.item())
                        losses['l_D/F'].append(d_loss_fake.item())

                    del a, x, a_tensor, x_tensor, z, d_loss_real, d_loss_fake, grad_penalty, edges_hat, nodes_hat
                    torch.cuda.empty_cache()

                    # =================================================================================== #
                    #                               2. Train the generator                                #
                    # =================================================================================== #
                    for p in netD.parameters():
                        p.requires_grad = False
                    self.g_optimizer.zero_grad()

                    z = self.sample_z(self.batch_size)
                    z = torch.from_numpy(z).to(self.device0).float()
                    edges_logits, nodes_logits = self.G(z)
                    # Post-process with Gumbel softmax
                    (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
                    edges_hat = edges_hat.to(device)
                    nodes_hat = nodes_hat.to(device)
                    logits_fake, features_fake = netD(edges_hat, None, nodes_hat)

                    # calculate the discriminator's accuracy based on fake samples
                    with torch.no_grad():
                        _odd_acc = self.d_acc[-1] if len(self.d_acc) > 0 else 0
                        _logits = logits_fake.detach().cpu().view(-1)
                        if self.wgan:
                            _logits = torch.sigmoid(logits_fake).view(-1)

                        dis_acc = get_accuracy(0, _logits)
                        _acc = discriminators_EMA(_odd_acc, dis_acc, self.beta, global_step=iter)
                        self.d_acc.append(_acc)

                    loss_G = -torch.mean(logits_fake)
                    losses['l_G'].append(loss_G.item())
                    if (iter + 1) % self.log_score_step != 0:
                        loss_G.backward()
                        self.g_optimizer.step()

                    # get scores
                    if (iter + 1) % self.log_score_step == 0:
                        # get Scores
                        mols = self.get_gen_mols(self.G, z, self.post_method)
                        m0, m1 = all_scores(mols, self.data, norm=True)  # 'mols' is output of Fake Reward
                        for k, v in m1.items():
                            scores[k].append(v)
                        for k, v in m0.items():
                            scores[k].append(np.array(v)[np.nonzero(v)].mean())

                        # Saving molecule images.
                        mol_f_name = os.path.join(self.img_dir_path, 'mol-{}.png'.format(iter))
                        save_mol_img(mols, mol_f_name, is_test=self.mode == 'test')

                        # Print out training information.
                        et = time.time() - self.start_time
                        et = str(datetime.timedelta(seconds=et))[:-7]
                        log = "Elapsed [{}], Iteration [{}/{}]:".format(et, iter + 1, pretrain_iterations)

                        is_first = True
                        for tag, value in losses.items():
                            if is_first:
                                log += "\n{}: {:.2f}".format(tag, value[-1])
                                is_first = False
                            else:
                                log += ", {}: {:.2f}".format(tag, value[-1])

                        if self.ema_d_steps:
                            log += ", {}: {:.2f}".format('d_acc', self.d_acc[-1])

                        is_first = True
                        for tag, value in scores.items():
                            if is_first:
                                log += "\n{}: {:.2f}".format(tag, value[-1])
                                is_first = False
                            else:
                                log += ", {}: {:.2f}".format(tag, value[-1])
                        print(log)

                        if self.log is not None:
                            self.log.info(log)

                # Save the pretrained model
                torch.save(netD.state_dict(), os.path.join(save_subdir, 'netD.pth'))
                print("\n--" * 6 + "Discriminator-%d has pre-trained" % netD_id + "--" * 6 + "\n")

    # # generated new graphs and evaluate them
    def testing_model(self):

        generated_all_similes = []
        all_mols = []
        while True:
            z = self.sample_z(self.batch_size)
            z = torch.from_numpy(z).to(self.device0).float()
            bacth_mols = self.get_gen_mols(self.G, z, self.post_method)

            for mol in bacth_mols:
                try:
                    smile = Chem.MolToSmiles(mol)
                    if smile is not None:
                        all_mols.append(mol)
                        generated_all_similes.append(smile)

                except:
                    continue

            if len(generated_all_similes) > self.test_sample_num:
                break

        # print(len(all_mols))
        m0, m1 = all_scores(all_mols, self.data, norm=True)  # 'mols' is output of Fake Reward
        for k, v in m1.items():
            self.scores[k].append(v)
        for k, v in m0.items():
            self.scores[k].append(np.array(v)[np.nonzero(v)].mean())

        # Print out training information.
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "[Testing with {} samples]:".format(self.test_sample_num)

        is_first = True
        for tag, value in self.scores.items():
            if is_first:
                log += "\n{}: {:.2f}".format(tag, value[-1])
                is_first = False
            else:
                log += ", {}: {:.2f}".format(tag, value[-1])

        print(log)
        if self.log is not None:
            self.log.info(log)

        dump_dir = os.path.join(self.img_dir_path, 'generated_smiles')
        dump(dump_dir, generated_all_similes)