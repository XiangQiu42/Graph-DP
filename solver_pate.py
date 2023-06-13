from collections import defaultdict

import os
import time
import copy
import datetime

import torch
import torch.nn.functional as F

from utils import *
from util_dir.rdp_utils import *
from util_dir.pate_core import *
from models_pate import Generator, Discriminator
from data.sparse_molecular_dataset import SparseMolecularDataset

from util_dir.dp_pca import ComputeDPPrincipalProjection
from sklearn.random_projection import GaussianRandomProjection


class Solver_PATE(object):
    """Solver for training and testing."""

    def __init__(self, config, log=None, orders=None):
        """Initialize configurations."""

        # Log
        self.log = log

        # Data loader.
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir, num_subsets=config.num_subsets)

        self.node_dims = [self.data.vertexes, self.data.atom_num_types]
        self.edge_dims = [self.data.vertexes, self.data.vertexes, self.data.bond_num_types]

        self.wgan = config.wgan

        # DP configurations
        self.non_private = config.non_private
        self.target_eps = config.target_eps
        self.delta = config.delta

        self.step_size = config.step_size

        self.sigma = config.sigma
        self.sigma_thresh = config.sigma_thresh
        self.pate_thresh = config.pate_thresh
        self.test_sample_num = config.test_sample_num

        self.random_proj = config.random_proj
        self.proj_mat = config.proj_mat

        self.pca = config.pca
        self.pca_dim = config.pca_dim

        if orders is not None:
            self.orders = np.asarray(orders)
        else:
            self.orders = np.hstack([1.1, np.arange(2, 200)])
        self.rdp_counter = np.zeros(self.orders.shape)

        # num_teachers in PATE
        self.num_subsets = config.num_subsets

        self.batch_teachers = config.batch_teachers
        assert (self.num_subsets % self.batch_teachers == 0)
        self.teachers_batch = self.num_subsets // self.batch_teachers

        # Model configurations.
        self.z_dim = config.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim

        self.la_gp = config.lambda_gp
        self.post_method = config.post_method

        self.metric = 'validity,qed'

        # Training configurations.
        self.batch_size = config.batch_size
        self.iterations = config.iterations
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout

        if self.data.train_count < self.batch_size:
            self.batch_size = self.data.train_count
            print("Adjust batch_size to {}".format(self.data.train_count))

        self.resume_iterations = config.resume_epoch

        # Training or testing, or pretraining.
        self.mode = config.mode

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

        self.load_dir = config.load_dir

        # Step size.
        self.model_save_step = config.model_save_step
        self.sample_step = config.sample_step

        self.optimizerD_list = []
        self.netD_list = []
        self.G = None
        self.netGS = None
        self.g_optimizer = None

        self.score_sum = 0.

        # Build the model.
        self.build_model(config.optim)

    def build_model(self, optimizer):
        """Create a generator (i.e. student) and discriminators (i.e. Teachers)."""
        self.G = Generator(self.g_conv_dim, self.z_dim,
                           self.data.vertexes,
                           self.data.bond_num_types,
                           self.data.atom_num_types,
                           self.dropout)
        self.netGS = copy.deepcopy(self.G)

        for _ in range(self.batch_teachers):
            netD = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim - 1, dropout_rate=self.dropout)
            self.netD_list.append(netD)

        if self.mode != 'pretrain' and self.load_dir is not None:
            _dir_name = os.path.join(self.load_dir, 'num_dis_%d' % self.num_subsets)
            print('==> Load pretrained discriminators from ', _dir_name)

            if self.log is not None:
                self.log.info(f'==> Load pretrained discriminators from {_dir_name}')

            for netD_id in range(self.num_subsets):
                network_path = os.path.join(_dir_name, 'netD_%d' % netD_id, 'netD.pth')
                netD = self.netD_list[netD_id]
                netD.load_state_dict(torch.load(network_path))

        self.G.to(self.device0)
        self.netGS.to(self.device0)
        for netD_id, netD in enumerate(self.netD_list):
            device = self.devices[get_device_id(netD_id, self.num_subsets, self.num_gpus)]
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

        # may need modification
        if self.pca:
            data = self.data.reshape([self.data.shape[0], -1])
            self.pca_components, rdp_budget = ComputeDPPrincipalProjection(
                data,
                self.pca_dim,
                self.orders,
                1.0,
            )
            self.rdp_counter += rdp_budget

        # self.print_network(self.G, 'G', self.log)
        # self.print_network(self.netD_list[0], 'D_0', self.log)

    @staticmethod
    def print_network(model, name, log=None):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        print(name)
        print(model)

        if log is not None:
            log.info(model)
            log.info(name)
            log.info("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_dir_path, '{}-netGs.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

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

        return [delistify(e) for e in softmax]

    def get_gen_mols(self, netG, noise_z, method):
        edges_logits, nodes_logits = netG(noise_z)
        (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        return mols

    def save_checkpoints(self, iterations_i):
        G_path = os.path.join(self.model_dir_path, '{}-netGs.ckpt'.format(iterations_i))
        torch.save(self.netGS.state_dict(), G_path)
        print('Saved model checkpoints into {}...'.format(self.model_dir_path))
        if self.log is not None:
            self.log.info('Saved model checkpoints into {}...'.format(self.model_dir_path))

    # Get scores of the generated sample
    def get_scores(self, iterations_i, losses, scores):
        z = self.sample_z(self.test_sample_num)
        z = torch.from_numpy(z).to(self.device0).float()
        mols = self.get_gen_mols(self.netGS, z, self.post_method)
        m0, m1 = all_scores(mols, self.data, norm=True)  # 'mols' is output of Fake Reward
        for k, v in m1.items():
            scores[k].append(v)
        for k, v in m0.items():
            scores[k].append(np.array(v)[np.nonzero(v)].mean())

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

        is_first = True
        for tag, value in scores.items():
            if is_first:
                log += "\n{}: {:.2f}".format(tag, np.mean(value))
                is_first = False
            else:
                log += ", {}: {:.2f}".format(tag, np.mean(value))

        save_status = False
        if len(scores['valid']) > 1:
            new_score = scores['valid'][-1] + scores['unique'][-1] + scores['novel'][-1]

            if new_score > self.score_sum:
                self.score_sum = new_score
                log += "\n=======> Better Score for this iteration <=======\n"
                save_status = True

        # Saving molecule images.
        if iterations_i % self.sample_step == 0 or save_status:
            mol_f_name = os.path.join(self.img_dir_path,
                                      'mol_{}_{:.2f}.png'.format(iterations_i, scores['QED'][-1]))
            save_mol_img(mols, mol_f_name, is_test=self.mode == 'test')

        # Save checkpoints if necessary
        if iterations_i % self.model_save_step == 0 or save_status:
            self.save_checkpoints(iterations_i=iterations_i)

        print(log)
        if self.log is not None:
            self.log.info(log)

    def aggregate_results(self, output_list, thresh=None, epoch=None, data_dims=[1]):
        if self.pca:
            res, rdp_budget = gradient_voting_rdp(
                output_list,
                self.step_size,
                self.sigma,
                self.sigma_thresh,
                self.orders,
                pca_mat=self.pca_components,
                thresh=thresh
            )

        elif self.random_proj:
            orig_dim = 1
            for dd in data_dims:
                orig_dim = orig_dim * dd

            if epoch is not None:
                proj_dim = min(epoch + 1, self.pca_dim)
            else:
                proj_dim = self.pca_dim

            n_data = output_list[0].shape[0]
            if self.proj_mat > 1:
                proj_dim_ = proj_dim // self.proj_mat
                n_data_ = n_data // self.proj_mat
                orig_dim_ = orig_dim // self.proj_mat
                print("n_data:", n_data)
                print("orig_dim:", orig_dim)
                transformers = [GaussianRandomProjection(n_components=proj_dim_) for _ in range(self.proj_mat)]
                for transformer in transformers:
                    transformer.fit(np.zeros([n_data_, orig_dim_]))
                    print(transformer.components_.shape)
                proj_matrices = [np.transpose(transformer.components_) for transformer in transformers]
                res, rdp_budget = gradient_voting_rdp_multiproj(
                    output_list,
                    self.step_size,
                    self.sigma,
                    self.sigma_thresh,
                    self.orders,
                    pca_mats=proj_matrices,
                    thresh=thresh
                )
            else:
                transformer = GaussianRandomProjection(n_components=proj_dim)
                transformer.fit(np.zeros([n_data, orig_dim]))  # only the shape of output_list[0] is used
                proj_matrix = np.transpose(transformer.components_)

                # proj_matrix = np.random.normal(loc=np.zeros([orig_dim, proj_dim]),
                #                                scale=1 / float(proj_dim), size=[orig_dim, proj_dim])
                res, rdp_budget = gradient_voting_rdp(
                    output_list,
                    self.step_size,
                    self.sigma,
                    self.sigma_thresh,
                    self.orders,
                    pca_mat=proj_matrix,
                    thresh=thresh
                )
        else:
            res, rdp_budget = gradient_voting_rdp(output_list, self.step_size, self.sigma, self.sigma_thresh,
                                                  self.orders, thresh=thresh)

        return res, rdp_budget

    def non_private_aggregation(self, output_list):
        # TODO update non_private aggregation
        # sum_arr = torch.zeros(output_list[0].shape)
        sum_arr = np.zeros(output_list[0].shape)
        for arr in output_list:
            sum_arr += arr
        return sum_arr / len(output_list)

    def train_and_validate(self):
        self.start_time = time.time()

        # Start training from scratch or resume training.
        start_epoch = 0
        if self.resume_iterations is not None:
            start_epoch = self.resume_iterations
            self.restore_model(self.resume_iterations)

        # Start training.
        if self.mode == 'train':
            for i in range(0, self.iterations):
                self.train(_iter=i)
                self.train(_iter=i, mode='val')

        elif self.mode == 'test':
            assert self.resume_iterations is not None
            self.testing_model()
        else:
            raise NotImplementedError

    def train(self, _iter, mode='train'):
        # Recordings
        losses = defaultdict(list)
        scores = defaultdict(list)

        num_batch = 1
        _tb = 1
        _bt = 1
        if mode == 'train':
            num_batch = int(self.data.train_count // self.batch_size)
            _tb = self.teachers_batch
            _bt = self.batch_teachers

        # =================================================================================== #
        #                               1. Train the teachers                                 #
        # =================================================================================== #
        for _ in range(0, num_batch):

            batch_z = self.sample_z(self.batch_size)
            W_D, L_D_F, L_D, L_D_R = 0., 0., 0., 0.

            # train teacher models in batches, teachers_batch: how many batches of teacher
            for batch_id in range(_tb):
                # train each teacher in this batch, batch_teachers: how many teacher in a batch
                for teacher_id in range(_bt):

                    if mode == 'train':
                        mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(
                            self.batch_size, teacher_id + batch_id * self.batch_teachers)
                    else:
                        mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()

                    device = self.devices[get_device_id(teacher_id, self.num_subsets, self.num_gpus)]
                    netD = self.netD_list[teacher_id]
                    optimizerD = self.optimizerD_list[teacher_id]

                    a = torch.from_numpy(a).to(device).long()  # Adjacency.
                    x = torch.from_numpy(x).to(device).long()  # Nodes.
                    a_tensor = self.label2onehot(a, self.b_dim, device)
                    x_tensor = self.label2onehot(x, self.m_dim, device)

                    d_steps = 3 if self.wgan else 1
                    # =================================================================================== #
                    #                            1. Train the discriminator                               #
                    # =================================================================================== #
                    for _ in range(d_steps):

                        optimizerD.zero_grad()

                        # generate fake samples.
                        z = torch.from_numpy(batch_z).to(self.device0).float()
                        edges_hat, nodes_hat = self.G(z)
                        edges_hat = edges_hat.to(device)
                        nodes_hat = nodes_hat.to(device)

                        if self.wgan:
                            # Compute losses with real inputs.
                            logits_real, _ = netD(a_tensor, None, x_tensor)
                            d_loss_real = -torch.mean(logits_real)

                            logits_fake, _ = netD(edges_hat, None, nodes_hat)
                            d_loss_fake = torch.mean(logits_fake)

                            # Compute losses for gradient penalty.
                            grad_penalty = self.gradient_penalty(logits_real.size(0), edges_hat, a_tensor, nodes_hat,
                                                                 x_tensor, device, netD)

                            loss_D = d_loss_real + d_loss_fake + grad_penalty
                            Wasserstein_D = -d_loss_real - d_loss_fake

                            losses['Wasserstein_D'].append(Wasserstein_D.item())
                            del grad_penalty
                        else:
                            logits_real, _ = netD(a_tensor, None, x_tensor, activation=torch.nn.Sigmoid())
                            true_label = torch.full((logits_real.size(0),), 1., dtype=torch.float, device=device)
                            d_loss_real = torch.nn.BCELoss()(logits_real.view(-1), true_label)

                            logits_fake, _ = netD(edges_hat, None, nodes_hat, activation=torch.nn.Sigmoid())
                            fake_label = torch.full((logits_fake.size(0),), 0., dtype=torch.float, device=device)
                            d_loss_fake = torch.nn.BCELoss()(logits_fake.view(-1), fake_label)
                            loss_D = d_loss_real + d_loss_fake

                        # Optimise discriminator.
                        if mode == 'train':
                            loss_D.backward()
                            optimizerD.step()

                        L_D = loss_D.item()
                        L_D_R = d_loss_real.item()
                        L_D_F = d_loss_fake.item()

            losses['l_D'].append(L_D)
            losses['l_D/R'].append(L_D_R)
            losses['l_D/F'].append(L_D_F)

            del mols, a, x, logits_real, logits_fake, z, d_loss_real, d_loss_fake
            torch.cuda.empty_cache()

            # =================================================================================== #
            #                            2. Train the student generator                           #
            # =================================================================================== #
            self.g_optimizer.zero_grad()

            node_grads_list = []
            edge_grads_list = []

            z = torch.from_numpy(batch_z).to(self.device0).float()

            # with torch.no_grad():
            edges_hat, nodes_hat = self.G(z)

            for batch_idx in range(self.teachers_batch):
                for teacher_id in range(self.batch_teachers):
                    device = self.devices[get_device_id(teacher_id, self.num_subsets, self.num_gpus)]
                    netD = self.netD_list[teacher_id]

                    edges_hat = edges_hat.to(device)
                    nodes_hat = nodes_hat.to(device)

                    if self.wgan:
                        logits_fake, _ = netD(edges_hat, None, nodes_hat)
                        _loss_G = -torch.mean(logits_fake)
                    else:
                        logits_fake, _ = netD(edges_hat, None, nodes_hat, activation=torch.nn.Sigmoid())
                        label = torch.full((self.batch_size,), 1., dtype=torch.float, device=self.device0)
                        _loss_G = torch.nn.BCELoss()(logits_fake.view(-1), label)

                    # add the minus gradients to guide the student generator
                    edge_gradients = -torch.autograd.grad(outputs=_loss_G,
                                                          inputs=edges_hat,
                                                          grad_outputs=torch.ones(_loss_G.size()).to(device),
                                                          create_graph=True,
                                                          # retain_graph=True,
                                                          )[0]
                    edge_grads_list.append(edge_gradients.cpu().detach().numpy())

                    node_gradients = -torch.autograd.grad(outputs=_loss_G,
                                                          inputs=nodes_hat,
                                                          grad_outputs=torch.ones(_loss_G.size()).to(device),
                                                          create_graph=True,
                                                          # retain_graph=True,
                                                          )[0]
                    node_grads_list.append(node_gradients.cpu().detach().numpy())

            self.g_optimizer.zero_grad()
            edges_hat, nodes_hat = self.G(z)

            node_grads_agg_list = []
            edge_grads_agg_list = []
            for j in range(self.batch_size):
                thresh = self.pate_thresh

                if self.non_private:
                    node_grads_agg_tmp = self.non_private_aggregation([grads[j] for grads in node_grads_list])
                    edge_grads_agg_tmp = self.non_private_aggregation([grads[j] for grads in edge_grads_list])
                    # self.rdp_counter += 0
                else:
                    node_grads_agg_tmp, rdp_budget_1 = self.aggregate_results(
                        [grads[j] for grads in node_grads_list], thresh=thresh, data_dims=self.node_dims)
                    edge_grads_agg_tmp, rdp_budget_2 = self.aggregate_results(
                        [grads[j] for grads in edge_grads_list], thresh=thresh, data_dims=self.edge_dims)

                    self.rdp_counter += rdp_budget_1
                    self.rdp_counter += rdp_budget_2

                node_grads_agg_list.append(node_grads_agg_tmp)
                edge_grads_agg_list.append(edge_grads_agg_tmp)

            # node_grads_agg = torch.stack(node_grads_agg_list).float().to(self.device0)
            # edge_grads_agg = torch.stack(edge_grads_agg_list).float().to(self.device0)

            node_grads_agg = np.asarray(node_grads_agg_list)
            edge_grads_agg = np.asarray(edge_grads_agg_list)

            update_node = torch.from_numpy(node_grads_agg).float().to(self.device0) + nodes_hat.detach().clone()
            update_edge = torch.from_numpy(edge_grads_agg).float().to(self.device0) + edges_hat.detach().clone()

            loss_G = torch.nn.MSELoss(reduction='sum')(edges_hat, update_edge) + torch.nn.MSELoss(reduction='sum')(
                nodes_hat, update_node)

            losses['l_G'].append(loss_G.item())

            if mode == 'train':
                loss_G.backward()
                self.g_optimizer.step()

            # update the exponential moving average
            exp_mov_avg(self.netGS, self.G, alpha=0.999, global_step=_iter)

        # DP log
        _status = True
        if not self.non_private and mode == 'train':
            eps, order = compute_eps_from_delta(self.orders, self.rdp_counter, self.delta)
            log = "\n(ε = {:.5f}, δ = {:.5f})".format(eps, self.delta)
            # if eps > self.target_eps:
            #     _status = False
            #     log += "\nTarget epsilon exceed and we terminate the training process now !"

            print(log)
            if self.log is not None:
                self.log.info(log)
        if not _status:
            sys.exit()
        # =================================================================================== #
        #                                 4. Miscellaneous                                    #
        # =================================================================================== #
        if mode == 'val':
            self.get_scores(_iter, losses, scores)

    # # generated new graphs and evaluate them
    def testing_model(self):
        scores = defaultdict(list)
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
            scores[k].append(v)
        for k, v in m0.items():
            scores[k].append(np.array(v)[np.nonzero(v)].mean())

        # Print out training information.
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "[Testing with {} samples]:".format(self.test_sample_num)

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

        dump_dir = os.path.join(self.img_dir_path, 'generated_smiles')
        dump(dump_dir, generated_all_similes)
