import os
import argparse


def str2bool(v):
    return v.lower() in ['true']


def get_GAN_config():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=32, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[128, 256, 512], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]],
                        help='number of conv filters in the first layer of D')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='weight for gradient penalty')
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])
    parser.add_argument('--wgan', action='store_true', help='train wassestein GAN')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--iterations', type=int, default=20000, help='iterations for training')
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'RMSprop'])
    parser.add_argument('--g_lr', type=float, default=0.001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.001, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--resume_iterations', type=int, default=None, help='resume training from this step')

    parser.add_argument('--n_critic', type=int, nargs='+', help='number of D updates per each G update')
    parser.add_argument('--ema_d_steps', type=bool, default=False, help='use adaptive discriminators steps')
    parser.add_argument('--beta', type=float, default=0.99, help='decay rate for EMA of discriminator\'s step')
    parser.add_argument('--threshold', type=float, default=0.6, help='the discriminator update threshold')

    # DP configuration. None denotes train the model with no dp guarantee
    parser.add_argument('--dp_mode', type=str, default=None, choices=['DP_SGD', 'GS_WGAN'])

    parser.add_argument('--non_private', action='store_true', help='No DP noise add to training')
    parser.add_argument('--noise_multiplier', '-noise', type=float, default=None, help='noise multiplier')
    parser.add_argument('--target_eps', '-eps', type=float, default=10., help='target eps for DP')
    parser.add_argument('--delta', type=float, default=1e-5, help='the delta')

    # configuration for pretrain of gswgan
    parser.add_argument('--pretrain_iterations', type=int, default=2000, help='iterations for pre-training')
    parser.add_argument('--net_ids', '-ids', type=int, nargs='+', help='the index list for the discriminator')

    # Test configuration.
    parser.add_argument('--test_epochs', type=int, default=100, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_gpus', '-ngpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', '-mode', type=str, default='train', choices=['train', 'test', 'pretrain'])

    # Use either of these two datasets.
    # parser.add_argument('--mol_data_dir', type=str, default='data/zinc_1.sparsedataset')
    # parser.add_argument('--mol_data_dir', type=str, default='data/qm9_5k_1.sparsedataset')
    parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes_1.sparsedataset')
    parser.add_argument('--num_subsets', '-ndis', type=int, default=1, help='number of subset')

    # Directories.
    parser.add_argument('--saving_dir', type=str, default='./exp_results/GAN/')
    parser.add_argument('--load_dir', type=str, default=None, help='checkpoint dir for loading pre-trained models')

    # Step size.
    parser.add_argument('--log_score_step', type=int, default=200)
    parser.add_argument('--sample_step', type=int, default=200)
    parser.add_argument('--model_save_step', type=int, default=2000)
    parser.add_argument('--test_sample_num', type=int, default=1000)

    config = parser.parse_args()

    # For testing
    # config.mode = 'test'
    # # config.z_dim = 64
    # config.mol_data_dir = 'data/qm9_5k_1.sparsedataset'
    # config.saving_dir = 'exp_results/GAN/2023-03-11_17-40-30'
    # config.resume_iterations = 400
    # config.test_sample_num = 20000

    # For training
    if config.mode == 'train':

        # config.dp_mode = 'DP_SGD'
        config.non_private = True
        # config.wgan = True

        # config.batch_size = 8
        # config.g_lr = config.d_lr = 5e-4

        # config.z_dim = 64
        # config.d_conv_dim = [[64, 32], 128, [64, 32]]

        # config.ema_d_steps = True

        if config.dp_mode == 'GS_WGAN':
            # config.mol_data_dir = 'data/qm9_5k_100.sparsedataset'
            # # config.mol_data_dir = 'data/zinc_100.sparsedataset'
            # config.num_subsets = 100

            config.mol_data_dir = 'data/gdb9_9nodes_1000.sparsedataset'
            # config.mol_data_dir = 'data/zinc_1000.sparsedataset'
            config.num_subsets = 1000

            # if config.load_dir is None:
            #     config.load_dir = '../pretrain'

        if not config.non_private:
            assert config.dp_mode is not None
            config.g_lr = config.d_lr = 2e-4

            if config.dp_mode == 'DP_SGD':
                config.batch_size = 2048

                if config.target_eps is None:
                    config.target_eps = 10.

                if config.noise_multiplier is None:
                    print("Noise multiplier not provided, we use sigma = 1.07 to train the model")
                    config.noise_multiplier = 1.07

            elif config.dp_mode == 'GS_WGAN':

                if config.noise_multiplier is None:
                    print("Noise multiplier not provided, we use sigma = 1.0 to train the model")
                    config.noise_multiplier = 1.07

            # this can be refer to the paper "private GANs, revisit"
            if config.ema_d_steps:
                config.n_critic = [1, 3, 5, 10, 20, 50, 100, 500]

                if config.dp_mode == 'GS_WGAN' and config.load_dir is not None:
                    config.n_critic = [5, 10, 20, 50, 100, 500]

                if config.batch_size >= 512:
                    config.threshold = 0.7

            else:
                config.n_critic = [5]

    return config


def get_PATE_config():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=32, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[128, 256, 512], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]],
                        help='number of conv filters in the first layer of D')

    parser.add_argument('--lambda_gp', type=float, default=10., help='weight for gradient penalty')
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--iterations', type=int, default=1000, help='iterations for training')
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'RMSprop'])
    parser.add_argument('--g_lr', type=float, default=1e-3, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=1e-3, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training from this step')
    parser.add_argument('--wgan', type=bool, default=True, help='train wassestein GAN')

    # DP configuration.
    parser.add_argument('--non_private', action='store_true', help='No DP noise add to training')
    parser.add_argument('--target_eps', '-eps', type=float, default=10., help='target eps for DP')
    parser.add_argument('--delta', type=float, default=1e-5, help='the delta')

    # configuration for
    parser.add_argument('--num_subsets', '-ndis', type=int, default=100, help='number of teachers')
    parser.add_argument('--batch_teachers', '-b_teacher', type=int, default=50, help='how many batches of teacher')
    parser.add_argument('--step_size', type=float, default=1e-4, help='Step size for gradient aggregation')
    parser.add_argument('--sigma', type=int, default=100, help='Scale of gaussian noise for gradient aggregation')
    parser.add_argument('--sigma_thresh', type=float, default=600.0, help='Scale of gaussian noise for thresh gnmax')
    parser.add_argument('--pate_thresh', '-p_thresh', type=float, default=0.5, help='threshold for thresh_gmax')
    parser.add_argument('--proj_mat', '-p_mat', type=int, default=1, help='#/ projection mat')

    parser.add_argument('--random_proj', type=bool, default=False, help='Apply random project for gradient aggregation')
    parser.add_argument('--pca', type=bool, default=False, help='Apply pca for gradient aggregation')
    parser.add_argument('--pca_dim', type=int, default=10, help='principal dimensions for pca')

    # Test configuration.
    parser.add_argument('--test_epochs', type=int, default=100, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_gpus', '-ngpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', '-mode', type=str, default='train', choices=['train', 'test', 'pretrain'])

    # Use either of these two datasets.
    parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes_1000.sparsedataset')
    # parser.add_argument('--mol_data_dir', type=str, default='data/qm9_5k_100.sparsedataset')

    # Directories.
    parser.add_argument('--saving_dir', type=str, default='./exp_results/PATE/')
    parser.add_argument('--load_dir', type=str, default=None, help='checkpoint dir for loading pre-trained teachers')

    # Step size.
    parser.add_argument('--log_score_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=int, default=20)
    parser.add_argument('--lr_update_step', type=int, default=1000)
    parser.add_argument('--test_sample_num', type=int, default=1000)

    # For training
    config = parser.parse_args()

    if config.mol_data_dir == 'data/qm9_5k_100.sparsedataset':
        config.num_subsets = 100
        config.batch_teachers = 10

    elif config.mol_data_dir == 'data/gdb9_9nodes_1000.sparsedataset':
        config.num_subsets = 1000
        config.batch_teachers = 25

    elif config.mol_data_dir == 'data/gdb9_9nodes_2000.sparsedataset':
        config.num_subsets = 2000
        config.batch_teachers = 50
    elif config.mol_data_dir == 'data/gdb9_9nodes_4000.sparsedataset':
        config.num_subsets = 4000
        config.batch_teachers = 50

    if config.mode == 'train':
        config.non_private = True

        config.load_dir = '../pretrain'

        # config.batch_size = 64
        # config.d_conv_dim = [[64, 32], 64, [64, 32]]

        config.wgan = False
        if config.wgan:
            config.g_lr = config.d_lr = 1e-4
            config.step_size = 1e-4

        if not config.non_private:
            config.random_proj = True

            if config.mol_data_dir == 'data/gdb9_9nodes_4000.sparsedataset':
                config.sigma_thresh = 3000
                config.sigma = 1000
            elif config.mol_data_dir == 'data/gdb9_9nodes_2000.sparsedataset':
                config.sigma_thresh = 600
                config.sigma = 100
            elif config.mol_data_dir == 'data/gdb9_9nodes_1000.sparsedataset':
                config.sigma_thresh = 600
                config.sigma = 100

    # For testing
    # config.mode = 'test'
    # config.saving_dir = 'exp_results/PATE/2023-03-12_04-05-32'
    # config.resume_epoch = 481
    # config.test_sample_num = 20000

    return config


def get_VAE_config():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=32, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[128, 256, 512], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]],
                        help='number of conv filters in the first layer of D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=1, help='weight for reconstruction loss')
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs for training D')
    parser.add_argument('--g_lr', type=float, default=0.001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.001, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_epochs', type=int, default=100, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Use either of these two datasets.
    parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes_1.sparsedataset')
    # parser.add_argument('--mol_data_dir', type=str, default='data/qm9_5k_1.sparsedataset')
    # parser.add_argument('--mol_data_dir', type=str, default='data/zinc_1.sparsedataset')

    # Directories.
    parser.add_argument('--saving_dir', type=str, default='./exp_results/VAE/')

    # Step size.
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1)
    parser.add_argument('--lr_update_step', type=int, default=1000)
    parser.add_argument('--test_sample_num', type=int, default=100)

    parser.add_argument('--non_private', action='store_true', help='No DP noise add to training')
    parser.add_argument('--noise_multiplier', '-noise', type=float, default=None, help='noise multiplier')
    parser.add_argument('--target_eps', '-eps', type=float, default=10., help='target eps for DP')
    parser.add_argument('--delta', type=float, default=1e-5, help='the delta')

    config = parser.parse_args()

    # For training
    config.mode = 'train'

    config.batch_size = 32
    # config.g_lr = config.d_lr = 1e-3
    # config.d_conv_dim = [[32, 64], 128, [32]]
    # config.dropout = 0.1
    config.mol_data_dir = 'data/qm9_5k_1.sparsedataset'

    config.non_private = True

    if not config.non_private:
        config.dropout = 0.
        config.g_lr = config.d_lr = 2e-4

        if config.noise_multiplier is None:
            config.noise_multiplier = 3.8

        config.batch_size = 2048

    # For testing
    config.mode = 'test'
    config.saving_dir = 'exp_results/VAE/2023-03-23_12-20-37'
    config.resume_epoch = 67
    config.non_private = True
    config.test_sample_num = 10000

    return config
