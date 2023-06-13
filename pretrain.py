# =================================================================================== #
#                  This scripts shows how to save pretrained models                   #
# =================================================================================== #
import os
import logging

from rdkit import RDLogger

from args import get_GAN_config
from util_dir.utils_io import get_date_postfix

# Remove flooding logs.
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from solver_gan import Solver
from torch.backends import cudnn


def main(config):
    # For fast training.
    cudnn.benchmark = True

    config.load_dir = os.path.join(config.load_dir, f'num_dis_{config.num_subsets}')
    config.log_dir_path = os.path.join(config.load_dir, 'log_dir')

    # Create directories if not exist.
    if not os.path.exists(config.log_dir_path):
        os.makedirs(config.log_dir_path)

    log_p_name = os.path.join(config.log_dir_path, get_date_postfix() + '_logger.log')
    logging.basicConfig(filename=log_p_name, level=logging.INFO)
    logging.info(config)

    solver = Solver(config, logging)

    solver.pretrain(config.load_dir, config.pretrain_iterations)


if __name__ == '__main__':
    config = get_GAN_config()

    config.mode = 'pretrain'
    config.dp_mode = None
    config.load_dir = './pretrain'
    config.noise_multiplier = 0.
    config.log_score_step = 50
    config.dropout = 0.
    config.g_lr = config.d_lr = 1e-3

    # note, we only log the dis_acc here but use NO EMA step to guide the discriminator training
    config.ema_d_steps = True

    # list_1 = range(100)
    # print("Pretrain the following Discriminators :", list_1)
    # config.net_ids = list_1

    print(config)
    main(config)
