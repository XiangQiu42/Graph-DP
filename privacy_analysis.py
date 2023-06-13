import sys
from autodp import rdp_acct, rdp_bank

# sys.path.insert(0, '..')
# from args import get_GAN_config

"""
This scripts helps to evaluate the selective sanitization method.
"""


# (10, 1e-5):
#     32 / 20000 / 1.07
#     256 / 10000 / 1.96

# (5, 1e-5):
#     32 / 20000 / 1.83
#     256 / 12000 / 3.97

# (1, 1e-5):
#     32 / 10000 / 5.6
#     64 / 8000 / 7.07
#     256 / 5000 / 11.2

def main(config):
    delta = 1e-5
    batch_size = 32
    prob = 1. / 1000  # subsampling rate
    n_steps = 20000  # training iterations
    sigma = 1.83  # noise scale
    func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)

    acct = rdp_acct.anaRDPacct()
    acct.compose_subsampled_mechanism(func, prob, coeff=n_steps * batch_size)
    epsilon = acct.get_eps(delta)
    print("Privacy cost is: epsilon={}, delta={}".format(epsilon, delta))


if __name__ == '__main__':
    # config = get_GAN_config()
    config = None
    main(config)
