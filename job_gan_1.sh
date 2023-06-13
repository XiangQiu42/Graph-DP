#BSUB -J gan_1
#BSUB -q gpu_v100
#BSUB -n 24
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

python main_gan.py --dp_mode 'GS_WGAN' --load_dir '../pretrain' --wgan --target_eps 1. --noise_multiplier 5.6 --batch_size 32

python main_gan.py --dp_mode 'GS_WGAN' --load_dir '../pretrain' --wgan --target_eps 1. --noise_multiplier 7.07 --batch_size 64 --iterations 8000

python main_gan.py --dp_mode 'DP_SGD' --noise_multiplier 5.6 --target_eps 1.

python main_gan.py --dp_mode 'DP_SGD' --noise_multiplier 7.07 --target_eps 1.