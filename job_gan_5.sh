#BSUB -J gan_5
#BSUB -q gpu_v100
#BSUB -n 24
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

python main_gan.py --dp_mode 'GS_WGAN' --load_dir '../pretrain' --wgan --target_eps 5. --noise_multiplier 1.83 --batch_size 32

python main_gan.py --dp_mode 'GS_WGAN' --load_dir '../pretrain' --wgan --target_eps 5. --noise_multiplier 3.97 --batch_size 256 --iterations 12000

python main_gan.py --dp_mode 'DP_SGD' --noise_multiplier 1.83 --target_eps 5.

python main_gan.py --dp_mode 'DP_SGD' --noise_multiplier 3.97 --target_eps 5.