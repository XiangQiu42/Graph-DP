#BSUB -J gan_10
#BSUB -q gpu_v100
#BSUB -n 24
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

python main_gan.py --dp_mode 'GS_WGAN' --load_dir '../pretrain' --wgan --target_eps 10. --noise_multiplier 1.07 --batch_size 32

python main_gan.py --dp_mode 'GS_WGAN' --load_dir '../pretrain' --wgan --target_eps 10. --noise_multiplier 1.96 --batch_size 256 --iterations 10000

python main_gan.py --dp_mode 'DP_SGD' --noise_multiplier 1.07 --target_eps 10.

python main_gan.py --dp_mode 'DP_SGD' --noise_multiplier 1.96 --target_eps 10.