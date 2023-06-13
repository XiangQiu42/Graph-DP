#BSUB -J gan_noDP
#BSUB -q gpu_v100
#BSUB -n 24
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

python main_gan.py --non_private --wgan --batch_size 1024

python main_gan.py --non_private --dp_mode 'GS_WGAN' --wgan --batch_size 2048 --load_dir '../pretrain'