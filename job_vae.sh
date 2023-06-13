#BSUB -J vae
#BSUB -q gpu_v100
#BSUB -n 24
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

python main_vae.py --non_private --batch_size 2048

python main_vae.py --noise_multiplier 1.07 --target_eps 10.

python main_vae.py --noise_multiplier 1.83 --target_eps 5.

python main_vae.py --noise_multiplier 5.6 --target_eps 1.