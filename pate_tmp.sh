#BSUB -J pate
#BSUB -q gpu_v100
#BSUB -n 24
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

python main_pate.py --mol_data_dir 'data/gdb9_9nodes_1000.sparsedataset' --target_eps 10. --load_dir '../pretrain'