#!/usr/bin/env bash
#BSUB -J pretrain
#BSUB -q gpu_v100
#BSUB -n 24
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"


# =================================================================================== #
#                    modify the 'meta_start' (and optionally 'njobs')                 #
#   and run this script multiple times to pretrain the discriminators in parallel     #
# =================================================================================== #

meta_start=0 # the discriminator start index for the current process (need to be modified for each process)
ndis=1000      # total number of discriminators
dis_per_job=50 # number of discriminators to be trained for each process
njobs=20      # ndis // dis_per_job

for i in $(seq 0 $njobs); do
	start=$((i * dis_per_job + meta_start))
	end=$((start + dis_per_job - 1))
	vals=$(seq $start $end)
	python pretrain.py -ids $vals -noise 0. -ndis $ndis -mode 'pretrain'
done