#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=co_rail
#
# Partition:
#SBATCH --partition=savio3_2080ti
#
# QoS:
#SBATCH --qos=rail_2080ti3_normal
#
# Wall clock limit:
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#
## Command(s) to run:
SINGULARITYENV_DATA=/global/scratch/febert/trainingdata \
SINGULARITYENV_EXP=/global/scratch/febert/experiments \
SINGULARITYENV_APPEND_PATH=/global/home/users/febert/miniconda3/bin:$PATH \
singularity exec \
     --nv -B /usr/lib64 -B /var/lib/dcv-gl \
     $IMG_DIR/spt_singularity_gpu.sif \
    /bin/bash -c "export EXP=/global/scratch/febert/experiments; export DATA=/global/scratch/febert/trainingdata;  export PATH=/global/home/users/febert/miniconda3/bin:$PATH; cd /global/home/users/febert/code/semiparametrictransfer; python semiparametrictransfer/train.py experiments/modeltraining/sawyer/transfer/conf.py --prefix gradrev_mult1.0"