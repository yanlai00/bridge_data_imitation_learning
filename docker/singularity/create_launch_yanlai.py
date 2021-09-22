import os
import argparse

def make_launch_script(script, conf_file, prefix, dry, ngpu, cmd_str, cpu):
    if cmd_str is "":
        cmd_str= f"{script} --path {conf_file} --prefix {prefix}"
    launch_script = \
        f"""#!/bin/bash
# Job name:
#SBATCH --job-name={prefix}
#
# Account:
#SBATCH --account=co_rail
#
# Partition:
#SBATCH --partition=savio3_gpu
#
# QoS:
#SBATCH --qos=rail_gpu3_normal
#
# Wall clock limit:
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpu}
#SBATCH --gres=gpu:TITAN:{ngpu}

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=yyang22@berkeley.edu

## Command(s) to run:
SINGULARITYENV_DATA=/global/scratch/yyang22/data/trainingdata \
SINGULARITYENV_EXP=/global/scratch/yyang22/data/experiments \
SINGULARITYENV_APPEND_PATH=/global/home/users/yyang22/miniconda3/bin:$PATH \
singularity exec \
     --nv -B /usr/lib64 -B /var/lib/dcv-gl \
     $IMG_DIR/spt_singularity_gpu.sif \
    /bin/bash -c "export EXP=/global/scratch/yyang22/data/experiments; export DATA=/global/scratch/yyang22/data/trainingdata;  export PATH=/global/home/users/yyang22/miniconda3/bin:$PATH; cd /global/scratch/yyang22/code/semiparametrictransfer; /global/scratch/yyang22/miniconda3/envs/py36/bin/python {cmd_str}"
"""
    print(launch_script)
    with open("autogen_launch.sh", "w+") as f:
        f.writelines(launch_script)
    if not dry:
        os.system('sbatch autogen_launch.sh')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--script', default='', type=str)
    parser.add_argument('--conf_file', default='', type=str)
    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--dry', default=False, action='store_true')
    parser.add_argument('--ngpu', default=1, type=int)
    parser.add_argument('--ncpu', default=4, type=int)
    parser.add_argument('--cmd', default='', type=str, help='comand used after python')
    args = parser.parse_args()
    make_launch_script(args.script, args.conf_file, args.prefix, args.dry, args.ngpu, args.cmd, args.ncpu)
