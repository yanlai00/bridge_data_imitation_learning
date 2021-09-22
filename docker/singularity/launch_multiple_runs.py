import os
import argparse

def launch_multiple(dry, extra_suffix):
    subfolders = ['bc_fromscratch', 'transfer', 'transfer/camera_class','bridge_targetfinetune','bridge_targetfinetune/camera_class']
    suffix = ['bc', 'tr', 'tr_cl', 'br', 'br_cl']
    suffix = [l + extra_suffix for l in suffix]

    for folder, suf in zip(subfolders, suffix):
        cmd =  f'python docker/singularity/create_launch_script.py --script  semiparametrictransfer/train.py --conf_file  experiments/modeltraining/widowx/sim/put_in_bowl/{folder}/conf.py  --prefix {suf}'
        print(cmd)
        if not dry:
            os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dry', default=False, action='store_true')
    parser.add_argument('--suffix', default='', type=str)
    args = parser.parse_args()
    launch_multiple(args.dry, args.suffix)
