import shutil
import glob

import argparse
import os
import time
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="orders and launches cameras")
    parser.add_argument("dir", type=str, default='',
                        help="dir with date forlders")

    args = parser.parse_args()

    files = glob.glob(args.dir + '/*/hdf5/*/*')
    print('found {}'.format(len(files)))
    num_files = len(files)
    random.shuffle(files)

    def move_to_dest(sel_files, dest_dir):
        os.makedirs(dest_dir)
        for file in sel_files:
            shutil.copy2(file, dest_dir + '/' + str.split(file, '/')[-4] + '_' + str.split(file, '/')[-1])

    move_to_dest(files[:int(num_files * 0.9)], args.dir + '/hdf5/train')
    move_to_dest(files[int(num_files*0.9):int(num_files * 0.95)], args.dir + '/hdf5/val')
    move_to_dest(files[int(num_files * 0.95):], args.dir + '/hdf5/test')
