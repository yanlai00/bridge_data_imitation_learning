import argparse
import glob
import os


def clone_data(outfolder, nclones = 50, dry=False):
    if not os.path.exists(outfolder + '/clone'):
        os.makedirs(outfolder + '/clone/hdf5')
    phases = ['train', 'val', 'test']
    for phase in phases:
        cmd = f'cp -r {outfolder}/hdf5/{phase}  {outfolder}/clone/hdf5'
        print(cmd)
        if not dry:
            os.system(cmd)

    train_files = glob.glob(outfolder + '/hdf5/train/*.hdf5')
    for file in train_files:
        for i in range(nclones):
            file_name_ending = str.split(str.split(file, '/')[-1], '.')[0]
            cmd = f'cp {file} {outfolder}/clone/hdf5/train/{file_name_ending}_clone{i}.hdf5'
            print(cmd)
            if not dry:
                os.system(cmd)

    os.system(f'cp {outfolder}/hdf5/normalizing_params.pkl  {outfolder}/clone/hdf5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="clones invdividual hdf5 files n times to improve loading efficiency for very small datasets")
    parser.add_argument('dataset_folder', default='', type=str, help='where to save')
    parser.add_argument('--dry', default=False, action='store_true')
    args = parser.parse_args()
    clone_data(args.dataset_folder, dry=args.dry)
