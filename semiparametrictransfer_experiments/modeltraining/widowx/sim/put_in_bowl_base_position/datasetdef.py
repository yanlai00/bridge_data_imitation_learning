import os
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

put_in_bowl_gatorade = AttrDict(
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/pick_only_gatorade/5k',
    # max_train_examples=300,  # for lowdata
    max_train_examples=1000,
)

put_in_bowl_gatorade_base3 = AttrDict(
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/randbase/base3/2021-05-09_01-01-56',
    # max_train_examples=300,  # for lowdata
    max_train_examples=1000,
)

put_in_bowl_gatorade_base5 = AttrDict(
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/randbase/base5/2021-05-09_01-16-47',
    # max_train_examples=300,  # for lowdata
    max_train_examples=1000,
)

put_in_bowl_gatorade_base10 = AttrDict(
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/randbase/base10/2021-05-09_01-31-32',
    # max_train_examples=300,  # for lowdata
    max_train_examples=1000,
)

put_in_bowl_gatorade_base25 = AttrDict(
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/randbase/base25/2021-05-09_01-48-40',
    # max_train_examples=300,  # for lowdata
    max_train_examples=1000,
)

put_in_bowl_gatorade_base50 = AttrDict(
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/randbase/base50/2021-05-09_02-15-37',
    # max_train_examples=300,  # for lowdata
    max_train_examples=1000,
)

put_in_bowl_rand_base = AttrDict(
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/randbase/10k_100/2021-04-23_20-41-34',
    # max_train_examples=300,  # for lowdata
)

put_in_bowl_gatorade_10k = AttrDict(
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/pick_only_gatorade/10k'
)
