import os
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

put_in_bowl = AttrDict(
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    # data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/randview_2out5obj_nogato/10view',  #5k total
    # data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/randview_2out5obj_nogato/5view',  #5k total
    data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/randview_2out5obj_nogato/50view',  #5k total
    # data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/pick_only_gatorade_randview/10view/2021-03-04_14-29-34',

    # data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/randview_2out5obj_nogato_noisy/5k_50view/2021-03-09_15-13-39',
    # data_dir = os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/randview_2out5obj_random_actions/5k_10view/2021-03-09_15-14-03',
    # max_train_examples=500, # for lowd data
    sel_camera=-1,
)

put_in_bowl_gatorade = AttrDict(
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/pick_only_gatorade/5k',
    # data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/pick_only_gatorade_randview/10view/2021-03-04_14-29-34',
    # max_train_examples=300,  # for lowdata
    max_train_examples=500,
)

# put_in_bowl_gatorade_10k = AttrDict(
#     random_crop=[48, 64],
#     image_size_beforecrop=[56, 72],
#     data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/pick_only_gatorade/10k'
# )


validation_data_conf = AttrDict(
    val0=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='put_in_bowl_gatorade_cam1',
            **put_in_bowl_gatorade,
            sel_camera=1,
        )
    ),
    val1=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='put_in_bowl_gatorade_cam2',
            **put_in_bowl_gatorade,
            sel_camera=2,
        )
    ),
)

