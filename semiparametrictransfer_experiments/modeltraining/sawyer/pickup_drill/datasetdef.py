import os
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset

pick_drill_wood_noclutter = AttrDict(
    color_augmentation=0.3,
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/sawyerdata/robonet_style_data/pickup_drill_wood_background/clone'
)

pick_drill_stone_back = AttrDict(
    color_augmentation=0.3,
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/sawyerdata/robonet_style_data/teleop_clutter_stone_background/clone'
)

pick_drill_wood_back = AttrDict(
    color_augmentation=0.3,
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/sawyerdata/robonet_style_data/teleop_clutter_wood_background/clone'
)
robonet_downsampled = AttrDict(
    color_augmentation=0.3,
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/misc_datasets/robonet/robonet/downsampled',
    sel_camera=-1
)


validation_data_conf = AttrDict(
    val0=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='pick_drill_wood_cam1',
            **pick_drill_wood_back,
            sel_camera=1,
        )
    ),
    val1=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='pick_drill_stone_cam0',
            **pick_drill_stone_back,
        )
    ),
    val2=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='pick_drill_stone_cam1',
            **pick_drill_stone_back,
            sel_camera=1,
        )
    ),
    val3=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='pick_drill_wood_noclutter_cam1',
            **pick_drill_wood_noclutter,
            sel_camera=1,
        )
    ),
)
