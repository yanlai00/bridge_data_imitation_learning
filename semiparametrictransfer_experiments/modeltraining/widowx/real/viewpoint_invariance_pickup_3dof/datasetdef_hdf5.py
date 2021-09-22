import os
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset

COLOR_AUGMENTATION = 0.1

source_and_target_data = AttrDict(
    T=25,
    image_size_beforecrop=[56, 72],
    random_crop=[48, 64],
    color_augmentation=COLOR_AUGMENTATION,
    robot_list=['widowx'],
    splits=None,
    data_dir=[os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_brown_mouse/hdf5'],
    # data_dir=[
        # os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_pink_doll/hdf5',
        # os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_grey_donkey/hdf5',
        # os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_yeelow_turtle/hdf5',
        # os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_green_frog/hdf5',
        # os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant/hdf5',
        # ],
    target_adim=4,
    target_sdim=4,
)


bridge_data = AttrDict(
    T=25,
    image_size_beforecrop=[56, 72],
    random_crop=[48, 64],
    color_augmentation=COLOR_AUGMENTATION,
    robot_list=['widowx'],
    splits=None,
    target_adim=4,
    target_sdim=4,
    # data_dir=[
    # os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/lift_grey_mouse/hdf5',
    # ],
    data_dir=[
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/2k/hdf5',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_pink_doll/hdf5',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_grey_donkey/hdf5',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_yeelow_turtle/hdf5',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_green_frog/hdf5',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant/hdf5',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_yellow_corn/hdf5',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_grey_mouse_alldistractors/hdf5',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_brown_mouse_alldistractors/hdf5',
    ],
)


validation_data_conf = AttrDict(
    val0=AttrDict(
        dataclass=FilteredRoboNetDataset,
        dataconf=AttrDict(
            name='target_cam1',
            **source_and_target_data,
            sel_camera=1,
        )
    ),
    val1=AttrDict(
        dataclass=FilteredRoboNetDataset,
        dataconf=AttrDict(
            name='target_cam2',
            **source_and_target_data,
            sel_camera=2,
        )
    ),
    val2=AttrDict(
        dataclass=FilteredRoboNetDataset,
        dataconf=AttrDict(
            name='target_cam3',
            **source_and_target_data,
            sel_camera=3,
        )
    ),
)


source_task_config = AttrDict(
            **source_and_target_data,
            name='source_task_cam0',
            sel_camera=0,
            # n_worker=0,  #######################
        )

bridge_data_config = AttrDict(
                    **bridge_data,
                    name='bridge_data',
                    sel_camera='random'
                    # n_worker=0,  ########################
)

if __name__ == '__main__':
    conf = source_and_target_data
    # conf = grasp_penguin_redbrush
    # conf['sel_camera'] = 0
    loader = FilteredRoboNetDataset(conf, phase='train').get_data_loader(12)
    from semiparametrictransfer.data_sets.data_utils.test_datasets import measure_time, make_gifs
    # measure_time(loader)
    make_gifs(loader, outdir='/home/febert/Desktop')