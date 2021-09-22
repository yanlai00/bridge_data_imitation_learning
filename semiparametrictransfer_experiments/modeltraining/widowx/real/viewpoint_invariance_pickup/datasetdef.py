import os
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset
from semiparametrictransfer.data_sets.robonet_dataloader_single_timestep import FilteredRoboNetDatasetSingleTimeStep

COLOR_AUGMENTATION = 0.1

grasp_pen = AttrDict(
    T=80,
    image_size_beforecrop=[56, 72],
    random_crop=[48, 64],
    # image_size_beforecrop=[112, 144],
    # random_crop=[96, 128],
    color_augmentation=COLOR_AUGMENTATION,
    robot_list=['widowx'],
    splits=None,
    data_dir=[os.environ['DATA'] + '/spt_trainingdata/control/widowx/vr_record_applied_actions/bww_grasp_pen/hdf5'],
    # data_dir=[os.environ['DATA'] + '/spt_trainingdata/control/widowx/vr_record_applied_actions/bww_grasp_pen/hdf5_separate_tsteps'],
)


grasp_penguin_redbrush = AttrDict(
    T=80,
    image_size_beforecrop=[56, 72],
    random_crop=[48, 64],
    # image_size_beforecrop=[112, 144],
    # random_crop=[96, 128],
    color_augmentation=COLOR_AUGMENTATION,
    robot_list=['widowx'],
    splits=None,
    data_dir=[
    os.environ['DATA'] + '/spt_trainingdata/control/widowx/vr_record_applied_actions/bww_grasp_toy_penguin/hdf5',
      os.environ['DATA'] + '/spt_trainingdata/control/widowx/vr_record_applied_actions/bww_grasp_red_brush/hdf5'
    ],
    # data_dir=[
            # os.environ['DATA'] + '/spt_trainingdata/control/widowx/vr_record_applied_actions/bww_grasp_toy_penguin/hdf5_separate_tsteps',
            #   os.environ['DATA'] + '/spt_trainingdata/control/widowx/vr_record_applied_actions/bww_grasp_red_brush/hdf5_separate_tsteps'
    # ],
)

source_and_target_data = grasp_pen

validation_data_conf = AttrDict(
    val0=AttrDict(
        dataclass=FilteredRoboNetDatasetSingleTimeStep,
        dataconf=AttrDict(
            name='target_cam2',
            **source_and_target_data,
            sel_camera=2,
        )
    ),
    val1=AttrDict(
        dataclass=FilteredRoboNetDatasetSingleTimeStep,
        dataconf=AttrDict(
            name='target_cam3',
            **source_and_target_data,
            sel_camera=3,
        )
    ),
    val2=AttrDict(
        dataclass=FilteredRoboNetDatasetSingleTimeStep,
        dataconf=AttrDict(
            name='target_cam4',
            **source_and_target_data,
            sel_camera=4,
        )
    ),
)

source_task_config = AttrDict(
            name='source_task_cam1',
            **source_and_target_data,
            sel_camera=0,
            # n_worker=0,
        )

bridge_data_config = AttrDict(
                    name='bridge_data',
                    **grasp_penguin_redbrush,
                    # n_worker=0,
                )

if __name__ == '__main__':
    conf = grasp_pen
    # conf = grasp_penguin_redbrush
    # conf['sel_camera'] = 0
    loader = FilteredRoboNetDataset(conf, phase='train').get_data_loader(12)
    from semiparametrictransfer.data_sets.data_utils.test_datasets import measure_time, make_gifs
    # measure_time(loader)
    make_gifs(loader, outdir='/home/febert/Desktop')