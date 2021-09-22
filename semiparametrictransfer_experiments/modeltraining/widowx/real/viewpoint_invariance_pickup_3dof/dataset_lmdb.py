import os
from semiparametrictransfer.utils.general_utils import AttrDict
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset, LMDB_Dataset_Pandas

COLOR_AUGMENTATION = 0.1


source_and_target_data = AttrDict(
        image_size_beforecrop=[56, 72],
        random_crop=[48, 64],
        color_augmentation=COLOR_AUGMENTATION,
        data_dir=[
            # os.environ['DATA'] + '/toykitchen_v0/vertical_to_front/lmdb',
            # os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant_no_distractors/lmdb',
            # os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant/lmdb'
            os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant_no_distractors/lmdb',
            os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant/lmdb'
            # os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_brown_mouse/lmdb'
        ],
    )



def filtering_func1(data_frame):
    return data_frame[(data_frame['policy_desc'] != 'human_demo, lift random object')]

def filtering_func2(data_frame):
    return data_frame[(data_frame['policy_desc'] == 'human_demo, lift blue elephant') & (data_frame['camera_index'] == 0) |
                      (data_frame['policy_desc'] != 'human_demo, lift blue elephant')]

def filtering_cam0(data_frame):
    return data_frame[(data_frame['camera_index'] == 0)]
def filtering_cam1(data_frame):
    return data_frame[(data_frame['camera_index'] == 1)]
def filtering_cam2(data_frame):
    return data_frame[(data_frame['camera_index'] == 2)]
def filtering_cam3(data_frame):
    return data_frame[(data_frame['camera_index'] == 3)]
def filtering_cam4(data_frame):
    return data_frame[(data_frame['camera_index'] == 4)]

bridge_data= AttrDict(
    image_size_beforecrop=[56, 72],
    random_crop=[48, 64],
    color_augmentation=COLOR_AUGMENTATION,
    data_dir=os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww',
    filtering_function=[filtering_func1, filtering_func2]
)

bridge_data_test= AttrDict(
    image_size_beforecrop=[56, 72],
    random_crop=[48, 64],
    color_augmentation=COLOR_AUGMENTATION,
    data_dir=[os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant/lmdb',
              os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant_no_distractors/lmdb',
    ],
    filtering_function=[filtering_func1, filtering_func2]
)

validation_data_conf = AttrDict(
    val0=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=AttrDict(
            name='target_cam1',
            **source_and_target_data,
            filtering_function=[filtering_cam1]
        )
    ),
    val1=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=AttrDict(
            name='target_cam2',
            **source_and_target_data,
            filtering_function=[filtering_cam2]
        )
    ),
    val2=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=AttrDict(
            name='target_cam3',
            **source_and_target_data,
            filtering_function=[filtering_cam3]
        )
    ),
)

source_task_config = AttrDict(
            **source_and_target_data,
            name='source_task_cam0',
            filtering_function=[filtering_cam0]
            # n_worker=0,  #######################
        )

source_task_config_all_cam = AttrDict(
            **source_and_target_data,
            name='source_task',
            # n_worker=0,  #######################
        )

bridge_data_config = AttrDict(
                    **bridge_data,
                    name='bridge_data',
)

bridge_data_test_config = AttrDict(
                    **bridge_data_test,
                    name='bridge_data_test',
)


bridge_data_multibackground = AttrDict(
    image_size_beforecrop=[56, 72],
    random_crop=[48, 64],
    color_augmentation=COLOR_AUGMENTATION,
    data_dir=[
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_pink_doll/lmdb',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_grey_donkey/lmdb',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_yeelow_turtle/lmdb',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_green_frog/lmdb',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_yellow_corn/lmdb',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_grey_mouse_alldistractors/lmdb',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_brown_mouse_alldistractors/lmdb',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/table_top_lab_vary_background/lmdb',
        os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/8052_bww/lmdb',
        AttrDict(dir=os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant/lmdb',
                 sel_camera=0),
        AttrDict(dir=os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant_no_distractors/lmdb',
                 sel_camera=0),
    ],
)

if __name__ == '__main__':
    conf = source_and_target_data
    # conf = grasp_penguin_redbrush
    # conf['sel_camera'] = 0
    loader = FilteredRoboNetDataset(conf, phase='train').get_data_loader(12)
    from semiparametrictransfer.data_sets.data_utils.test_datasets import measure_time, make_gifs
    # measure_time(loader)
    make_gifs(loader, outdir='/home/febert/Desktop')