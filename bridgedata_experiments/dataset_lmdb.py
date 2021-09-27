import os
from bridgedata.utils.general_utils import AttrDict
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset_Pandas, LMDB_Dataset_Success_Classifier

TOTAL_NUM_TASKS=78
TOTAL_NUM_TASKS_ALIASING=66

task_name_aliasing_dict = {
    "human_demo, put sweet_potato in pot which is in sink": "human_demo, put sweet potato in pot",
    "human_demo, put cup from counter or drying rack into sink": "human_demo, put cup from anywhere into sink",
    "human_demo, put eggplant in pot or pan": "human_demo, put eggplant into pot or pan",
    "human_demo, put eggplant into pan": "human_demo, put eggplant into pot or pan",
    "human_demo, put green squash in pot or pan ": "human_demo, put green squash into pot or pan",
    "human_demo, put pot in sink": "human_demo, put pot or pan in sink",
    "human_demo, put pan in sink": "human_demo, put pot or pan in sink",
    "human_demo, put pan from stove to sink": "human_demo, put pot or pan in sink",
    "human_demo, put pan from drying rack into sink": "human_demo, put pot or pan in sink",
    "human_demo, put pan on stove from sink": "human_demo, put pot or pan on stove",
    "human_demo, put pot on stove which is near stove": "human_demo, put pot or pan on stove",
    "human_demo, put pan from sink into drying rack": "human_demo, put pot or pan from sink into drying rack",
    'human_demo, open small 4-flap box flaps': 'human_demo, open box',
    'human_demo, open white 1-flap box flap': 'human_demo, open box',
    'human_demo, open brown 1-flap box flap': 'human_demo, open box',
    'human_demo, open large 4-flap box flaps': 'human_demo, open box',
    'human_demo, close small 4-flap box flaps': 'human_demo, close box',
    'human_demo, close white 1-flap box flap': 'human_demo, close box',
    'human_demo, close brown 1-flap box flap': 'human_demo, close box',
    'human_demo, close large 4-flap box flaps': 'human_demo, close box',
    'human_demo, put pepper in pan': 'human_demo, put pepper in pot or pan',
}

excluded_dirs = ['initial_testconfig', 'cropped', 'initial_test_config', 'put_eggplant_in_pot_or_pan','chest',
                 'put_big_spoon_from_basket_to_tray', 'put_small_spoon_from_basket_to_tray',
                 'put_fork_from_basket_to_tray']

bridge_data_config = AttrDict(
    name='alldata',
    random_crop=[96, 128],
    color_augmentation=0.1,
    image_size_beforecrop=[112, 144],
    data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/',
    excluded_dirs=excluded_dirs,
    aliasing_dict=task_name_aliasing_dict,
)

toysink1_room8052 = AttrDict(
            name='toysink1_room8052',
            random_crop=[96, 128],
            color_augmentation=0.1,
            image_size_beforecrop=[112, 144],
            data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/',
            filtering_function=[lambda dframe: dframe[(dframe['environment'] == 'toysink1_room8052')]],
            excluded_dirs=excluded_dirs,
            aliasing_dict=task_name_aliasing_dict,
)

validation_conf_toysink1_room8052 = AttrDict(
    val0=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=toysink1_room8052
    ),
)

toysink3_bww = AttrDict(
    name='toysink3_bww',
    random_crop=[96, 128],
    color_augmentation=0.1,
    image_size_beforecrop=[112, 144],
    data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/',
    filtering_function=[lambda dframe: dframe[(dframe['environment'] == 'toysink3_bww')]],
    excluded_dirs=excluded_dirs,
    aliasing_dict=task_name_aliasing_dict,
)

validation_conf_toysink3= AttrDict(
    val0=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=toysink3_bww
    ),
)

toykitchen2 = AttrDict(
            name='toykitchen2',
            random_crop=[96, 128],
            color_augmentation=0.1,
            image_size_beforecrop=[112, 144],
            data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/',
            filtering_function=[lambda dframe: dframe[(dframe['environment'] == 'toykitchen2_room8052') | (dframe['environment'] == 'toykitchen2')]],
            excluded_dirs=excluded_dirs,
            aliasing_dict=task_name_aliasing_dict,
        )

toykitchen2_room8052 = AttrDict(
            name='toykitchen2_room8052',
            random_crop=[96, 128],
            color_augmentation=0.1,
            image_size_beforecrop=[112, 144],
            data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/',
            filtering_function=[lambda dframe: dframe[(dframe['environment'] == 'toykitchen2_room8052')]],
            excluded_dirs=excluded_dirs,
            aliasing_dict=task_name_aliasing_dict,
        )

toykitchen1 = AttrDict(
            name='toykitchen1',
            random_crop=[96, 128],
            color_augmentation=0.1,
            image_size_beforecrop=[112, 144],
            data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/berkeley/toykitchen1',
            aliasing_dict=task_name_aliasing_dict,
            excluded_dirs=excluded_dirs,
)

excl_toykitchen2 = AttrDict(
            name='excl_toykitchen2',
            random_crop=[96, 128],
            color_augmentation=0.1,
            image_size_beforecrop=[112, 144],
            data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/',
            filtering_function=[lambda dframe: dframe[(dframe['environment'] != 'toykitchen2_room8052') & (dframe['environment'] != 'toykitchen2')]],
            aliasing_dict=task_name_aliasing_dict,
            excluded_dirs=excluded_dirs,
        )

excl_toykitchen1 = AttrDict(
            name='excl_toykitchen1',
            random_crop=[96, 128],
            color_augmentation=0.1,
            image_size_beforecrop=[112, 144],
            data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/',
            filtering_function=[lambda dframe: dframe[(dframe['environment'] != 'toykitchen_bww') & (dframe['environment'] != 'toykitchen1')]],
            aliasing_dict=task_name_aliasing_dict,
            excluded_dirs=excluded_dirs,
        )

excl_toysink3= AttrDict(
    name='excl_toysink3',
    random_crop=[96, 128],
    color_augmentation=0.1,
    image_size_beforecrop=[112, 144],
    data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/',
    excluded_dirs=['initial_testconfig', 'cropped', 'initial_test_config', 'put_eggplant_in_pot_or_pan'],
    filtering_function=[lambda dframe: dframe[(dframe['environment'] != 'toysink3_bww')]],
    aliasing_dict=task_name_aliasing_dict,
)

validation_conf_toykitchen2_room8052 = AttrDict(
    val0=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=toykitchen2_room8052
    ),
)


