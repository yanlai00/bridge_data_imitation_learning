import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
import numpy as np
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
# from semiparametrictransfer_experiments.modeltraining.widowx.real.toy_kitchen_v0.dataset_lmdb import source_data_config, bridge_data_config, validation_data_conf
import copy
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset, FinalImageZerosLMDB_Dataset, LMDB_Dataset_Pandas, TaskConditioningLMDB_Dataset
from semiparametrictransfer.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

configuration = AttrDict(
    main=AttrDict(
        model=GCBCImages,
        max_iterations=400000,
    ),
)

# task = 'human_demo, put eggplant on plate'
# task = 'human_demo, put eggplant in pot or pan'
task = 'human_demo, put green squash in pot or pan '

def filter_right_side(data_frame):
    return data_frame[(data_frame['environment_portion'] == 'right_side')]

def filter_not_right_side(data_frame):
    return data_frame[(data_frame['environment_portion'] != 'right_side')]

def exclude_task(data_frame):
    return data_frame[(data_frame['policy_desc'] != task)]

def include_task(data_frame):
    return data_frame[(data_frame['policy_desc'] == task)]

source_data_config = AttrDict(
    name='source_data',
    random_crop=[96, 128],
    color_augmentation=0.1,
    image_size_beforecrop=[112, 144],
    data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam',
    filtering_function=[include_task, filter_right_side]
)

validation_data_config = copy.deepcopy(source_data_config)
validation_data_config.filtering_function = [include_task, filter_not_right_side]
validation_data_config.name = task + ' not right side'

bridge_data_config = AttrDict(
    name='alldata',
    random_crop=[96, 128],
    color_augmentation=0.1,
    image_size_beforecrop=[112, 144],
    data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam',
    excluded_dirs=['initial_testconfig', 'cropped'],
    filtering_function=[exclude_task]
)

validation_data_conf = AttrDict(
    val0=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=validation_data_config
    ),
)


data_config = AttrDict(
    main=AttrDict(
        dataclass=RandomMixingDatasetLoader,
        dataconf=AttrDict(
            dataset0=[
                LMDB_Dataset_Pandas,
                source_data_config,
                0.3
            ],
            dataset1=[
                LMDB_Dataset_Pandas,
                bridge_data_config,
                0.7
            ],
        ),
        **validation_data_conf
    )
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=7,
        state_dim=7,
        resnet='resnet34',
        task_id_conditioning=22,
        img_sz=[96, 128]
    )
)