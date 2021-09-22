import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
import numpy as np
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer_experiments.modeltraining.widowx.real.toy_kitchen_v0.dataset_lmdb import bridge_data_config_aliasing, TOTAL_NUM_TASKS_ALIASING, task_name_aliasing_dict
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset_Pandas
from semiparametrictransfer.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

configuration = AttrDict(
    main=AttrDict(
        model=GCBCImages,
        max_iterations=400000,
    ),
)

bridge_data_config_kitchen1_aliasing = AttrDict(
            name='toykitchen1',
            random_crop=[96, 128],
            color_augmentation=0.1,
            image_size_beforecrop=[112, 144],
            data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/',
            filtering_function=[lambda dframe: dframe[(dframe['environment'] == 'toykitchen1')]],
            aliasing_dict=task_name_aliasing_dict,
        )

validation_conf_toykitchen1_aliasing = AttrDict(
    val0=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=bridge_data_config_kitchen1_aliasing
    ),
)

source_data = bridge_data_config_kitchen1_aliasing
validation_data = validation_conf_toykitchen1_aliasing

data_config = AttrDict(
    main=AttrDict(
        dataclass=RandomMixingDatasetLoader,
        dataconf=AttrDict(
            dataset0=[
                LMDB_Dataset_Pandas,
                source_data,
                0.3
            ],
            dataset1=[
                LMDB_Dataset_Pandas,
                bridge_data_config_aliasing,
                0.7
            ],
        ),
        **validation_data
    )
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=7,
        state_dim=7,
        resnet='resnet34',
        task_id_conditioning=TOTAL_NUM_TASKS_ALIASING,
        img_sz=[96, 128]
    )
)