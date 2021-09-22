import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
import numpy as np
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer_experiments.modeltraining.widowx.real.toy_kitchen_v0.dataset_lmdb import *
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset, FinalImageZerosLMDB_Dataset, LMDB_Dataset_Pandas, TaskConditioningLMDB_Dataset
from semiparametrictransfer.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

configuration = AttrDict(
    main=AttrDict(
        model=GCBCImages,
        max_iterations=400000,
    ),
)

# source_data = toysink1_room8052
# validation_data = validation_conf_toysink1_room8052
source_data = toykitchen2_4tasks
validation_data = validation_conf_toykitchen2_room8052_aliasing

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
                bridge_data_config_excl_kitchen2_aliasing,
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