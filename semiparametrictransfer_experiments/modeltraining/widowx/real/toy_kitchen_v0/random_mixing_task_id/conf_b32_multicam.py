import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
import numpy as np
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer_experiments.modeltraining.widowx.real.toy_kitchen_v0.dataset_lmdb import bridge_data_config_aliasing, bridge_data_config_kitchen2_aliasing, validation_conf_toykitchen2_room8052_aliasing, TOTAL_NUM_TASKS_ALIASING
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset, FinalImageZerosLMDB_Dataset, LMDB_Dataset_Pandas, TaskConditioningLMDB_Dataset
from semiparametrictransfer.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

configuration = AttrDict(
    main=AttrDict(
        model=GCBCImages,
        max_iterations=400000,
    ),
)

from semiparametrictransfer_experiments.modeltraining.widowx.real.toy_kitchen_v0.dataset_lmdb import toykitchen1
source_data = toykitchen1
source_data.concat_random_cam = True
source_data.filtering_function = [lambda dframe: dframe[(dframe['camera_index'] == 0)]]
# source_data.n_worker = 0

bridge_data_config_aliasing.concat_random_cam = True
bridge_data_config_aliasing.filtering_function = [lambda dframe: dframe[(dframe['camera_index'] == 0)]]

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
    )
)

# data_config = AttrDict(
#     main=AttrDict(
#         dataclass=LMDB_Dataset_Pandas,
#         dataconf=source_data
#     )
# )

model_config = AttrDict(
    main=AttrDict(
        action_dim=7,
        state_dim=7,
        resnet='resnet34',
        task_id_conditioning=80,
        img_sz=[96, 128],
        concatenate_cameras=True
    )
)