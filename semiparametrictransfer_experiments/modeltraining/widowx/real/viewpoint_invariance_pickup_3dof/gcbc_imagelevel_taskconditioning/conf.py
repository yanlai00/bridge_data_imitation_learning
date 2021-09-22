import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
import numpy as np
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.dataset_lmdb import bridge_data_config, source_task_config, validation_data_conf
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset, FinalImageZerosLMDB_Dataset, TaskConditioningLMDB_Dataset
from semiparametrictransfer.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

configuration = AttrDict(
    batch_size=8,
    main=AttrDict(
        model=GCBCImages,
        max_iterations=400000,
    ),
)



# data_config = AttrDict(
#     main=AttrDict(
#         dataclass=TaskConditioningLMDB_Dataset,
#         dataconf=AttrDict(
#             **bridge_data_config,
#             conditioning_task='human_demo, lift blue elephant'
#         ),
#         **validation_data_conf
#     )
# )

data_config = AttrDict(
    main=AttrDict(
        dataclass=RandomMixingDatasetLoader,
        dataconf=AttrDict(
            dataset0=[
                FinalImageZerosLMDB_Dataset,
                source_task_config,
                0.3
            ],
            dataset1=[
                LMDB_Dataset,
                bridge_data_config,
                0.7
            ],
        )
    ),
        # **validation_data_conf
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=4,
        resnet='resnet34',
        goal_cond=True,
    )
)