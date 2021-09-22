
import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
import numpy as np
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset_Pandas
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

from semiparametrictransfer_experiments.modeltraining.widowx.real.toy_kitchen_v0.dataset_lmdb import source_data_config, bridge_data_config, validation_data_conf
current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = AttrDict(
    batch_size=8,
    main=AttrDict(
        model=GCBCImages,
        max_iterations=400000,
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
        state_dim=7,
        action_dim=7,
        goal_cond=True,
        resnet='resnet34',
        img_sz=[96, 128]
    )
)

