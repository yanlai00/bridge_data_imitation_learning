import os
from imitation_learning.models.gcbc_images import GCBCImages
import numpy as np
from imitation_learning.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from imitation_learning_experiments.modeltraining.widowx.real.toy_kitchen_v0.dataset_lmdb import bridge_data_config, validation_data_conf, validation_conf_toysink1_room8052, toysink1_room8052
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset, LMDB_Dataset_Pandas

configuration = AttrDict(
    main=AttrDict(
        model=GCBCImages,
        max_iterations=400000,
    ),
)

bridge_data = toysink1_room8052

data_config = AttrDict(
    main=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=bridge_data,
        **validation_conf_toysink1_room8052
    )
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=7,
        state_dim=7,
        resnet='resnet34',
        task_id_conditioning=28,
        img_sz=[96, 128]
    )
)