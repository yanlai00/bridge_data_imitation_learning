import os
from imitation_learning.models.gcbc_images import GCBCImages
import numpy as np
from imitation_learning.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from imitation_learning_experiments.dataset_lmdb import TOTAL_NUM_TASKS_ALIASING, bridge_data_config_aliasing, task_name_aliasing_dict
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset_Pandas

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

data_config = AttrDict(
    main=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=bridge_data_config_aliasing,
        **validation_conf_toykitchen1_aliasing
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