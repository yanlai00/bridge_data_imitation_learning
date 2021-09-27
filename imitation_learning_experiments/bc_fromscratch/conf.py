import os
from imitation_learning.models.gcbc_images import GCBCImages
import numpy as np
from imitation_learning.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from imitation_learning_experiments.dataset_lmdb import long_distance_trajs
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset_Pandas

configuration = AttrDict(
    main=AttrDict(
        model=GCBCImages,
        max_iterations=400000,
    ),
)

sponge_wipe = AttrDict(
            name='sponge_wipe',
            random_crop=[96, 128],
            color_augmentation=0.1,
            image_size_beforecrop=[112, 144],
            data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/realkitchen1_counter/',
            filtering_function=[lambda dframe: dframe[(dframe['policy_desc'] == 'human_demo, pick up sponge and wipe plate')]],
        )

data_config = AttrDict(
    main=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=sponge_wipe,
    )
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=7,
        state_dim=7,
        resnet='resnet34',
        img_sz=[96, 128]
    )
)