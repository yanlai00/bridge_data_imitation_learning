import os
from imitation_learning.models.gcbc_images import GCBCImages
import numpy as np
from imitation_learning.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from experiments.modeltraining.widowx.real.toy_kitchen_v0.dataset_lmdb import task_name_aliasing_dict
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset_Pandas
from imitation_learning.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

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
            aliasing_dict=task_name_aliasing_dict,
        )

validation_sponge_wipe = AttrDict(
    val0=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=sponge_wipe
    ),
)

excl_real_kitchen_and_toolchest= AttrDict(
    name='excl_real_kitchen_and_toolchest',
    random_crop=[96, 128],
    color_augmentation=0.1,
    image_size_beforecrop=[112, 144],
    data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam',
    excluded_dirs=['initial_testconfig', 'cropped', 'initial_test_config', 'put_eggplant_in_pot_or_pan', 'realkitchen1_counter', 'realkitchen1_dishwasher', 'tool_chest'],
    aliasing_dict=task_name_aliasing_dict,
)

source_data = sponge_wipe
validation_data = validation_sponge_wipe
bridge_data = excl_real_kitchen_and_toolchest

data_config = AttrDict(
    main=AttrDict(
        dataclass=RandomMixingDatasetLoader,
        dataconf=AttrDict(
            dataset0=[
                LMDB_Dataset_Pandas,
                source_data,
                0.1
            ],
            dataset1=[
                LMDB_Dataset_Pandas,
                bridge_data,
                0.9
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
        task_id_conditioning=72,
        img_sz=[96, 128]
    )
)