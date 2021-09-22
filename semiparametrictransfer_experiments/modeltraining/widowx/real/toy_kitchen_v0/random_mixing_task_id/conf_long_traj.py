import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
import numpy as np
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from experiments.modeltraining.widowx.real.toy_kitchen_v0.dataset_lmdb import task_name_aliasing_dict
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset_Pandas
from semiparametrictransfer.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

configuration = AttrDict(
    main=AttrDict(
        model=GCBCImages,
        max_iterations=400000,
    ),
)

long_distance_trajs = AttrDict(
            name='long_distance_trajs',
            random_crop=[96, 128],
            color_augmentation=0.1,
            image_size_beforecrop=[112, 144],
            data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/',
            excluded_dirs=['initial_testconfig', 'cropped', 'initial_test_config', 'put_eggplant_in_pot_or_pan', 'tool_chest', 'from_basket_to_tray'],
            filtering_function=[lambda dframe: dframe[(dframe['month'] > 7 ) | ((dframe['month'] == 7) & (dframe['day'] >= 5))]],
            aliasing_dict=task_name_aliasing_dict,
        )

validation_long_distance_trajs = AttrDict(
    val0=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=long_distance_trajs
    ),
)

bridge_data_config = AttrDict(
    name='alldata',
    random_crop=[96, 128],
    color_augmentation=0.1,
    image_size_beforecrop=[112, 144],
    data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam',
    excluded_dirs=['initial_testconfig', 'cropped', 'initial_test_config', 'put_eggplant_in_pot_or_pan', 'tool_chest', 'from_basket_to_tray'],
    aliasing_dict=task_name_aliasing_dict,
)

source_data = long_distance_trajs
validation_data = validation_long_distance_trajs

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
                bridge_data_config,
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
        task_id_conditioning=66,
        img_sz=[96, 128]
    )
)