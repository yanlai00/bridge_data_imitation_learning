import os
from bridgedata.models.gcbc_images import GCBCImages
from bridgedata.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from bridgedata_experiments.dataset_lmdb import task_name_aliasing_dict
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset_Pandas
from bridgedata.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

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
            excluded_dirs=['initial_testconfig', 'cropped', 'initial_test_config', 'put_eggplant_in_pot_or_pan', 'tool_chest', 'from_basket_to_tray', 'realkitchen1'],
            filtering_function=[lambda dframe: dframe[(dframe['environment'] == 'toykitchen1') | (dframe['environment'] == 'toykitchen_bww')]],
            aliasing_dict=task_name_aliasing_dict,
        )

validation_conf_toykitchen1_aliasing = AttrDict(
    val0=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=bridge_data_config_kitchen1_aliasing
    ),
)

bridge_data_config = AttrDict(
    name='alldata',
    random_crop=[96, 128],
    color_augmentation=0.1,
    image_size_beforecrop=[112, 144],
    data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam',
    excluded_dirs=['initial_testconfig', 'cropped', 'initial_test_config', 'put_eggplant_in_pot_or_pan', 'tool_chest', 'from_basket_to_tray', 'realkitchen1'],
    aliasing_dict=task_name_aliasing_dict,
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
        task_id_conditioning=70,
        img_sz=[96, 128]
    )
)