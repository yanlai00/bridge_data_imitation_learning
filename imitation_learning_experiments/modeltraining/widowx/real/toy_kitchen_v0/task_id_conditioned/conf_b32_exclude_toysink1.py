import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
import numpy as np
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer_experiments.modeltraining.widowx.real.toy_kitchen_v0.dataset_lmdb import *
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset, LMDB_Dataset_Pandas

configuration = AttrDict(
    main=AttrDict(
        model=GCBCImages,
        max_iterations=400000,
    ),
)


bridge_data= AttrDict(
    name='alldata',
    random_crop=[96, 128],
    color_augmentation=0.1,
    image_size_beforecrop=[112, 144],
    data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam',
    excluded_dirs=['initial_testconfig', 'cropped'],
    filtering_function=[lambda dframe: dframe[(dframe['environment'] != 'toysink1_room8052')]]
)


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
        task_id_conditioning=TOTAL_NUM_TASKS,
        img_sz=[96, 128]
    )
)