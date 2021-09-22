import os
from semiparametrictransfer.models.gcbc_images import SuccessClassifierImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from experiments.modeltraining.widowx.real.toy_kitchen_v0.dataset_lmdb import TOTAL_NUM_TASKS_ALIASING, task_name_aliasing_dict 
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset_Success_Classifier
from semiparametrictransfer.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

toykitchen2_room8052_aliasing_cam0_positive = AttrDict(
            name='toykitchen2_cam0_pos',
            random_crop=[96, 128],
            color_augmentation=0.1,
            image_size_beforecrop=[112, 144],
            data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/',
            filtering_function=[lambda dframe: dframe[(dframe['environment'] == 'toykitchen2_room8052') & (dframe['camera_index'] == 0) & (dframe['tstep_reverse'] < 2)]],
            aliasing_dict=task_name_aliasing_dict,
        )

toykitchen2_room8052_aliasing_cam0_negative = AttrDict(
            name='toykitchen2_cam0_neg',
            random_crop=[96, 128],
            color_augmentation=0.1,
            image_size_beforecrop=[112, 144],
            data_dir=os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/',
            filtering_function=[lambda dframe: dframe[(dframe['environment'] == 'toykitchen2_room8052') & (dframe['camera_index'] == 0) & (dframe['tstep_reverse'] > 1)]],
            aliasing_dict=task_name_aliasing_dict,
        )

configuration = AttrDict(
    weight_decay=0.1,
    # batch_size=32,
    dataset_normalization=False,
    delta_step_val=10,
    delta_step_control_val=10,
    delta_step_save=10,
    main=AttrDict(
        model=SuccessClassifierImages,
        max_iterations=3000,
    ),
)

data_config = AttrDict(
    main=AttrDict(
        dataclass=RandomMixingDatasetLoader,
        dataconf=AttrDict(
            dataset0=[
                LMDB_Dataset_Success_Classifier,
                toykitchen2_room8052_aliasing_cam0_positive,
                0.5,
            ],
            dataset1=[
                LMDB_Dataset_Success_Classifier,
                toykitchen2_room8052_aliasing_cam0_negative,
                0.5,
            ],
        ),
    )
)

model_config = AttrDict(
    main=AttrDict(
        resnet='resnet34',
        task_id_conditioning=TOTAL_NUM_TASKS_ALIASING,
        img_sz=[96, 128],
        pretrained_resnet=True,
        dataset_normalization=False,
    )
)
