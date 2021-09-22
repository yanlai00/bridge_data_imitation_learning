import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.multi_dataset_loader import MultiDatasetLoader
from semiparametrictransfer.models.gcbc_transfer import GCBCTransfer
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset

blue_brush_to_left = AttrDict(
    color_augmentation=0.3,
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/sawyerdata/annies_data/blue_brush_toleft/clone'
)

data_config = AttrDict(
    main=AttrDict(
        dataclass=MultiDatasetLoader,
        dataconf=AttrDict(
            single_task=(FixLenVideoDataset,
                         AttrDict(name='blue_brush_toleft', **blue_brush_to_left)
            ),
            bridge_data=(FilteredRoboNetDataset,
                         AttrDict(
                             name='robonet_sawyer',
                             T=15,
                             robot_list=['sawyer'],
                             # train_val_split=[0.8, 0.1, 0.1],
                             train_val_split=[0.95, 0.025, 0.025],
                             color_augmentation=0.3,
                             random_crop=True,
                             # data_dir=os.environ['DATA'] + '/misc_datasets/robonet/robonet_sampler/hdf5'
                             data_dir=os.environ['DATA'] + '/misc_datasets/robonet/robonet/hdf5'
                        )
            ),
        ),
        val0=AttrDict(
            dataclass=FixLenVideoDataset,
            dataconf=AttrDict(name='blue_brush_toleft', **blue_brush_to_left)
        ),
        val1=AttrDict(
            dataclass=FixLenVideoDataset,
            dataconf=AttrDict(
                name='blue_brush_toleft_cam1',
                sel_camera=1,
                **blue_brush_to_left
            )
        )
    ),
    finetuning=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(name='blue_brush_toleft', **blue_brush_to_left),
        val0=AttrDict(
            dataclass=FixLenVideoDataset,
            dataconf=AttrDict(
                name='blue_brush_toleft_cam1',
                sel_camera=1,
                **blue_brush_to_left
            )
        )
    )
)

configuration = AttrDict(
    model=GCBCTransfer,
    finetuning_model=GCBCImages,
    batch_size=8,
    max_iterations=200000,
    finetuning_max_iterations=300000
)

model_config = AttrDict(
    main=AttrDict(
        datasource_class_mult=0.1,
        shared_params=AttrDict(
            action_dim=4,
            state_dim=4,
            resnet='resnet34'
        ),
        single_task_params=AttrDict(
            goal_cond=True,
            goal_state_delta_t=None,
        ),
        bridge_data_params=AttrDict(
            goal_cond=True,
        ),
    ),
    finetuning=AttrDict(
        action_dim=4,
        state_dim=4,
        goal_cond=True,
        goal_state_delta_t=None,
        resnet='resnet34'
    )
)
