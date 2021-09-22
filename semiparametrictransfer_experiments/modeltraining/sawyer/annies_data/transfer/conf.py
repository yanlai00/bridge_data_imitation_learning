import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.models.gcbc import GCBCModel

current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.multi_dataset_loader import MultiDatasetLoader
from semiparametrictransfer.models.gcbc_transfer import GCBCTransfer
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset

data_config = AttrDict(
    main=(MultiDatasetLoader, AttrDict(
        single_task=(FixLenVideoDataset,
                     AttrDict(
                         name='annies_data',
                         T=15,
                         color_augmentation=0.3,
                         data_dir= os.environ['DATA'] + '/sawyerdata/annies_data/kinesthetic_demos',
                     )),
        bridge_data=(FilteredRoboNetDataset,
                     AttrDict(
                         name='robonet_sawyer',
                         T=15,
                         robot_list=['sawyer'],
                         # train_val_split=[0.9, 0.05, 0.05],
                         train_val_split=[0.8, 0.1, 0.1],
                         color_augmentation=0.3,
                         random_crop=True,
                         data_dir=os.environ['DATA'] + '/misc_datasets/robonet/robonet/hdf5'
                     ))
    )),
    finetuning=(FixLenVideoDataset,
                 AttrDict(
                     name='annies_data',
                     T=15,
                     color_augmentation=0.3,
                     data_dir= os.environ['DATA'] + '/sawyerdata/annies_data/kinesthetic_demos',
                 ))
)

configuration = AttrDict(
    model=GCBCTransfer,
    finetuning_model=GCBCImages,
    batch_size=8,
    max_iterations=300000,
    finetuning_max_iterations=200000
)

model_config = AttrDict(
    main=AttrDict(
        shared_params=AttrDict(
            action_dim=4,
            state_dim=4,
            # resnet='resnet34'
        ),
        single_task_params=AttrDict(
            goal_cond=True,
            goal_state_delta_t=None,
        ),
        bridge_data_params=AttrDict(
            goal_cond=True,
            concatentate_cameras=True,
        ),
    ),
    finetuning=AttrDict(
        action_dim=4,
        state_dim=4,
        goal_cond=True,
        goal_state_delta_t=None,
        # resnet='resnet34'
    )
)
