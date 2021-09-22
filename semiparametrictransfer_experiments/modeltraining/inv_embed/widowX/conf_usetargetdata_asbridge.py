import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.models.gcbc import GCBCModel
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.dual_loader import DualVideoDataset
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer.models.gcbc_transfer import GCBCTransfer
from semiparametrictransfer.data_sets.multi_dataset_loader import MultiDatasetLoader

configuration = AttrDict(
    model=GCBCTransfer,
    finetuning_model=GCBCImages,
    batch_size=8,
    val_every_n=20,
    num_epochs=10000,
    num_finetuning_epochs=100000,
    dataset_class=DualVideoDataset,
    finetuning_dataset_class=FixLenVideoDataset
)

data_config = AttrDict(
    main=(MultiDatasetLoader, AttrDict(
        single_task=[FixLenVideoDataset,
                     AttrDict(
                         T=30,
                         color_augmentation=0.3,
                         random_crop=[48, 64],
                         image_size_beforecrop=[56, 72],
                         data_dir=os.environ['DATA'] + '/spt_trainingdata' + '/realworld/can_pushing_line',
                     )
                     ],
        bridge_data=[FixLenVideoDataset,
                     AttrDict(
                         T=30,
                         color_augmentation=0.3,
                         random_crop=[48, 64],
                         image_size_beforecrop=[56, 72],
                         data_dir=os.environ['DATA'] + '/spt_trainingdata' + '/realworld/can_pushing_line',
                     )
                     ]
    )
    ),
    finetuning=[FixLenVideoDataset,
                     AttrDict(
                         T=30,
                         color_augmentation=0.3,
                         random_crop=[48, 64],
                         image_size_beforecrop=[56, 72],
                         data_dir=os.environ['DATA'] + '/spt_trainingdata' + '/realworld/can_pushing_line',
                     )
                     ]
)


model_config = AttrDict(
    main=AttrDict(
        shared_params=AttrDict(
            action_dim=4,
            state_dim=4,
        ),
        single_task_params=AttrDict(
            # goal_cond=False,
            # sel_camera=0
        ),
        bridge_data_params=AttrDict(
            goal_cond=True,
            concatentate_cameras=True
        ),
    ),
    finetuning=AttrDict(
        action_dim=4,
        state_dim=4,
        sel_camera=1
    )
)

