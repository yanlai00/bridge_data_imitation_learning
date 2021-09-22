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
    main=(MultiDatasetLoader,
          AttrDict(
    single_task=[
        FixLenVideoDataset,
        AttrDict(
            T=31,
            data_dir=os.environ['DATA'] + '/spt_trainingdata' + '/spt_control_experiments/control/datacollection/gtruth_model_demo',
        )
    ],
    bridge_data=[
        FixLenVideoDataset,
        AttrDict(
            T=31,
            data_dir=os.environ['DATA'] + '/spt_trainingdata' + '/spt_control_experiments/control/datacollection/2cam',
        )
    ])
    ),
    finetuning=[
        FixLenVideoDataset,
        AttrDict(
            T=31,
            data_dir=os.environ['DATA'] + '/spt_trainingdata' + '/spt_control_experiments/control/datacollection/gtruth_model_demo',
        )
    ]
)

configuration = AttrDict(
    model=GCBCTransfer,
    finetuning_model=GCBCImages,
    batch_size=8,
    max_iterations=50000,
    finetuning_max_iterations=50000
)

model_config = AttrDict(
    main=AttrDict(
        shared_params=AttrDict(
            action_dim=4,
            state_dim=4,
        ),
        single_task_params=AttrDict(
            goal_cond=True
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
        sel_camera=1
    )
)
