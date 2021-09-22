import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset
from semiparametrictransfer.data_sets.robonet_dataloader_single_timestep import FilteredRoboNetDatasetSingleTimeStep
from semiparametrictransfer.data_sets.replay_buffer import DatasetReplayBuffer

from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup.datasetdef import source_task_config, validation_data_conf, bridge_data_config

configuration = AttrDict(
    batch_size=8,
    main=AttrDict(
        model=GCBCImages,
        max_iterations=150000
        # max_iterations=15
    ),
    finetuning=AttrDict(
        model=GCBCImages,
        max_iterations=150000,
        # max_iterations=15,
    )
)

data_config = AttrDict(
    main=AttrDict(
        # dataclass=DatasetReplayBuffer,
        dataclass = FilteredRoboNetDatasetSingleTimeStep,
        dataconf = bridge_data_config,
),
finetuning=AttrDict(
        dataclass = FilteredRoboNetDatasetSingleTimeStep,
        dataconf = source_task_config,
        # validation datasets:
        **validation_data_conf,
    ),
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=7,
        goal_cond=True,
        goal_state_delta_t=None,
        img_sz=[96, 128],
        resnet='resnet34'
    ),
    finetuning=AttrDict(
        resnet='resnet34',
        action_dim=7,
        img_sz=[96, 128]
    )
)
