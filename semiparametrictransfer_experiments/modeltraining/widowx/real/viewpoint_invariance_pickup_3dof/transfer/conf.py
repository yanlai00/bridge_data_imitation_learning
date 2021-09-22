import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict

current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.replay_buffer import MultiDatasetReplayBuffer
from semiparametrictransfer.data_sets.robonet_dataloader_single_timestep import FilteredRoboNetDatasetSingleTimeStep
from semiparametrictransfer.data_sets.multi_dataset_loader import MultiDatasetLoader
from semiparametrictransfer.models.gcbc_transfer import GCBCTransfer
from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset
from semiparametrictransfer.data_sets.replay_buffer import DatasetReplayBuffer
from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.datasetdef_hdf5 import source_task_config, \
    validation_data_conf, bridge_data_config

data_config = AttrDict(
    main=AttrDict(
        dataclass=MultiDatasetReplayBuffer,
        dataconf=AttrDict(
            single_task=AttrDict(
                dataconf=AttrDict(
                    dataset_type=FilteredRoboNetDataset,
                    **source_task_config,
                    # debug=True  #############
                ),
            ),
            bridge_data=AttrDict(
                dataconf=AttrDict(
                    dataset_type=FilteredRoboNetDataset,
                    **bridge_data_config,
                    # debug=True,  #############
                )
            )
        ),
        **validation_data_conf
    ),
)



configuration = AttrDict(
    batch_size=8,
    main=AttrDict(
        model=GCBCTransfer,
        max_iterations=150000,
        # max_iterations=300000,
    ),
)

model_config = AttrDict(
    main=AttrDict(
        # use_grad_reverse=False,  #############
        shared_params=AttrDict(
            action_dim=4,
            resnet='resnet34',
        ),
        single_task_params=AttrDict(
            model_loss_mult=0.1
        ),
        bridge_data_params=AttrDict(
            goal_cond=True,
            # use_grad_reverse=False  ############
        ),
    ),
)
