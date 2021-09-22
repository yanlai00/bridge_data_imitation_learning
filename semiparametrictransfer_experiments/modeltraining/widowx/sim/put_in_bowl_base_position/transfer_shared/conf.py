import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.multi_dataset_loader import MultiDatasetLoader
from semiparametrictransfer.models.gcbc_transfer import GCBCTransfer
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl_base_position.datasetdef import put_in_bowl_gatorade, put_in_bowl_rand_base
from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl_base_position.bc_fromscratch.conf import control_conf
from semiparametrictransfer.data_sets.replay_buffer import MultiDatasetReplayBuffer

data_config = AttrDict(
    main=AttrDict(
        dataclass=MultiDatasetLoader,
        dataconf=AttrDict(
            single_task=AttrDict(
                dataclass=FixLenVideoDataset,
                dataconf=AttrDict(
                    name='put_in_bowl_gatorade_base0',
                    **put_in_bowl_gatorade,
                ),
            ),
            bridge_data=AttrDict(
                dataclass=FixLenVideoDataset,
                dataconf=AttrDict(
                    name='put_in_bowl_rand_base',
                    **put_in_bowl_rand_base
                ),
            ),
        ),
    ),
)

configuration = AttrDict(
    batch_size=16,
    main=AttrDict(
        model=GCBCTransfer,
        max_iterations=150000,
        control_conf=control_conf
    ),
)

model_config = AttrDict(
    main=AttrDict(
        shared_params=AttrDict(
            action_dim=7,
            state_dim=10,
            shared_classifier=True,
            # encoder_embedding_size=128,
            # encoder_spatial_softmax=False,
        ),
        single_task_params=AttrDict(
            model_loss_mult=0.1
        ),
        bridge_data_params=AttrDict(
            goal_cond=True,
            goal_state_delta_t=None,
        ),
    ),
)
