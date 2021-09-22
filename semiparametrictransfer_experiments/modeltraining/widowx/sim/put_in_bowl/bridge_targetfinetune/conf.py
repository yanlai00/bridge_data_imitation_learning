import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl.datasetdef import validation_data_conf, put_in_bowl_gatorade, put_in_bowl
from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl.bc_fromscratch.conf import control_conf
from semiparametrictransfer.data_sets.replay_buffer import DatasetReplayBuffer

configuration = AttrDict(
    batch_size=16,
    main=AttrDict(
        model=GCBCImages,
        max_iterations=150000
        # max_iterations=2100
    ),
    finetuning=AttrDict(
        model=GCBCImages,
        max_iterations=150000,
        # max_iterations=2100,
        control_conf=control_conf,
    )
)

data_config = AttrDict(
    main=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='put_in_bowl',
            **put_in_bowl
        )
    ),
    finetuning=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='put_in_bowl_gatorade_cam0',
            **put_in_bowl_gatorade,
        ),
        # validation datasets:
        **validation_data_conf
    ),
)

# data_config = AttrDict(
#     main=AttrDict(
#         dataclass=DatasetReplayBuffer,
#         dataconf=AttrDict(
#             name='put_in_bowl',
#             dataset_type=FixLenVideoDataset,
#             max_train_examples=400,
#             **put_in_bowl
#         )
#     ),
#     finetuning=AttrDict(
#         dataclass=DatasetReplayBuffer,
#         dataconf=AttrDict(
#             name='put_in_bowl_gatorade_cam0',
#             dataset_type=FixLenVideoDataset,
#             max_train_examples=300,
#             **put_in_bowl_gatorade,
#         ),
#         # validation datasets:
#         **validation_data_conf
#     ),
# )

model_config = AttrDict(
    main=AttrDict(
        action_dim=7,
        state_dim=10,
        goal_cond=True,
        goal_state_delta_t=None,
    ),
    finetuning=AttrDict(
        action_dim=7,
        state_dim=10,
    )
)
