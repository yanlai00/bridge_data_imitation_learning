import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.multi_dataset_loader import MultiDatasetLoader
from semiparametrictransfer.models.gcbc_transfer import GCBCTransfer
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl.datasetdef import validation_data_conf, put_in_bowl_gatorade, put_in_bowl
from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl.bc_fromscratch.conf import control_conf
from semiparametrictransfer.data_sets.replay_buffer import MultiDatasetReplayBuffer

data_config = AttrDict(
    main=AttrDict(
        dataclass=MultiDatasetLoader,
        dataconf=AttrDict(
            # n_worker=0,  ###################
            single_task=AttrDict(
                dataclass=FixLenVideoDataset,
                dataconf=AttrDict(
                    name='put_in_bowl_gatorade_cam0',
                    **put_in_bowl_gatorade,
                ),
            ),
            bridge_data=AttrDict(
                dataclass=FixLenVideoDataset,
                dataconf=AttrDict(
                    name='put_in_bowl',
                    **put_in_bowl,
                ),
            ),
            classifier_validation=AttrDict(   # for validation purposes to train classifier to see if embeddings are domain-invariant
                dataclass=FixLenVideoDataset,
                dataconf=AttrDict(
                    name='classifier_validation',
                    **put_in_bowl_gatorade,
                    sel_camera='random',
                ),
            ),
        ),
        **validation_data_conf
    ),
)

# data_config = AttrDict(
#     main=AttrDict(
#         dataclass=MultiDatasetReplayBuffer,
#         dataconf=AttrDict(
#             single_task=AttrDict(
#                 dataconf=AttrDict(
#                     dataset_type=FixLenVideoDataset,
#                     name='put_in_bowl_gatorade_cam0',
#                     **put_in_bowl_gatorade,
#                 ),
#             ),
#             bridge_data=AttrDict(
#                 dataconf=AttrDict(
#                     dataset_type=FixLenVideoDataset,
#                     name='put_in_bowl',
#                     **put_in_bowl
#                 ),
#             ),
#         ),
#         **validation_data_conf
#     ),
# )


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
        datasource_class_mult=0.1,  ##########
        shared_params=AttrDict(
            action_dim=7,
            state_dim=10,
        ),
        single_task_params=AttrDict(
            model_loss_mult=0.1
        ),
        bridge_data_params=AttrDict(
            goal_cond=True,
            goal_state_delta_t=None,
        ),
        classifier_validation_params=AttrDict(
            freeze_encoder=True,
            domain_class_mult=1.,
            num_domains=3,
            use_grad_reverse=False
        )
    ),
)
