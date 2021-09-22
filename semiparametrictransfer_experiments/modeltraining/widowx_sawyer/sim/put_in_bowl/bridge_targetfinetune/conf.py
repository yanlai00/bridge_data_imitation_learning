import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer_experiments.modeltraining.widowx_sawyer.sim.put_in_bowl.datasetdef import validation_data_conf, put_in_bowl_gatorade_widow, sawyer_widowx_rand_mix
from semiparametrictransfer_experiments.modeltraining.widowx_sawyer.sim.put_in_bowl.control_conf import control_conf

configuration = AttrDict(
    batch_size=16,
    main=AttrDict(
        model=GCBCImages,
        max_iterations=150000
        # max_iterations=15
    ),
    finetuning=AttrDict(
        model=GCBCImages,
        max_iterations=150000,
        # max_iterations=15,
        control_conf=control_conf,
    )
)

data_config = AttrDict(
    main=AttrDict(
        **sawyer_widowx_rand_mix
    ),
    finetuning=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='put_in_bowl_gatorade_widow',
            **put_in_bowl_gatorade_widow,
        ),
        # validation datasets:
        **validation_data_conf
    ),
)

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
