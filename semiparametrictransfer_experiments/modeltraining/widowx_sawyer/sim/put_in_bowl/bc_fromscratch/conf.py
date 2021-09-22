import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
import numpy as np
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
import copy
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer_experiments.modeltraining.widowx_sawyer.sim.put_in_bowl.datasetdef import validation_data_conf, put_in_bowl_gatorade_widow
from semiparametrictransfer_experiments.modeltraining.widowx_sawyer.sim.put_in_bowl.control_conf import control_conf

configuration = AttrDict(
    batch_size=16,
    main=AttrDict(
        model=GCBCImages,
        max_iterations=200000,
        control_conf=control_conf
    ),
)

data_config = AttrDict(
    main=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='put_in_bowl_gatorade_widow',
            **put_in_bowl_gatorade_widow
        ),
        # validation datasets:
        **validation_data_conf,
    )
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=7,
        state_dim=10,
    )
)