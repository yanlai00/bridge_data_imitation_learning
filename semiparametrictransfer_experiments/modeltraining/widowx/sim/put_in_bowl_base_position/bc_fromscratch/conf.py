import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
import numpy as np
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
import copy
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl_base_position.datasetdef import put_in_bowl_gatorade
from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl_base_position.control_conf import config as control_config
from visual_mpc.envs.pybullet_envs.container_env import Widow250Container
from semiparametrictransfer.data_sets.replay_buffer import DatasetReplayBuffer

control_conf_base0 = copy.deepcopy(control_config)
control_conf_base1 = copy.deepcopy(control_config)
env_params = AttrDict(gui=False, # default base
                      target_object_setting='gatorade'
                      )
control_conf_base0['agent']['env'] = (Widow250Container, env_params)

env_params = AttrDict(base_position=AttrDict(base_x=0.7, base_y=-0.1, base_z=-0.4), # unseen base positon
                      gui=False,
                      target_object_setting='gatorade'
                      )
control_conf_base1['agent']['env'] = (Widow250Container, env_params)

control_conf = AttrDict(
        pickgatorade_base0=control_conf_base0,
        pickgatorade_base1=control_conf_base1
    )

configuration = AttrDict(
    # batch_size=16,
    batch_size=8,
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
            name='put_in_bowl_gatorade_base0',
            **put_in_bowl_gatorade
        ),
        # validation datasets:
        # **validation_data_conf,
    )
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=7,
        state_dim=10,
        # encoder_spatial_softmax=False,
        # encoder_embedding_size=128,
    )
)