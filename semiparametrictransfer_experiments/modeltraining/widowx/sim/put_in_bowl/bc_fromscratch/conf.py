import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
import numpy as np
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
import copy
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl.datasetdef import validation_data_conf, put_in_bowl_gatorade
from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl.control_conf import config as control_config
from visual_mpc.envs.pybullet_envs.container_env import Widow250Container
from semiparametrictransfer.data_sets.replay_buffer import DatasetReplayBuffer

from semiparametrictransfer_experiments.control.widowx.sim.conf import DEFAULT_CAMERA

control_conf_cam0 = copy.deepcopy(control_config)
control_conf_cam1 = copy.deepcopy(control_config)
env_params = AttrDict(camera_settings=[AttrDict(**DEFAULT_CAMERA)], # using default camera with 180
                      gui=False,
                      target_object_setting='gatorade'
            )
control_conf_cam0['agent']['env'] = (Widow250Container, env_params)

env_params = AttrDict(camera_settings=[AttrDict(**DEFAULT_CAMERA, yaw=200)],
                      gui=False,
                      target_object_setting='gatorade'
                      )
control_conf_cam1['agent']['env'] = (Widow250Container, env_params)

control_conf = AttrDict(
        pickgatorade_cam0=control_conf_cam0,
        pickgatorade_cam1=control_conf_cam1
    )

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
            name='put_in_bowl_gatorade_cam0',
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