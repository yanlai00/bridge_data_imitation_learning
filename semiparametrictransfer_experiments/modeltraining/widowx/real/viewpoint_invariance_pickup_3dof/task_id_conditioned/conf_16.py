import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
import numpy as np
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.replay_buffer import DatasetReplayBuffer
from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.dataset_lmdb import bridge_data_config, validation_data_conf
# from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset
# from semiparametrictransfer.data_sets.robonet_dataloader_single_timestep import FilteredRoboNetDatasetSingleTimeStep
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset, LMDB_Dataset_Pandas

configuration = AttrDict(
    # batch_size=8,
    main=AttrDict(
        model=GCBCImages,
        max_iterations=400000,
    ),
)

def filtering_func1(data_frame):
    return data_frame[(data_frame['policy_desc'] != 'human_demo, lift random object')]

def filtering_func2(data_frame):
    return data_frame[(data_frame['policy_desc'] == 'human_demo, lift blue elephant') & (data_frame['camera_index'] == 0) |
                      (data_frame['policy_desc'] != 'human_demo, lift blue elephant')]

bridge_data= AttrDict(
    image_size_beforecrop=[56, 72],
    random_crop=[48, 64],
    color_augmentation=0.1,
    data_dir=os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww',
    filtering_function=[filtering_func1, filtering_func2]
)


data_config = AttrDict(
    main=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=bridge_data,
        **validation_data_conf
    )
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=4,
        state_dim=4,
        resnet='resnet34',
        task_id_conditioning=16,
    )
)