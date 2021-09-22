import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
import numpy as np
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.replay_buffer import DatasetReplayBuffer
from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.dataset_lmdb import source_task_config, validation_data_conf
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset, LMDB_Dataset_Pandas

configuration = AttrDict(
    batch_size=8,
    main=AttrDict(
        model=GCBCImages,
        max_iterations=200000,
    ),
)


source = AttrDict(
        name='source_task_randcam',
        image_size_beforecrop=[56, 72],
        random_crop=[48, 64],
        color_augmentation=0.1,
        data_dir=[
            os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant_no_distractors/lmdb',
            os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant/lmdb',
        ],
)

data_config = AttrDict(
    main=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=source,
        # validation datasets:
        # **validation_data_conf,
    )
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=4,
        resnet='resnet34',
    )
)