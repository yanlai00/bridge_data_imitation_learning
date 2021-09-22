import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
# from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset
# from semiparametrictransfer.data_sets.robonet_dataloader_single_timestep import FilteredRoboNetDatasetSingleTimeStep
from semiparametrictransfer.data_sets.replay_buffer import DatasetReplayBuffer
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset, LMDB_Dataset_Pandas

from semiparametrictransfer_experiments.modeltraining.widowx.real.toy_kitchen_v0.dataset_lmdb import source_data_config, validation_data_conf, bridge_data_config

configuration = AttrDict(
    batch_size=8,
    main=AttrDict(
        model=GCBCImages,
        max_iterations=150000
        # max_iterations=15
    ),
    finetuning=AttrDict(
        model=GCBCImages,
        max_iterations=150000,
        # max_iterations=15,
    )
)

data_config = AttrDict(
    main=AttrDict(
        dataclass=LMDB_Dataset_Pandas,
        dataconf=bridge_data_config,
    ),
    finetuning=AttrDict(
            dataclass=LMDB_Dataset_Pandas,
            dataconf=source_data_config,
            **validation_data_conf,
    ),
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=7,
        state_dim=7,
        resnet='resnet34',
        task_id_conditioning=15,
        img_sz=[96, 128]
    ),
    finetuning=AttrDict(
        action_dim=7,
        state_dim=7,
        resnet='resnet34',
        task_id_conditioning=15,
        img_sz=[96, 128]
    )
)
