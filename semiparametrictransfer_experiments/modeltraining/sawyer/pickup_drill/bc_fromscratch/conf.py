import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset

configuration = AttrDict(
    model=GCBCImages,
    finetuning_model=GCBCImages,
    batch_size=16,
    max_iterations=200000,
)

from semiparametrictransfer_experiments.modeltraining.sawyer.pickup_drill.datasetdef import validation_data_conf, pick_drill_wood_noclutter

data_config = AttrDict(
    main=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='pick_drill_wood_noclutter',
            **pick_drill_wood_noclutter
        ),
        # validation datasets:
        **validation_data_conf
    )
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=4,
        state_dim=4,
        # encoder_spatial_softmax=False,
        # encoder_embedding_size=128,
        resnet='resnet34',
    )
)