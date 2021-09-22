import os
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.models.gcbc_transfer import GCBCTransfer
from semiparametrictransfer.models.gcbc_images_context import GCBCImagesContext

from semiparametrictransfer_experiments.modeltraining.sawyer.transfer import conf

data_config = conf.data_config

configuration = AttrDict(
    model=GCBCTransfer,
    finetuning_model=GCBCImagesContext,
    batch_size=8,
    max_iterations=150000,
    finetuning_max_iterations=150000
)

model_config = conf.model_config
model_config.update(AttrDict(
        child_model_class=GCBCImagesContext
    )
)


