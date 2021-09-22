import os
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.models.gcbc_transfer import GCBCTransfer
from semiparametrictransfer.models.gcbc_images_context import GCBCImagesContext

from semiparametrictransfer_experiments.modeltraining.sawyer.pickup_drill.transfer import conf

data_config = conf.data_config

configuration = conf.configuration
configuration.finetuning_model = GCBCImagesContext

model_config = conf.model_config
model_config.main.child_model_class=GCBCImagesContext


