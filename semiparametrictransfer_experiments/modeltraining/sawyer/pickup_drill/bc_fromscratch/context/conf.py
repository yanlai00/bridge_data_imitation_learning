import os
from semiparametrictransfer.models.gcbc_images_context import GCBCImagesContext
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer_experiments.modeltraining.sawyer.pickup_drill.bc_fromscratch import conf

configuration = conf.configuration
configuration.update(AttrDict(
    model=GCBCImagesContext,
    finetuning_model=GCBCImagesContext,
))

data_config = conf.data_config

model_config = conf.model_config
model_config.main.num_context = 1
