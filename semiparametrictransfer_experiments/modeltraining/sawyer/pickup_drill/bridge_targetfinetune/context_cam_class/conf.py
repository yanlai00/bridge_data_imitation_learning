import os
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.models.gcbc_images_context import GCBCImagesContext
from semiparametrictransfer_experiments.modeltraining.sawyer.pickup_drill.bridge_targetfinetune.camera_class import conf

configuration = conf.configuration
configuration.update(AttrDict(
        model=GCBCImagesContext,
        finetuning_model=GCBCImagesContext,
    ))
data_config = conf.data_config
model_config = conf.model_config
