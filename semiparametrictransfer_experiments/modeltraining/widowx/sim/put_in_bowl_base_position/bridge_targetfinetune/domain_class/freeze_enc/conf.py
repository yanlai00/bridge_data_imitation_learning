import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
# from semiparametrictransfer.models.gcbc_images_context import GCBCImagesContext
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))

from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl_base_position.bridge_targetfinetune.domain_class import conf10 as conf

configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config
model_config.finetuning.freeze_encoder = True
