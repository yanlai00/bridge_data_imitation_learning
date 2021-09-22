import os
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))

from semiparametrictransfer_experiments.modeltraining.widowx.real.toy_kitchen_v0.bridge_targetfinetune import conf

configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config
model_config.finetuning.freeze_encoder = True
