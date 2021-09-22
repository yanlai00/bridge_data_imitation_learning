import os
from semiparametrictransfer.models.gcbc_images_context import GCBCImagesContext
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer_experiments.modeltraining.sawyer.bridge_targetfinetune import conf
from semiparametrictransfer.models.discrete_model import GCBCImagesDiscrete

configuration = AttrDict(
    model=GCBCImagesDiscrete,
    finetuning_model=GCBCImagesDiscrete,
    batch_size=8,
    max_iterations=100000,
    finetuning_max_iterations=200000
)

data_config = conf.data_config
model_config = conf.model_config

