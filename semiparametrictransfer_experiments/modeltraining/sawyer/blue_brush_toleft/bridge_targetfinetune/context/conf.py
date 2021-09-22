import os
from semiparametrictransfer.models.gcbc_images_context import GCBCImagesContext
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset
from semiparametrictransfer_experiments.modeltraining.sawyer.bridge_targetfinetune import conf

configuration = AttrDict(
    model=GCBCImagesContext,
    finetuning_model=GCBCImagesContext,
    batch_size=16,
    max_iterations=200000,
    finetuning_max_iterations=200000
)

data_config = conf.data_config
model_config = conf.model_config
