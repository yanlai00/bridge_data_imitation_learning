import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.multi_dataset_loader import MultiDatasetLoader
from semiparametrictransfer.models.gcbc_transfer import GCBCTransfer
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer.models.gcbc_images_context import GCBCImagesContext

from semiparametrictransfer_experiments.modeltraining.sawyer.pickup_drill.transfer.camera_class import conf


configuration = conf.configuration
configuration.finetuning_model = GCBCImagesContext

data_config = conf.data_config

model_config = conf.model_config
model_config.main.child_model_class = GCBCImagesContext

