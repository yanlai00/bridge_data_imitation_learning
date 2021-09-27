import os
from imitation_learning.models.gcbc_images import GCBCImages
# from imitation_learning.models.gcbc_images_context import GCBCImagesContext
from imitation_learning.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))

from imitation_learning_experiments.modeltraining.widowx.real.toy_kitchen_v0.bc_fromscratch import conf

configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config
model_config.main.domain_class_mult = 0.01
model_config.main.num_domains = 35
