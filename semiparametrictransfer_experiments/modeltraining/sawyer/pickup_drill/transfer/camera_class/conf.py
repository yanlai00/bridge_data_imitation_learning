import os
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer_experiments.modeltraining.sawyer.pickup_drill.transfer import conf

configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config
model_config.main.bridge_data_params.domain_class_mult = 1

