from semiparametrictransfer_experiments.modeltraining.base_configs import conf
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.models.trajfollwmodel import TrajFollowModel

import os

current_dir = os.path.dirname(os.path.realpath(__file__))

conf.configuration.update(AttrDict(
    model=TrajFollowModel)
)

conf.model_config.update(AttrDict(
))

configuration = conf.configuration
model_config = conf.model_config
data_config = conf.data_config