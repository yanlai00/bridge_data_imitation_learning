from semiparametrictransfer_experiments.modeltraining.base_configs import conf
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.models.gcbc import GCBCModel
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

conf.configuration.update(AttrDict(
    model=GCBCModel)
)

conf.model_config.update(AttrDict(
    goal_cond=False
))

configuration = conf.configuration
model_config = conf.model_config
data_config = conf.data_config
