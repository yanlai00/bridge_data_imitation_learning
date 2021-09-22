from semiparametrictransfer_experiments.modeltraining.base_configs import conf
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.models.gcp import GCPModel
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

conf.configuration.update(AttrDict(
    model=GCPModel,
    # val_every_n=10,
    # num_epochs=1000
)
)

conf.model_config.update(AttrDict(
    normalize_dataset=os.environ['DATA'] + '/spt_trainingdata' + '/sim/tabletop-texture/noramlizing_params.pkl'
))

conf.data_config.update(AttrDict(
    # train_data_percent=0.05,
))

configuration = conf.configuration
model_config = conf.model_config
data_config = conf.data_config

