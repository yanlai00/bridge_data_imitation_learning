from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl_base_position.bc_fromscratch import conf
from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl_base_position.datasetdef import put_in_bowl_gatorade_10k
from semiparametrictransfer.utils.general_utils import AttrDict

configuration = conf.configuration
data_config = conf.data_config
data_config.main.dataconf = AttrDict(
            name='put_in_bowl_gatorade_base0_10k',
            **put_in_bowl_gatorade_10k
        )

model_config = conf.model_config
