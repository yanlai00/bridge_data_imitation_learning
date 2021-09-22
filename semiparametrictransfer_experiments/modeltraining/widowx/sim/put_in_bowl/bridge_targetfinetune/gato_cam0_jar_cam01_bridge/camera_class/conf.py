from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl.bridge_targetfinetune import conf
from semiparametrictransfer.utils.general_utils import AttrDict

configuration = conf.configuration

data_config = conf.data_config
from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl.datasetdef import mix_gato_cam0_jar_cam01
data_config.main = mix_gato_cam0_jar_cam01

model_config = conf.model_config
model_config.main = AttrDict(
        action_dim=7,
        state_dim=10,
        goal_cond=True,
        goal_state_delta_t=None,
        domain_class_mult = 0.1,
        num_domains=3
    )

