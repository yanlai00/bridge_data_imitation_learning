import os
current_dir = os.path.dirname(os.path.realpath(__file__))

from semiparametrictransfer_experiments.modeltraining.widowx_sawyer.sim.put_in_bowl.bridge_targetfinetune import conf

configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config
model_config.main.robot_class_mult = 0.1
model_config.main.use_domain_onehot = True
