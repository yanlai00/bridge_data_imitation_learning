from semiparametrictransfer_experiments.modeltraining.widowx_sawyer.sim.put_in_bowl.transfer import conf
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config
model_config.main.bridge_data_params.robot_class_mult = 0.1
model_config.main.bridge_data_params.use_domain_onehot = True
