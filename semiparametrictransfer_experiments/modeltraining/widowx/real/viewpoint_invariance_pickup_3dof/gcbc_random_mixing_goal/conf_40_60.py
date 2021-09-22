from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.gcbc_random_mixing_goal import conf
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config

data_config.main.dataconf.dataset0[2] = 0.4
data_config.main.dataconf.dataset1[2] = 0.6
