from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.random_mixing_task_id import conf
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config

data_config.main.dataconf.dataset0[2] = 0.1
data_config.main.dataconf.dataset1[2] = 0.9
