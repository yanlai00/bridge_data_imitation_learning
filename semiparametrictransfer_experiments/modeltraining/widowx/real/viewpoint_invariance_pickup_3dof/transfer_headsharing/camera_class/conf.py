from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.transfer_headsharing import conf
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config
model_config.main.single_task_params.domain_class_mult = 0.01
model_config.main.single_task_params.num_domains = 417
