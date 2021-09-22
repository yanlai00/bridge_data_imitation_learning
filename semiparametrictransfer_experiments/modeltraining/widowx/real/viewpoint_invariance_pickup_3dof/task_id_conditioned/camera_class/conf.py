from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.task_id_conditioned import conf
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config
model_config.main.domain_class_mult = 0.1
model_config.main.num_domains = 195
