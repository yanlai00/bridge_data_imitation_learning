from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.transfer_lmdbloader import conf
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config

model_config.main.datasource_class_mult = 0.01
model_config.main.bridge_data_params.domain_class_mult = 0.01
model_config.main.bridge_data_params.num_domains = 175
# model_config.main.bridge_data_params.num_domains = 135
