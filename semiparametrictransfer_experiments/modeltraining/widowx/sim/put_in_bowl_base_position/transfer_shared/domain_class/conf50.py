from semiparametrictransfer_experiments.modeltraining.widowx.sim.put_in_bowl_base_position.transfer_shared import conf
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config
model_config.main.shared_params.domain_class_mult = 0.1
model_config.main.shared_params.num_domains = 50