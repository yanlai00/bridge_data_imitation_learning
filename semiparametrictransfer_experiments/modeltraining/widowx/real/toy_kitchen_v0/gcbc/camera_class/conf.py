from semiparametrictransfer_experiments.modeltraining.widowx.real.toy_kitchen_v0.gcbc import conf
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config
model_config.main.domain_class_mult = 0.01
model_config.main.num_domains = 285
