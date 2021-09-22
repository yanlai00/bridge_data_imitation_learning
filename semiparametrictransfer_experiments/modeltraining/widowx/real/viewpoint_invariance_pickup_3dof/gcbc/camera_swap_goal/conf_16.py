from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.gcbc import conf
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config
model_config.main.domain_class_mult = 0.01
model_config.main.num_domains = 677
data_config.main.dataconf.get_final_image_from_same_traj = False
model_config.main.separate_classifier = True
configuration.batch_size = 16
