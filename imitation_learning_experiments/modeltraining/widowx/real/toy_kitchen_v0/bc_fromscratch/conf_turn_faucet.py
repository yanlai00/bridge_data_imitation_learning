from imitation_learning_experiments.modeltraining.widowx.real.toy_kitchen_v0.bc_fromscratch import conf
import os
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config

data_config.main.dataconf.data_dir = [
            os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/turn_faucet_front_to_left/lmdb',
            ]
