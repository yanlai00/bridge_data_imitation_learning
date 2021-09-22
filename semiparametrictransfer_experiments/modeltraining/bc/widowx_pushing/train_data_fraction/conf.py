import os
current_dir = os.path.dirname(os.path.realpath(__file__))

from semiparametrictransfer_experiments.modeltraining.bc.widowx_pushing import conf

conf.configuration.update({
    'num_epochs':100000,
    # 'resume_checkpoint': os.environ['EXP'] + '/spt_experiments' + '/modeltraining/bc/widowx_pushing/can_freeze_nopretrained/weights/weights_ep9995.pth'
})
configuration = conf.configuration

conf.data_config.update({
    'train_data_fraction': 0.1,
})
data_config = conf.data_config

model_config = conf.model_config