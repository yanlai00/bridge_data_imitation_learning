import os
current_dir = os.path.dirname(os.path.realpath(__file__))

from semiparametrictransfer_experiments.modeltraining.inv_embed.widowX import conf

conf.configuration.update({
    'resume_checkpoint': os.environ['EXP'] + '/spt_experiments' + '/modeltraining/inv_embed/widowX/domain_spec/weights/weights_ep4995.pth',
    'num_finetuning_epochs':100000,
})
configuration = conf.configuration

conf.data_config.update({
    'train_data_fraction': 0.1,
})
data_config = conf.data_config

model_config = conf.model_config

