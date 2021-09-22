import torch
import numpy as np
import torch.nn as nn
from semiparametrictransfer.utils.general_utils import AttrDict, trch2npy
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.models.base_model import BaseModel
from semiparametrictransfer.models.utils.layers import BaseProcessingNet

class GCBCTransferHeadsharing(BaseModel):
    """Semi parametric transfer model"""

    def __init__(self, overrideparams, logger=None):
        super().__init__(overrideparams, logger)
        self._hp = self._default_hparams()
        self._override_defaults(overrideparams)  # override defaults with config file

        self._hp.shared_params.batch_size = self._hp.batch_size
        self._hp.shared_params.device = self._hp.device
        self._logger = logger
        self.build_network()

    def set_dataset_sufix(self, hp):
        if hasattr(hp, 'name'):
            self.dataset_sufix = hp.name
            self.single_task_model.dataset_sufix = hp.name # used for validation run
        else:
            self.dataset_sufix = 'multi_dataset'
            self.dataset_hp = hp

    def _default_hparams(self):
        default_dict = AttrDict(
            shared_params=None,
            single_task_params=None,
            child_model_class=GCBCImages,
        )
        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def build_network(self):
        self.build_single_task_model()

    def build_single_task_model(self):
        single_task_params = AttrDict()
        single_task_params.update(self._hp.shared_params)
        single_task_params.update(self._hp.single_task_params)
        single_task_params.data_conf = self._hp.data_conf.single_task.dataconf
        self.single_task_model = self._hp.child_model_class(single_task_params, self._logger)
        self._hp.single_task_params = self.single_task_model._hp

    def forward(self, inputs):
        out = AttrDict()
        if 'single_task' not in inputs:
            out.validation_run = self.single_task_model.forward(inputs)
            return out
        inputs.single_task.global_step = inputs.global_step
        inputs.single_task.max_iterations = inputs.max_iterations
        inputs.bridge_data.global_step = inputs.global_step
        inputs.bridge_data.max_iterations = inputs.max_iterations
        inputs.single_task.goal_embedding = torch.zeros(self._hp.batch_size, self.single_task_model.embedding_size).to(self._hp.device)
        out.single_task = self.single_task_model.forward(inputs.single_task)
        out.bridge_data = self.single_task_model.forward(inputs.bridge_data)
        return out

    def loss(self, model_input, model_output):
        if 'validation_run' in model_output:
            return self.single_task_model.loss(model_input, model_output.validation_run)
        losses = AttrDict()
        losses_single_task = self.single_task_model.loss(model_input.single_task, model_output.single_task, compute_total_loss=False)
        losses_bridge_data = self.single_task_model.loss(model_input.bridge_data, model_output.bridge_data, compute_total_loss=False)
        for k, v in losses_single_task.items():
            losses['single_task_' + k] = (v[0], v[1]*self.single_task_model._hp.model_loss_mult)
        for k, v in losses_bridge_data.items():
            losses['bridge_data_' + k] = (v[0], v[1])

        # compute total loss
        losses.total_loss = torch.stack([l[0] * l[1] for l in losses.values()]).sum()
        return losses

    def _log_outputs(self, model_output, inputs, losses, step, phase):
        if 'validation_run' in model_output:
            assert phase == 'val'
            self.single_task_model._log_outputs(model_output.validation_run, inputs, losses, step, phase)
            return
        model_output.single_task = self.single_task_model.forward(inputs.single_task)
        self.single_task_model._hp.goal_cond = False  # to prevent it from visualizing goal-images
        self.single_task_model._log_outputs(model_output.single_task, inputs.single_task, losses, step, phase, override_sufix=self.dataset_hp['single_task'].name)
        self.single_task_model._hp.goal_cond = True
        model_output.bridge_data = self.single_task_model.forward(inputs.bridge_data)
        self.single_task_model._log_outputs(model_output.bridge_data, inputs.bridge_data, losses, step, phase, override_sufix=self.dataset_hp['bridge_data'].name)
