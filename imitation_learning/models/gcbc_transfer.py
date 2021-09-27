import torch
import numpy as np
import torch.nn as nn
from imitation_learning.utils.general_utils import AttrDict, trch2npy
from imitation_learning.models.gcbc_images import GCBCImages
from imitation_learning.models.base_model import BaseModel
from imitation_learning.models.utils.layers import BaseProcessingNet
from imitation_learning.models.utils.gradient_reversal_layer import ReverseLayerF, compute_alpha
import copy

class GCBCTransfer(BaseModel):
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
            bridge_data_params=None,
            classifier_validation_params=None,
            datasource_class_mult=0,
            use_grad_reverse=True,
            child_model_class=GCBCImages,
            alpha_delay=0
        )
        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def build_network(self):
        self.build_single_task_model()
        self.build_bridge_model()
        if self._hp.classifier_validation_params is not None:
            self.build_classifier_validation_model()
        if self._hp.datasource_class_mult != 0:
            self.classifier = BaseProcessingNet(self.single_task_model.embedding_size, mid_dim=256,
                                                out_dim=2, num_layers=3)

    def build_single_task_model(self):
        single_task_params = AttrDict()
        single_task_params.update(self._hp.shared_params)
        single_task_params.update(self._hp.single_task_params)
        single_task_params.data_conf = self._hp.data_conf.single_task.dataconf
        self.single_task_model = self._hp.child_model_class(single_task_params, self._logger)
        self._hp.single_task_params = self.single_task_model._hp

    def build_bridge_model(self):
        bridge_data_params = AttrDict()
        bridge_data_params.update(self._hp.shared_params)
        bridge_data_params.update(self._hp.bridge_data_params)
        bridge_data_params.data_conf = self._hp.data_conf.bridge_data.dataconf
        self.bridge_data_model = self._hp.child_model_class(bridge_data_params, self._logger)
        assert self.bridge_data_model.encoder is not None
        delattr(self.bridge_data_model, 'encoder')
        self.bridge_data_model.encoder = self.single_task_model.encoder
        if self.bridge_data_model._hp.shared_classifier:
            assert self.bridge_data_model.classifier is not None
            delattr(self.bridge_data_model, 'classifier')
            self.bridge_data_model.classifier = self.single_task_model.classifier
        self._hp.bridge_data_params = self.bridge_data_model._hp

    def build_classifier_validation_model(self):
        bridge_data_params = AttrDict()
        bridge_data_params.update(self._hp.shared_params)
        bridge_data_params.update(self._hp.classifier_validation_params)
        bridge_data_params.data_conf = self._hp.data_conf.classifier_validation.dataconf
        self.classifier_validation_model = self._hp.child_model_class(bridge_data_params, self._logger)
        assert self.classifier_validation_model.encoder is not None
        delattr(self.classifier_validation_model, 'encoder')
        self.classifier_validation_model.encoder = self.single_task_model.encoder
        self._hp.classifier_validation_params = self.bridge_data_model._hp

    def forward(self, inputs):
        out = AttrDict()
        if 'single_task' not in inputs:
            out.validation_run = self.single_task_model.forward(inputs)
            return out
        inputs.single_task.global_step = inputs.global_step
        inputs.bridge_data.global_step = inputs.global_step
        inputs.single_task.max_iterations = inputs.max_iterations
        inputs.bridge_data.max_iterations = inputs.max_iterations
        out.single_task = self.single_task_model.forward(inputs.single_task)
        out.bridge_data = self.bridge_data_model.forward(inputs.bridge_data)
        if self._hp.classifier_validation_params is not None:
                out.classifier_validation = self.classifier_validation_model.forward(inputs.classifier_validation)
        if self._hp.datasource_class_mult != 0:
            embeddings = torch.cat([out.single_task.embedding, out.bridge_data.embedding], dim=0)
            if self._hp.use_grad_reverse:
                alpha = compute_alpha(inputs, self._hp.alpha_delay)
                out.alpha = alpha
                embeddings = ReverseLayerF.apply(embeddings, alpha)
            else:
                embeddings = embeddings.detach()
            out.pred_logit = self.classifier(embeddings)
        return out

    def loss(self, model_input, model_output):
        if 'validation_run' in model_output:
            return self.single_task_model.loss(model_input, model_output.validation_run)
        losses = AttrDict()
        losses_single_task = self.single_task_model.loss(model_input.single_task, model_output.single_task, compute_total_loss=False)
        losses_bridge_data = self.bridge_data_model.loss(model_input.bridge_data, model_output.bridge_data, compute_total_loss=False)
        for k, v in losses_single_task.items():
            losses['single_task_' + k] = (v[0], v[1]*self.single_task_model._hp.model_loss_mult)
        for k, v in losses_bridge_data.items():
            losses['bridge_data_' + k] = (v[0], v[1]*self.bridge_data_model._hp.model_loss_mult)
        if self._hp.classifier_validation_params is not None:
            losses_classifier_validation = self.classifier_validation_model.loss(model_input.classifier_validation, model_output.classifier_validation, compute_total_loss=False)
            for k, v in losses_classifier_validation.items():
                losses['classifier_validation_' + k] = (v[0], v[1]*self.classifier_validation_model._hp.model_loss_mult)

        if self._hp.datasource_class_mult != 0:
            self.class_labels= []
            for i in range(2):
                self.class_labels.append(torch.ones(self._hp.batch_size, dtype=torch.long) * i)
            self.class_labels = torch.cat(self.class_labels, dim=0).to(torch.device('cuda'))
            losses.datasource_classification_loss = [nn.CrossEntropyLoss()(model_output.pred_logit, self.class_labels),
                                                 self._hp.datasource_class_mult]

        # compute total loss
        losses.total_loss = torch.stack([l[0] * l[1] for l in losses.values()]).sum()
        return losses

    def _log_outputs(self, model_output, inputs, losses, step, phase):
        if 'validation_run' in model_output:
            assert phase == 'val'
            self.single_task_model._log_outputs(model_output.validation_run, inputs, losses, step, phase)
            return

        self.single_task_model._log_outputs(model_output.single_task, inputs.single_task, losses, step, phase, override_sufix=self.dataset_hp['single_task'].name)
        self.bridge_data_model._log_outputs(model_output.bridge_data, inputs.bridge_data, losses, step, phase, override_sufix=self.dataset_hp['bridge_data'].name)
        if self._hp.classifier_validation_params is not None:
            self.classifier_validation_model._log_outputs(model_output.classifier_validation, inputs.classifier_validation, losses, step, phase,
                                                override_sufix=self.dataset_hp['classifier_validation'].name)

        if self._hp.datasource_class_mult != 0:
            predictions = torch.argmax(model_output.pred_logit, dim=1)
            error_rate = np.mean(trch2npy(predictions) != trch2npy(self.class_labels))
            self._logger.log_scalar(error_rate, 'datasource_class_error_rate', step)
            if self._hp.use_grad_reverse:
                self._logger.log_scalar(model_output.alpha, 'data_source_alpha', step)
