import numpy as np
import pdb
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from imitation_learning.utils.general_utils import AttrDict
from imitation_learning.utils.general_utils import select_indices, trch2npy
from imitation_learning.models.base_model import BaseModel
from imitation_learning.models.utils.resnet import get_resnet_encoder

from imitation_learning.models.utils.subnetworks import ConvEncoder
from imitation_learning.models.utils.layers import BaseProcessingNet
from imitation_learning.utils.general_utils import np_unstack
from imitation_learning.models.utils.spatial_softmax import SpatialSoftmax
from imitation_learning.data_sets.data_augmentation import get_random_crop
from imitation_learning.models.gcbc_images import GCBCImages
from imitation_learning.models.gcbc_images import get_tlen_from_padmask
import cv2
from imitation_learning.models.gcbc_images import GeneralImageEncoder


class GCBCImagesContext(GCBCImages):
    def __init__(self, overrideparams, logger):
        super().__init__(overrideparams, logger)
        self._hp = self._default_hparams()
        self._override_defaults(overrideparams)  # override defaults with config file

    def _default_hparams(self):
        default_dict = AttrDict(
            encoder_embedding_size=128,
            num_context=3,
        )
        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def build_network(self):
        if self._hp.resnet is not None:
            self.encoder = GeneralImageEncoder(self._hp.resnet, out_dim=self._hp.encoder_embedding_size,
                                               use_spatial_softmax=self._hp.encoder_spatial_softmax)
            self.embedding_size = self._hp.encoder_embedding_size*2 + self._hp.action_dim*self._hp.num_context
            if self._hp.goal_cond:
                input_dim = 2*self.embedding_size
            else:
                input_dim = self.embedding_size
        else:
            raise NotImplementedError
        self.action_predictor = BaseProcessingNet(input_dim, mid_dim=256, out_dim=self._hp.action_dim, num_layers=2)
        self.future_action_predictor = BaseProcessingNet(input_dim, mid_dim=256,
                                                         out_dim=self._hp.action_dim*self._hp.extra_horizon, num_layers=3)
        if self._hp.domain_class_mult:
            assert self._hp.num_domains > 1
            self.classifier = BaseProcessingNet(input_dim, mid_dim=256,
                                                out_dim=self._hp.num_domains, num_layers=3)

    def get_context(self, actions, batch_size, images, tstart_context):
        context_actions = []
        context_images = []
        for b in range(batch_size):
            context_actions.append(actions[b, tstart_context[b]:tstart_context[b] + self._hp.num_context])
            context_images.append(images[b, tstart_context[b]:tstart_context[b] + self._hp.num_context])
        context_actions = torch.stack(context_actions, dim=0)
        context_images = torch.stack(context_images, dim=0)
        return AttrDict(actions=context_actions, images=context_images)

    def get_embedding(self, pred_input, context):
        assert np.all(np.array(pred_input.shape[-3:]) == np.array([3, 48, 64]))
        embedding = self.encoder(pred_input)
        context_emb = [self.encoder(c.squeeze()) for c in torch.split(context.images, 1, 1)]
        context_emb = torch.stack(context_emb, dim=0).mean(dim=0)
        context_actions = torch.unbind(context.actions, 1)
        return torch.cat([embedding, context_emb, *context_actions], dim=1)

    def get_context_image_rows(self):
        context_images = torch.unbind(self.context.images, dim=1)
        image_rows = []
        for context_image in context_images:
            row = trch2npy(torch.cat(torch.unbind((context_image + 1)/2, dim=0), dim=2)).transpose(1, 2, 0)
            image_rows.append(row)
        return image_rows

