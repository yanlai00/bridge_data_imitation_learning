import numpy as np
import torch
import os
import torch.nn as nn
from bridgedata.models.utils.modelutils import get_one_hot
from bridgedata.utils.general_utils import AttrDict
from bridgedata.utils.general_utils import select_indices, trch2npy
from bridgedata.models.base_model import BaseModel
from bridgedata.models.utils.resnet import get_resnet_encoder
from bridgedata.models.utils.gradient_reversal_layer import ReverseLayerF, compute_alpha
from bridgedata.models.utils.subnetworks import ConvEncoder
from bridgedata.models.utils.layers import BaseProcessingNet
from bridgedata.utils.general_utils import np_unstack
from bridgedata.models.utils.spatial_softmax import SpatialSoftmax
from bridgedata.data_sets.data_augmentation import get_random_crop
from bridgedata.utils.general_utils import npy2trch
import cv2

def get_tlen_from_padmask(padmask):
    tlen = np.zeros(padmask.shape[0], dtype=np.int)
    for b in range(padmask.shape[0]):
        for t in range(padmask.shape[1]):
            if padmask[b,t] == 1:
                tlen[b] = int(t + 1)
            else:
                break
    return tlen


class GeneralImageEncoder(nn.Module):
    """
    combine resnet encoder with spatial softmax and dense layer on top
    """
    def __init__(self, resnet_type, img_sz=False, in_channel=3, out_dim=None, use_spatial_softmax=True, pretrained_resnet=False):
        """
        param out_dim: dimensionality of output vector
        """
        super(GeneralImageEncoder, self).__init__()
        self.img_sz = img_sz
        self.out_dim = out_dim
        self.in_channel = in_channel
        if not use_spatial_softmax:
            assert resnet_type == 'resnet34'
            resnet_type = 'resnet34shallow'

        if self.img_sz == [96, 128]:
            self.encoder = get_resnet_encoder(resnet_type, in_channel, pretrained=pretrained_resnet, strides=(2, 2, 2, 1, 1))
        elif self.img_sz == [48, 64]:
            self.encoder = get_resnet_encoder(resnet_type, in_channel, pretrained=pretrained_resnet)
        else:
            raise NotImplementedError
        num_out_features = self.encoder.get_num_output_featuremaps()
        if use_spatial_softmax:
            self.spatial_softmax = SpatialSoftmax(12, 16, num_out_features[-1])
            dense_input_size = num_out_features[-1] * 2
        else:
            dense_input_size = num_out_features[0]*num_out_features[1]*num_out_features[2]
            self.spatial_softmax = None
        if out_dim is not None:
            self.dense = BaseProcessingNet(dense_input_size, mid_dim=256, out_dim=out_dim,
                                           num_layers=1)
        else:
            self.dense = None

    @property
    def output_dim(self):
        if self.dense is not None:
            return self.out_dim
        if self.spatial_softmax is not None:
            return self.encoder.get_num_output_featuremaps()[-1]*2
        else:
            nf = self.encoder.get_num_output_featuremaps()
            return nf[0]*nf[1]*nf[2]

    def forward(self, input_image):
        assert np.all(np.array(input_image.shape[-3:]) == np.concatenate([np.array([self.in_channel]), self.img_sz]))
        emb = self.encoder(input_image)
        if self.spatial_softmax is not None:
            emb = self.spatial_softmax(emb)
        else:
            emb = emb.view(emb.shape[0], -1)
        if self.dense is not None:
            emb = self.dense(emb)
        return emb

class ActionHeadBase():
    def loss(self, model_output, targets):
        pass

    def make_prediction(self, emb, outdict):
        pass

    def make_summaries(self):
        pass

class ActionHead(nn.Module, ActionHeadBase):
    def __init__(self, params, input_size):
        super(ActionHead, self).__init__()
        self._hp = params
        self.action_predictor = BaseProcessingNet(input_size, mid_dim=256, out_dim=params.action_dim, num_layers=3)

    def loss(self, model_output, targets):
        if self._hp.dataset_normalization:
            a_pred = model_output.normed_pred_actions
        else:
            a_pred = model_output.pred_actions

        if self._hp.action_dim_weighting is None:
            return (torch.nn.MSELoss()(a_pred, targets), 1.0)
        else:
            mse_loss = torch.mean((a_pred - targets)**2*npy2trch(np.array(self._hp.action_dim_weighting)))
            return (mse_loss, 1.0)

    def make_prediction(self, emb, outdict):
        return self.action_predictor(emb)

class ActionHeadGMM(nn.Module, ActionHeadBase):
    def __init__(self, params, input_size):
        super(ActionHeadGMM, self).__init__()
        self._hp = params
        assert self._hp.action_gmm[1] == 'diag_covariance'
        self.num_components = params.action_gmm[0]
        apred_output_size = params.action_dim * 2 * self.num_components + self.num_components
        self.action_predictor = BaseProcessingNet(input_size, mid_dim=256, out_dim=apred_output_size, num_layers=3)

    def loss(self, model_output, targets):
        return (compute_gmm_loss(model_output.gmm_parameters.means,
                                 model_output.gmm_parameters.covariances,
                                 model_output.gmm_parameters.mixing_coefficients,
                                 targets), 1.0)

    def make_prediction(self, emb, outdict):
        a_pred = self.action_predictor(emb)

        means = a_pred[:, 0:self._hp.action_dim * self.num_components].reshape(-1, self.num_components, self._hp.action_dim)
        covariances = a_pred[:, self._hp.action_dim * self.num_components:self._hp.action_dim * self.num_components * 2].reshape(-1, self.num_components, self._hp.action_dim)
        mixing_coefficients = a_pred[:, self._hp.action_dim * self.num_components * 2:]
        outdict.gmm_parameters = AttrDict(means=means, covariances=covariances, mixing_coefficients=mixing_coefficients)

        best_index = torch.argmax(mixing_coefficients, dim=1)
        best_mean = torch.gather(means, dim=1, index=best_index[:, None, None].repeat(1, 1, 7)).squeeze()
        #when sampling:
        # best_cov = torch.gather(covariances, dim=1, index=best_index[:, None, None].repeat(1, 1, 7)).squeeze()
        # actions = torch.distributions.multivariate_normal.MultivariateNormal(best_mean, covariance_matrix=torch.diag(best_cov)).sample(1)
        print('mixing coefficients', mixing_coefficients)
        return best_mean

def compute_gmm_loss(means, covariances, mixing_coefficients, action_targets):
    """
    compute negative log likelihood of GMM
    means: shape B, num_comp, adim
    covariances: shape B, num_comp, adim  (assuming diagonal covariance)
    action targets: shape B, adim
    """
    log_prob = 0
    num_components = means.shape[1]
    mix = torch.nn.Softmax(dim=1)(mixing_coefficients)
    covariances = nn.Sigmoid()(covariances) * 5  # numberically better than exponential?

    batch_size = means.shape[0]

    for i in range(num_components):
        cov = []
        for b in range(batch_size):
            cov.append(torch.diag(covariances[b, i]))
        cov = torch.stack(cov, 0)
        log_prob += -torch.distributions.multivariate_normal.MultivariateNormal(means[:, i], covariance_matrix=cov).log_prob(action_targets)*mix[:, i]
    loss = log_prob.mean()
    return loss

class GCBCImages(BaseModel):
    def __init__(self, overrideparams, logger=None):
        super(GCBCImages, self).__init__(overrideparams, logger)
        self._hp = self._default_hparams()
        self._override_defaults(overrideparams)  # override defaults with config file

        assert self._hp.batch_size != -1
        assert self._hp.action_dim!= -1

        self.action_targets = None
        self.goal_image = None
        self.context = None
        if self._hp.stage == 'finetuning':
            self._hp.domain_class_mult = 0

        self.build_network()

    def _default_hparams(self):
        default_dict = AttrDict(
            state_dim=-1,
            action_dim=-1,
            goal_cond=False,
            goal_state_delta_t=None,
            img_sz=[48, 64],
            encoder_embedding_size=None, # if not None a dense layer is added in the encoder.
            encoder_spatial_softmax=True,
            data_conf=None,
            domain_class_mult=0,
            concatenate_cameras=False,
            num_concat_cams=2,
            sel_camera=0,
            sample_camera=False,
            resnet='resnet18',
            pretrained_resnet=False,
            input_nc=None,
            use_grad_reverse=True,
            extra_horizon=5,
            num_context=0,
            model_loss_mult=1.,  # multiplier for all losses of the model, only used in GCBCTransfer
            reuse_action_predictor=None,
            action_dim_weighting=None,  # weighting the gripper action by 10
            action_gmm=False,  # (num_components, covariance_type), covariance_type is either diag_covariance, or full_covariance,
            freeze_encoder=False,
            shared_classifier=False,
            separate_classifier=False,
            alpha_delay=0,
            task_id_conditioning=False,
            stack_goal_images=1,
            separate_encoder=False,
        )
        # add new params to parent params
        parent_params = super(GCBCImages, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def sample_tsteps(self, images, actions, tlen, sample_goal):
        tlen = trch2npy(tlen)
        assert len(images.shape) == 5  # b, t, 3, h, w
        batch_size = images.shape[0]

        tstart_context = np.array([np.random.randint(0, tlen - self._hp.num_context - 1) for tlen in tlen])
        t0 = tstart_context + self._hp.num_context

        current_image = select_indices(images, t0)
        action_targets = select_indices(actions, t0)
        goal_image = None
        if sample_goal:
            assert self._hp.goal_cond
            if self._hp.goal_state_delta_t is not None:
                tg = t0 + np.random.randint(1, self._hp.goal_state_delta_t + 1, batch_size)
                tg = np.array([np.clip(tg[b], 0, tlen[b] - 1) for b in range(batch_size)])
                goal_image = select_indices(images, tg)
            else:
                goal_image = select_indices(images, tlen - 1)

        return current_image, goal_image, action_targets

    def build_network(self):
        if self._hp.concatenate_cameras:
            in_channel = 3*self._hp.num_concat_cams
        else:
            in_channel = 3
        self.encoder = GeneralImageEncoder(self._hp.resnet, in_channel=in_channel, out_dim=self._hp.encoder_embedding_size,
                                            use_spatial_softmax=self._hp.encoder_spatial_softmax, img_sz=self._hp.img_sz)
        if self._hp.separate_encoder:
            goal_in_channel = 3 * self._hp.stack_goal_images
            self.goal_encoder = GeneralImageEncoder(self._hp.resnet, in_channel=goal_in_channel, out_dim=self._hp.encoder_embedding_size,
                                                use_spatial_softmax=self._hp.encoder_spatial_softmax, img_sz=self._hp.img_sz)
        self.embedding_size = self.encoder.output_dim
        if self._hp.goal_cond:
            if self._hp.separate_encoder:
                apred_input_size = 2 * self.embedding_size
            else:
                apred_input_size = (self._hp.stack_goal_images + 1) * self.embedding_size
            if (not self._hp.shared_classifier) and (not self._hp.separate_classifier):
                input_size = apred_input_size
            else:
                input_size = self.embedding_size
        else: # if shared classifier, apply classifier on the source and goal embedding separately
            apred_input_size = self.embedding_size
            input_size = apred_input_size
        if self._hp.action_gmm:
            actionhead_class = ActionHeadGMM
        else:
            actionhead_class = ActionHead
        if self._hp.task_id_conditioning:
            apred_input_size += self._hp.task_id_conditioning
        self.action_head = actionhead_class(self._hp, apred_input_size)

        if self._hp.domain_class_mult:
            assert self._hp.num_domains > 1
            self.classifier = BaseProcessingNet(input_size, mid_dim=256,
                                                out_dim=self._hp.num_domains, num_layers=3)
            if self._hp.separate_classifier:
                self.classifier_goal = BaseProcessingNet(input_size, mid_dim=256,
                                                out_dim=self._hp.num_domains, num_layers=3)

    def get_network_inputs(self, inputs, sample_goal=False):
        if 'domain_ind' in inputs:
            self.class_labels = inputs.domain_ind.squeeze()
        if 'final_image_domain_ind' in inputs:
            self.class_labels_goal = inputs.final_image_domain_ind.squeeze()
        current_image = inputs.images
        goal_image = inputs.final_image
        action_targets = self.apply_dataset_normalization(inputs.actions, 'actions')
        return current_image, goal_image, action_targets

    def get_embedding(self, pred_input, context):
        return self.encoder(pred_input)

    def get_goal_embedding(self, pred_input, context):
        assert self._hp.separate_encoder
        return self.goal_encoder(pred_input)

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        self.current_image, self.goal_image, self.action_targets = self.get_network_inputs(inputs, self._hp.goal_cond)
        outdict = AttrDict()
        if 'goal_embedding' in inputs:
            emb = self.get_embedding(self.current_image, self.context)
            emb = torch.cat([emb, inputs.goal_embedding], dim=1)
        elif self._hp.goal_cond:
            if self._hp.separate_encoder:
                outdict.embedding = current_img_emb = self.get_embedding(self.current_image, self.context)
                # import ipdb; ipdb.set_trace()
                goal_img_emb = self.get_goal_embedding(torch.cat(self.goal_image, dim=1), self.context)
                emb = torch.cat([current_img_emb, goal_img_emb], dim=1)
            else:
                embs = [self.get_embedding(img, self.context) for img in [self.current_image, *self.goal_image]]
            # outdict.embedding is used for the datasource classifier in the GCBCTransfer model. We only use the current
            # time step's embedding for this classification to make goal-conditioned and non-goal-conditioned settings
            # compatible
                outdict.embedding = embs[0]
                emb = torch.cat(embs, dim=1)
        else:
            emb = self.get_embedding(self.current_image, self.context)
            outdict.embedding = emb

        apred_emb = emb
        if self._hp.task_id_conditioning:
            assert not self._hp.goal_cond
            one_hot = get_one_hot(self._hp.task_id_conditioning, inputs.task_id).to(self._hp.device)
            apred_emb = torch.cat([apred_emb, one_hot], dim=1)

        if self._hp.freeze_encoder:
            apred_emb = apred_emb.detach()
        a_pred = self.action_head.make_prediction(apred_emb, outdict)

        if self._hp.dataset_normalization:
            outdict.normed_pred_actions = a_pred
            outdict.pred_actions = self.unnormalize_dataset(a_pred, 'actions')
        else:
            outdict.pred_actions = a_pred

        if self._hp.use_grad_reverse and self.training:
            if self._hp.domain_class_mult != 0:
                alpha = compute_alpha(inputs, self._hp.alpha_delay)
                outdict.alpha = alpha
                emb = ReverseLayerF.apply(emb, alpha)
        else:
            emb = emb.detach()
        if self._hp.domain_class_mult != 0:
            if self._hp.goal_cond and self._hp.shared_classifier:
                emb_dim = emb.shape[1]
                outdict.domain_pred_logit = self.classifier(emb[:, :emb_dim//2])
                outdict.domain_pred_logit_goal = self.classifier(emb[:, emb_dim//2:])
            elif self._hp.goal_cond and self._hp.separate_classifier:
                emb_dim = emb.shape[1]
                # print('emb_dim', emb_dim)
                # print('embedding size', self.embedding_size)
                outdict.domain_pred_logit = self.classifier(emb[:, :emb_dim//2])
                outdict.domain_pred_logit_goal = self.classifier_goal(emb[:, emb_dim//2:])
            else:
                outdict.domain_pred_logit = self.classifier(emb)
            # print('outdict.domain_pred_logit shape', outdict.domain_pred_logit.shape)
        return outdict


    def loss(self, model_input, model_output, compute_total_loss=True):
        losses = AttrDict()

        losses.action_head_loss = self.action_head.loss(model_output, self.action_targets)

        if self._hp.domain_class_mult != 0:
            if self._hp.goal_cond and self._hp.shared_classifier:
                losses.classification_loss_goal = [nn.CrossEntropyLoss()(model_output.domain_pred_logit_goal, self.class_labels),
                                          self._hp.domain_class_mult]
            elif self._hp.goal_cond and self._hp.separate_classifier:
                losses.classification_loss_goal = [nn.CrossEntropyLoss()(model_output.domain_pred_logit_goal, self.class_labels_goal),
                                          self._hp.domain_class_mult]
            losses.classification_loss = [nn.CrossEntropyLoss()(model_output.domain_pred_logit, self.class_labels),
                                        self._hp.domain_class_mult]

        # compute total loss
        if compute_total_loss:
            losses.total_loss = torch.stack([l[0] * l[1] for l in losses.values()]).sum()
        return losses

    def _log_outputs(self, model_output, inputs, losses, step, phase, override_sufix=None):
        if 'normed_pred_actions' in model_output:
            use_pred_actions = model_output.normed_pred_actions
        elif 'pred_actions' in model_output:
            use_pred_actions = model_output.pred_actions
        else:
            return
        if self._hp.concatenate_cameras:
            current_image = torch.cat(torch.split(self.current_image, 3, 1), 2)
            if self._hp.goal_cond:
                # TODO adapt to stack goal image setting
                goal_image = torch.cat(torch.split(self.goal_image, 3, 1), 2)
        else:
            current_image = self.current_image
            goal_image = self.goal_image
        sel_image_row = trch2npy(torch.cat(torch.unbind((current_image + 1) / 2, dim=0), dim=2)).transpose(1, 2, 0)

        gtruth_action_row1 = batch_action2img(trch2npy(self.action_targets[:, :2]), self._hp.img_sz[1], 3, action_scale=1)
        gtruth_action_row1 = np.concatenate(np_unstack(gtruth_action_row1, axis=0), axis=1)
        pred_action_row1 = batch_action2img(trch2npy(use_pred_actions[:, :2]), self._hp.img_sz[1], 3, action_scale=1)
        pred_action_row1 = np.concatenate(np_unstack(pred_action_row1, axis=0), axis=1)
        context_image_rows = self.get_context_image_rows()

        if self._hp.goal_cond:
            goal_image_rows = []
            for goal_image_single in goal_image:
                goal_image_row = trch2npy(torch.cat(torch.unbind((goal_image_single + 1)/2, dim=0), dim=2)).transpose(1, 2, 0)
                goal_image_rows.append(goal_image_row)
            image_rows = context_image_rows + [sel_image_row, *goal_image_rows, gtruth_action_row1, pred_action_row1]
        else:
            image_rows = context_image_rows + [sel_image_row, gtruth_action_row1, pred_action_row1]
        out = np.concatenate(image_rows, axis=0)
        out = out.transpose(2, 0, 1)
        if override_sufix is not None:
            suffix = "_" + override_sufix
        else:
            suffix = "_" + self.dataset_sufix
        print('logging {} for {}'.format('images' + suffix, phase))
        self._logger.log_image(out, 'images' + suffix, step, phase)

        if self._hp.domain_class_mult != 0:
            predictions = torch.argmax(model_output.domain_pred_logit, dim=1)
            error_rate = np.mean(trch2npy(predictions) != trch2npy(self.class_labels))
            self._logger.log_scalar(error_rate, 'domain_class_error_rate' + suffix, step)
            if self._hp.goal_cond and (self._hp.shared_classifier or self._hp.separate_classifier):
                predictions = torch.argmax(model_output.domain_pred_logit_goal, dim=1)
                if self._hp.shared_classifier:
                    error_rate = np.mean(trch2npy(predictions) != trch2npy(self.class_labels))
                else:
                    error_rate = np.mean(trch2npy(predictions) != trch2npy(self.class_labels_goal))
                self._logger.log_scalar(error_rate, 'domain_class_error_rate_goal' + suffix, step)

        if hasattr(model_output, 'alpha'):
            self._logger.log_scalar(model_output.alpha, 'alpha', step)

        if len(inputs.images.shape) == 6 and step < 1000:
            input_images = (((trch2npy(inputs.images[:, :, 0]) + 1) / 2) * 255.).astype(np.uint8)
            input_images = np.concatenate(np_unstack(input_images, 0)[:5], 3)
            self._logger.log_video(input_images, 'inputimages_video' + suffix, step, phase, 6)

    def get_context_image_rows(self):
        return []

class SuccessClassifierImages(BaseModel):
    """End-to-end success classifier"""

    def __init__(self, overrideparams, logger=None):
        super(SuccessClassifierImages, self).__init__(overrideparams, logger)
        self._hp = self._default_hparams()
        self._override_defaults(overrideparams)  # override defaults with config file

        assert self._hp.batch_size != -1

        self.action_targets = None
        self.context = None

        self.build_network()

    def _default_hparams(self):
        default_dict = AttrDict(
            img_sz=[48, 64],
            encoder_embedding_size=None, # if not None a dense layer is added in the encoder.
            encoder_spatial_softmax=True,
            num_classes=2,
            resnet='resnet18',
            pretrained_resnet=False,
            freeze_encoder=False,
            task_id_conditioning=False,
        )
        # add new params to parent params
        parent_params = super(SuccessClassifierImages, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def build_network(self):
        in_channel = 3
        self.encoder = GeneralImageEncoder(self._hp.resnet, in_channel=in_channel, out_dim=self._hp.encoder_embedding_size,
                                            use_spatial_softmax=self._hp.encoder_spatial_softmax, img_sz=self._hp.img_sz, pretrained_resnet=self._hp.pretrained_resnet)
        self.embedding_size = self.encoder.output_dim

        apred_input_size = self.embedding_size
        if self._hp.task_id_conditioning:
            apred_input_size += self._hp.task_id_conditioning

        self.classifier = BaseProcessingNet(apred_input_size, mid_dim=256, out_dim=self._hp.num_classes, num_layers=3)

    def get_network_inputs(self, inputs):
        class_labels = inputs.classes.squeeze()
        current_image = inputs.images
        return current_image, class_labels

    def get_embedding(self, pred_input):
        return self.encoder(pred_input)

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        self.current_image, self.class_labels = self.get_network_inputs(inputs)
        outdict = AttrDict()
        emb = self.get_embedding(self.current_image)
        outdict.embedding = emb

        apred_emb = emb
        if self._hp.task_id_conditioning:
            one_hot = get_one_hot(self._hp.task_id_conditioning, inputs.task_id).to(self._hp.device)
            apred_emb = torch.cat([apred_emb, one_hot], dim=1)

        if self._hp.freeze_encoder:
            apred_emb = apred_emb.detach()

        outdict.pred_logit = self.classifier(apred_emb)
        return outdict

    def loss(self, model_input, model_output, compute_total_loss=True):
        losses = AttrDict()
        losses.classification_loss = [nn.CrossEntropyLoss()(model_output.pred_logit, self.class_labels), 1.0]
        if compute_total_loss:
            losses.total_loss = torch.stack([l[0] * l[1] for l in losses.values()]).sum()
        return losses

    def _log_outputs(self, model_output, inputs, losses, step, phase, override_sufix=None):
        if override_sufix is not None:
            suffix = "_" + override_sufix
        else:
            suffix = "_" + self.dataset_sufix
        predictions = torch.argmax(model_output.pred_logit, dim=1)
        error_rate = np.mean(trch2npy(predictions) != trch2npy(self.class_labels))
        self._logger.log_scalar(error_rate, 'classifier_error_rate' + suffix, step)

        current_image = self.current_image
        text = trch2npy(predictions)
        processed_image = batch_put_text_on_img(current_image, text)
        out = processed_image.transpose(2, 0, 1)
        # import ipdb; ipdb.set_trace()
        print('logging {} for {}'.format('images' + suffix, phase))
        self._logger.log_image(out, 'images' + suffix, step, phase)


def action2img(action, res, channels, action_scale):
    assert action.size == 2  # can only plot 2-dimensional actions
    img = np.zeros((res, res, channels), dtype=np.float32).copy()
    start_pt = res / 2 * np.ones((2,))
    end_pt = start_pt + action * action_scale * (res / 2 - 1) * np.array([1, -1])  # swaps last dimension
    np2pt = lambda x: tuple(np.asarray(x, int))
    img = cv2.arrowedLine(img, np2pt(start_pt), np2pt(end_pt), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.2)
    return img * 255.0

def batch_action2img(actions, res, channels, action_scale=50):
    batch = actions.shape[0]
    im = np.empty((batch, res, res, channels), dtype=np.float32)
    for b in range(batch):
        im[b] = action2img(actions[b], res, channels, action_scale)
    return im

def put_text_on_img(img, text, font=1, org=(50,50), fontScale=2, color=(0,0,255)):
    cv2.putText(img, text, org, font, fontScale, color)

def batch_put_text_on_img(images, texts):
    images_out = []
    for img, text in zip(torch.unbind((images + 1) / 2, dim=0), texts):
        img = np.ascontiguousarray(trch2npy(img).transpose(1, 2, 0))
        # Somehow I cannot get the padding to work; comment out for now
        # img = cv2.copyMakeBorder(img, 10, 0, 10, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        put_text_on_img(img, str(text))
        images_out.append(img)
    return np.concatenate(images_out, axis=1)

class GCBCImagesModelTest(GCBCImages):
    def __init__(self, overrideparams):
        overrideparams['batch_size'] = 1
        super(GCBCImagesModelTest, self).__init__(overrideparams)
        print('starting restoring parameters from ', self._hp.restore_path)
        if self._hp.get_sub_model:
            checkpoint = torch.load(self._hp.restore_path, map_location=self._hp.device)
            if self._hp.get_sub_model == 'single_task_params':
                sub_model_key = 'single_task_model'
            elif self._hp.get_sub_model == 'bridge_data_params':
                sub_model_key = 'bridge_data_model'
            else:
                raise NotImplementedError
            print('restoring for submodel key: ', sub_model_key)
            new_state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if sub_model_key in k:
                    new_key = str.split(k, '.')[1:]
                    new_key = '.'.join(new_key)
                    new_state_dict[new_key] = v
                    # print('orig key {}, new key {}'.format(k, new_key))
            # for k, v in self.state_dict().items():
                # print(k)
            self.load_state_dict(new_state_dict)
        else:
            self._restore_params(self._hp.strict_loading)
        self._hp = self._default_hparams()
        self._override_defaults(overrideparams)

    def _default_hparams(self):
        default_dict = AttrDict(
            get_sub_model=False,
            strict_loading=True,
            test_time_task_id=None,
        )
        # add new params to parent params
        parent_params = super(GCBCImagesModelTest, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def get_network_inputs(self, inputs, sample_goal=False):
        def maybecrop_and_select_cam(emb_input, sel_cam=True):
            if self._hp.data_conf['random_crop'] is not None:
                assert np.all(np.array(emb_input.shape[-2:]) == np.array(self._hp.data_conf['image_size_beforecrop']))
                emb_input = get_random_crop(emb_input, self._hp.data_conf['random_crop'], center_crop=True)
            if self._hp.concatenate_cameras:
                return torch.cat(torch.unbind(emb_input, dim=0), 0)[None]
            else:
                if not sel_cam:
                    return emb_input
                else:
                    return emb_input[self._hp.sel_camera][None]
        current_image = maybecrop_and_select_cam(inputs.I_0)
        goal_image = None
        if self._hp.goal_cond:
            if self._hp.stack_goal_images > 1:
                goal_image = [maybecrop_and_select_cam(goal_image_single, sel_cam=False) for goal_image_single in inputs.I_g]
            else:
                goal_image = [maybecrop_and_select_cam(inputs.I_g)]
        action_targets = None
        return current_image, goal_image, action_targets

    def forward(self, inputs):
        if self._hp.test_time_task_id is not None:
            inputs.task_id = npy2trch(np.array([self._hp.test_time_task_id]))
        outputs = super(GCBCImagesModelTest, self).forward(inputs)
        return outputs
