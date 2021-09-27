import torch
import torch.nn as nn

import sys
from imitation_learning.models.utils.orig_resnet.resnet import ResNet, BasicBlock, Bottleneck, model_urls

def repeat_weights(weights, new_channels):
    prev_channels = weights.shape[1]
    assert prev_channels == 3, "Original weights should have three input channels"
    new_shape = list(weights.shape[:])
    new_shape[1] = new_channels
    new_weights = torch.zeros(new_shape, dtype=weights.dtype, layout=weights.layout, device=weights.device)
    for i in range(new_channels):
        new_weights.data[:, i] = weights[:, i % prev_channels].clone()
    return new_weights

if sys.version_info[0] == 3:
    from torch.hub import load_state_dict_from_url

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetCustomStride(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet34shallow(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3, 1], pretrained, progress, strides=(2, 2, 1, 1, 1, 1), planes=(64, 128, 256, 512, 8),
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

class ResNetCustomStride(ResNet):
    # planes = (64, 128, 128, 256)
    def __init__(self, block, layers, strides=(2, 2, 1, 1, 1), planes=(64, 128, 256, 512), num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, create_final_fc_layer=False):
        # super(ResNet, self).__init__()
        self.block = block
        nn.Module.__init__(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.create_final_fc_layer = create_final_fc_layer

        self.planes = planes
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1)
        self.layer1 = self._make_layer(block, planes[0], layers[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=strides[2],
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=strides[3],
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=strides[4],
                                       dilate=replace_stride_with_dilation[2])
        if len(planes) == 5:
            self.layer5 = self._make_layer(block, planes[4], layers[4], stride=strides[5],
                                           dilate=replace_stride_with_dilation[2])
        else:
            self.layer5 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.create_final_fc_layer:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_num_output_featuremaps(self):
        if self.block.__name__ == BasicBlock.__name__:
            return [12, 16, self.planes[-1]]
        if self.block.__name__ == Bottleneck.__name__:
            return [12, 16, self.planes[-1]*4]
        else:
            raise NotImplementedError

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.layer5 is not None:
            return self.layer5(x)
        if not self.create_final_fc_layer:
            return x
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x


def get_resnet_encoder(resnet_type, channels_in=3, pretrained=True, **kwargs):
    if resnet_type == 'resnet50':
        Model = resnet50
    elif resnet_type == 'resnet34':
        Model = resnet34
    elif resnet_type == 'resnet34shallow':
        Model = resnet34shallow
    elif resnet_type == 'resnet18':
        Model = resnet18
    else:
        raise NotImplementedError
    model = Model(pretrained=pretrained, progress=True, **kwargs)
    for param in model.parameters():
        param.requires_grad = True

    if channels_in != 3:
        orig_weights = model.conv1.weight.clone().detach().data
        new_weights = repeat_weights(orig_weights, channels_in)
        new_layer = nn.Conv2d(channels_in, orig_weights.shape[0],  kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        new_layer.weight = nn.Parameter(new_weights)
        model.conv1 = new_layer

    return model