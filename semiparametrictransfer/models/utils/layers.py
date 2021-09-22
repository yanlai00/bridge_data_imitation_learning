import torch.nn as nn
from functools import partial
from semiparametrictransfer.utils.general_utils import HasParameters
import math
from semiparametrictransfer.utils.general_utils import AttrDict


def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if isinstance(m, nn.Conv2d):
        pass    # by default PyTorch uses Kaiming_Normal initializer

class Block(nn.Sequential):
    def __init__(self, **kwargs):
        nn.Sequential.__init__(self)
        self.params = self.get_default_params()
        self.override_defaults(kwargs)

        self.build_block()
        self.complete_block()

    def get_default_params(self):
        params = AttrDict(
            activation=nn.LeakyReLU(0.2, inplace=True),
            normalization='batch',
            normalization_params=AttrDict()
        )
        return params

    def override_defaults(self, override):
        for name, value in override.items():
            # print('overriding param {} to value {}'.format(name, value))
            self.params[name] = value

    def build_block(self):
        raise NotImplementedError

    def complete_block(self):
        if self.params.normalization is not None:
            self.params.normalization_params.affine = True
            # TODO add a warning if the normalization is over 1 element
            if self.params.normalization == 'batch':
                normalization = nn.BatchNorm1d if self.params.d == 1 else nn.BatchNorm2d
                self.params.normalization_params.track_running_stats = True

            elif self.params.normalization == 'instance':
                normalization = nn.InstanceNorm1d if self.params.d == 1 else nn.InstanceNorm2d
                self.params.normalization_params.track_running_stats = True
                # TODO if affine is false, the biases will not be learned

            elif self.params.normalization == 'group':
                normalization = partial(nn.GroupNorm, 8)
                if self.params.out_dim < 32:
                    raise NotImplementedError("note that group norm is likely to not work with this small groups")

            else:
                raise ValueError("Normalization type {} unknown".format(self.params.normalization))
            self.add_module('norm', normalization(self.params.out_dim, **self.params.normalization_params))

        if self.params.activation is not None:
            self.add_module('activation', self.params.activation)

    def calc_output_size_and_padding(self, input_size):
        """
        :param input_size:  list of H, W
        :return:
        """

        p = (self.params.kernel_size - self.params.stride) // 2

        s = self.params.stride
        k = self.params.kernel_size
        i_h = input_size[0]
        i_w = input_size[1]
        out_h = (i_h + 2 * p - k) / s + 1
        out_w = (i_w + 2 * p - k) / s + 1
        return [out_h, out_w]


class ConvBlock(Block):
    def get_default_params(self):
        params = super(ConvBlock, self).get_default_params()
        params.update(AttrDict(
            d=2,
            kernel_size=3,
            stride=1,
        ))
        return params

    def build_block(self):
        if self.params.d == 2:
            cls = nn.Conv2d
        elif self.params.d == 1:
            cls = nn.Conv1d
        elif self.params.d == -2:
            cls = nn.ConvTranspose2d

        padding = (self.params.kernel_size - self.params.stride) // 2
        self.add_module('conv', cls(
            self.params.in_dim, self.params.out_dim, self.params.kernel_size, self.params.stride, padding))

class ConvBlockEnc(ConvBlock):
    def get_default_params(self):
        params = super(ConvBlockEnc, self).get_default_params()
        params.update(AttrDict(
            kernel_size=4,
            stride=2,
        ))
        return params

class ConvBlockDec(ConvBlock):
    def get_default_params(self):
        params = super(ConvBlockDec, self).get_default_params()
        params.update(AttrDict(
            d = -2,
            kernel_size=4,
            stride=2,
        ))
        return params

class FCBlock(Block):
    def get_default_params(self):
        params = super(FCBlock, self).get_default_params()
        params.update(AttrDict(
            d=1,
        ))
        return params

    def build_block(self):
        self.add_module('linear', nn.Linear(self.params.in_dim, self.params.out_dim))


class Linear(FCBlock):
    def get_default_params(self):
        params = super(Linear, self).get_default_params()
        params.update(AttrDict(
            activation=None
        ))
        return params


class BaseProcessingNet(nn.Sequential):
    """ Constructs a network that keeps the activation dimensions the same throughout the network
    Builds an MLP or CNN, depending on the builder. Alternatively uses custom blocks """

    def __init__(self, in_dim, mid_dim, out_dim, num_layers, block=FCBlock,
                 final_activation=None, normalization='batch'):
        super(BaseProcessingNet, self).__init__()

        self.add_module('input', block(in_dim=in_dim, out_dim=mid_dim, normalization=None))
        for i in range(num_layers):
            self.add_module('pyramid-{}'.format(i),
                            block(in_dim=mid_dim, out_dim=mid_dim, normalization=normalization))

        self.add_module('head'.format(i + 1),
                        block(in_dim=mid_dim, out_dim=out_dim, normalization=None, activation=final_activation))
        self.apply(init_weights_xavier)


