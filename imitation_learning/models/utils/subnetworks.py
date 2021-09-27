
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from semiparametrictransfer.models.utils.layers import ConvBlockEnc, ConvBlockDec, Linear

def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if isinstance(m, nn.Conv2d):
        pass    # by default PyTorch uses Kaiming_Normal initializer

class SequentialWithConditional(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, inp_dict):
        """Computes forward pass through the network outputting all intermediate activations with final output."""
        action = inp_dict['act']
        input = inp_dict['input']
        for i, module in enumerate(self._modules.values()):
            if isinstance(module, FiLM):
                input = module(input, action)
            else:
                input = module(input)
        return input

class GetIntermediatesSequential(nn.Sequential):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, input):
        """Computes forward pass through the network outputting all intermediate activations with final output."""
        skips = []
        for i, module in enumerate(self._modules.values()):
            input = module(input)

            if i % self.stride == 0:
                skips.append(input)
            else:
                skips.append(None)
        return input, skips[:-1]


class FiLM(nn.Module):
    def __init__(self, hp, inp_dim, feature_size):
        super().__init__()
        self._hp = hp

        self.inp_dim = inp_dim
        self.feature_size = feature_size
        self.linear = Linear(in_dim=inp_dim, out_dim=2*feature_size, builder=self._hp.builder)

    def forward(self, feats, inp):
        gb = self.linear(inp)
        gamma, beta = gb[:, :self.feature_size], gb[:, self.feature_size:]
        gamma = gamma.view(feats.size(0), feats.size(1), 1, 1)
        beta = beta.view(feats.size(0), feats.size(1), 1, 1)
        return feats * gamma + beta

def get_num_conv_layers(img_sz):
    n = math.log2(img_sz[1])
    assert n >= 3, 'imageSize must be at least 8'
    return int(n)

def calc_output_size_and_padding(input_size, k, s, p):
    """
    :param input_size:  list of H, W
    :return:
    """

    i_h = input_size[0]
    i_w = input_size[1]
    out_h = (i_h + 2 * p - k) / s + 1
    out_w = (i_w + 2 * p - k) / s + 1
    return [out_h, out_w]


class ConvEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp

        self.n = get_num_conv_layers(hp.img_sz)
        if self._hp.use_skips:
            self.net = GetIntermediatesSequential(hp.skips_stride)
        else:
            self.net = nn.Sequential()

        self.size_list = []  # C, H, W

        input_c = hp.input_nc
        
        print('l-1: indim {} outdim {}'.format(input_c, hp.ngf))
        self.size_list.append([hp.img_sz[0], hp.img_sz[1]])

        blk = ConvBlockEnc(in_dim=input_c, out_dim=hp.ngf, normalization=None, input_size=self.size_list[-1])
        self.size_list.append(blk.calc_output_size_and_padding(self.size_list[-1]))
        self.net.add_module('input', blk)

        for i in range(self.n - 3):
            filters_in = hp.ngf * 2 ** i

            blk = ConvBlockEnc(in_dim=filters_in, out_dim=filters_in * 2)
            self.size_list.append(blk.calc_output_size_and_padding(self.size_list[-1]))
            self.net.add_module('pyramid-{}'.format(i), blk)
            print('l{}: indim {} outdim {}'.format(i, filters_in, filters_in*2))

        # add output layer
        self.size_list.append(calc_output_size_and_padding(self.size_list[-1], 3, 1, 1))
        self.net.add_module('head', nn.Conv2d(hp.ngf * 2 ** (self.n - 3), hp.nz_enc, 3, padding=1, stride=1))
        print('l out: indim {} outdim {}'.format(hp.ngf * 2 ** (self.n - 3), hp.nz_enc))

        self.net.apply(init_weights_xavier)

    def get_output_size(self):
        return list(map(int, self.size_list[-1]))

    def forward(self, input):
        return self.net(input)

class ConvDecoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp

        self.n = get_num_conv_layers(self.img_sz)
        self.net = GetIntermediatesSequential(hp.skips_stride) if hp.use_skips else nn.Sequential()

#         print('l-1: indim {} outdim {}'.format(64, hp./))
        self.net.add_module('head', nn.ConvTranspose2d(64, 32, 4))
        
        
        for i in range(self.n - 3):
            filters_in = 32 // 2 ** i
            self.net.add_module('pyramid-{}'.format(i),
                                ConvBlockDec(in_dim=filters_in, out_dim=filters_in // 2, normalize=hp.apply_dataset_normalization))
            print('l{}: indim {} outdim {}'.format(i, filters_in, filters_in // 2))

        self.net.add_module('input', ConvBlockDec(in_dim=8, out_dim=hp.input_nc, normalization=None))

        # add output layer
        
#         print('l out: indim {} outdim {}'.format(hp.ngf * 2 ** (self.n - 3), hp.nz_enc))

        self.net.apply(init_weights_xavier)

    def get_output_size(self):
        # return (self._hp.nz_enc, self._hp.img_sz[0]//(2**self.n), self._hp.img_sz[1]//(2**self.n))
        return (3, 64, 64)   # todo calc this, fix the padding in the convs!

    def forward(self, input):
        return self.net(input)