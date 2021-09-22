from torch.autograd import Function

import numpy as np

def compute_alpha(inputs, alpha_delay):
    done_ratio = max(inputs.global_step - alpha_delay, 0) / (inputs.max_iterations - alpha_delay)
    alpha = 2. / (1. + np.exp(-10 * done_ratio)) - 1
    return alpha


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None