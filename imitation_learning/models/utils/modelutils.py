import torch

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_one_hot(nb_digits, active_dim):
    """
    param: active_dim: B tensor with indices that need to be set to 1
    """
    active_dim = active_dim.type(torch.LongTensor)
    batch_size = active_dim.shape[0]
    y_onehot = torch.FloatTensor(batch_size, nb_digits)
    y_onehot.zero_()
    y_onehot.scatter_(1, active_dim[:, None], 1)
    return y_onehot
