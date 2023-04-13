import os
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
from torch.autograd import Variable

import torch
from torch.distributions import kl_divergence, Normal


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def show_plot(points, output_dir, mode):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt_path = os.path.join(output_dir, '{}_loss.png'.format(mode))
    plt.savefig(plt_path)


def sequence_mask(length, max_length=None):
    """Tensorflow의 sequence_mask"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def get_masked_with_pad_tensor(size, src, trg, pad_token):
    """
    :param size: the size of target input
    :param src: source tensor
    :param trg: target tensor
    :param pad_token: pad token
    :return:
    """
    src = src[:, None, None, :]
    trg = trg[:, None, None, :]
    src_pad_tensor = torch.ones_like(src).to(src.device.type) * pad_token
    src_mask = torch.equal(src, src_pad_tensor)
    trg_mask = torch.equal(src, src_pad_tensor)
    if trg is not None:
        trg_pad_tensor = torch.ones_like(trg).to(trg.device.type) * pad_token
        dec_trg_mask = trg == trg_pad_tensor
        # boolean reversing i.e) True * -1 + 1 = False
        seq_mask = ~sequence_mask(torch.arange(1, size+1).to(trg.device), size)
        # look_ahead_mask = torch.max(dec_trg_mask, seq_mask)
        look_ahead_mask = dec_trg_mask | seq_mask

    else:
        trg_mask = None
        look_ahead_mask = None

    return src_mask, trg_mask, look_ahead_mask


def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N

def to_cuda_variable(tensor):
    """
    Converts tensor to cuda variable
    :param tensor: torch tensor, of any size
    :return: torch Variable, of same size as tensor
    """
    if torch.cuda.is_available():
        return Variable(tensor).cuda()
    else:
        return Variable(tensor)


def to_cuda_variable_long(tensor):
    """
    Converts tensor to cuda variable
    :param tensor: torch tensor, of any size
    :return: torch Variable, of same size as tensor
    """
    if torch.cuda.is_available():
        return Variable(tensor.long()).cuda()
    else:
        return Variable(tensor.long())


def to_numpy(variable: Variable):
    """
    Converts torch Variable to numpy nd array
    :param variable: torch Variable, of any size
    :return: numpy nd array, of same size as variable
    """
    if torch.cuda.is_available():
        return variable.data.cpu().numpy()
    else:
        return variable.data.numpy()


def init_hidden_lstm(num_layers, batch_size, lstm_hidden_size):
    hidden = (
        to_cuda_variable(
            torch.zeros(num_layers, batch_size, lstm_hidden_size)
        ),
        to_cuda_variable(
            torch.zeros(num_layers, batch_size, lstm_hidden_size)
        )
    )
    return hidden