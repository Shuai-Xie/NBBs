import torch
import math
from torch.autograd import Variable
import numpy as np
from PIL import Image
import numpy as np


def error_map(A, B):
    error_map = (A - B).pow(2).sum(1, keepdim=True) / A.size(1)
    return error_map.pow(0.5)


def patch_distance(A, B, average=True):
    error = (A - B).pow(2).sum()
    if average == False:
        return error
    else:
        tensor_volume = A.size(0) * A.size(1) * A.size(2) * A.size(3)
        return error / tensor_volume


# l2-norm, feature vector -> 单位向量
def normalize_per_pix(A):
    return A / A.pow(2).sum(1, keepdim=True).pow(0.5).expand_as(A)  # keep_dim 保持 dim1 还在，然后 expand_as(A)


def normalize_tensor(A):
    return A / np.power(A.pow(2).sum(), 0.5)


def spatial_distance(point_A, point_B):
    # return (point_A - point_B).pow(2).sum().pow(0.5)
    return (point_A - point_B).pow(2).sum()  # 判断 d=0, sum 即可


# l2-norm, 1,C,H,W -> 1,1,H,W
def response(F, style='l2'):
    # F: feature map
    if style == 'max':
        [response, indices] = F.max(1, keepdim=True)
    elif style == 'l2':
        response = F.pow(2).sum(1, keepdim=True).pow(0.5)  # B,C,H,W; sum(1) L2-norm
    else:
        raise ValueError("unknown response style: ", style)
    return response


def stretch_tensor_0_to_1(F):
    assert (F.dim() == 4)  # F: response 1,1,H,W
    max_val = F.max()
    min_val = F.min()
    if max_val != min_val:
        F_normalized = (F - min_val) / (max_val - min_val)
    else:
        F_normalized = F.fill_(0)
    return F_normalized


def FA_to_HA_norm(F):  # l2 + min_max norm
    F = F.pow(2).sum(1, keepdim=True).pow(0.5)  # B,C,H,W; sum(1) L2-norm
    min_val, max_val = F.min(), F.max()
    if max_val != min_val:
        F_normalized = (F - min_val) / (max_val - min_val)
    else:
        F_normalized = F.fill_(0)
    return F_normalized


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum.expand_as(F) / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    epsilon = 10 ** -20
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3)) + epsilon
    return F_variance.expand_as(F).pow(0.5)


def identity_map(size):
    idnty_map = torch.Tensor(size[0], 2, size[2], size[3])
    idnty_map[0, 0, :, :].copy_(torch.arange(0, size[2]).repeat(size[3], 1).transpose(0, 1))
    idnty_map[0, 1, :, :].copy_(torch.arange(0, size[3]).repeat(size[2], 1))
    return idnty_map


def gaussian(kernel_width, stdv=1):
    w = identity_map([1, 1, kernel_width, kernel_width]) - math.floor(kernel_width / 2)
    kernel = torch.exp(-w.pow(2).sum(1, keepdim=True) / (2 * stdv))
    return kernel
