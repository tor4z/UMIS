from itertools import cycle
from scipy import ndimage as ndi
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np


class Fcycle(object):

    def __init__(self, funcs):
        """Call functions from the iterable each time it is called."""
        self.funcs = funcs
        self.i = 0

    def __call__(self, img, masks):
        f = self.funcs[self.i % 2]
        self.i += 1
        return f(img, masks)


def sup_inf(img, masks):
    layers = []
    device = img.get_device()
    img = img.cpu().numpy()
    masks = masks.cpu().numpy()

    for mask in masks:
        img[0] = ndi.binary_dilation(img[0], mask).astype(np.int8)
        layers.append(img)
    result = np.array(layers).min(0)
    result = torch.autograd.Variable(torch.Tensor(result), requires_grad=True)
    return result.cuda(device)


def inf_sup(img, masks):
    layers = []
    device = img.get_device()
    img = img.cpu().numpy()
    masks = masks.cpu().numpy()

    for mask in masks:
        img[0] = ndi.binary_erosion(img[0], mask).astype(np.int8)
        layers.append(img)
    result = np.array(layers).max(0)
    result = torch.autograd.Variable(torch.Tensor(result), requires_grad=True)
    return result.cuda(device)


cyc = Fcycle([lambda x, masks: inf_sup(sup_inf(x, masks), masks),
               lambda x, masks: sup_inf(inf_sup(x, masks), masks)])


class MorphPoolFunction(Function):
    @staticmethod
    def forward(ctx, input, aux, mask, device):
        with torch.no_grad():
            outputs = input
            mask = mask.cuda(device)
            size = outputs.size(0)
            
            for i in range(size):
                outputs[i][aux[i] < 0] = 1
                outputs[i][aux[i] > 0] = 0
                outputs[i] = cyc(outputs[i], mask)

        ctx.save_for_backward(input, aux, mask, outputs)

        return outputs

    @staticmethod
    def backward(ctx, grad):
        input, aux, mask, output = ctx.saved_variables
        return (grad, None, None, None, None)


class MorphPool3D(nn.Module):
    def __init__(self):
        super(MorphPool3D, self).__init__()
        _P3 = [np.zeros((3, 3, 3)) for i in range(9)]
        _P3[0][:, :, 1] = 1
        _P3[1][:, 1, :] = 1
        _P3[2][1, :, :] = 1
        _P3[3][:, [0, 1, 2], [0, 1, 2]] = 1
        _P3[4][:, [0, 1, 2], [2, 1, 0]] = 1
        _P3[5][[0, 1, 2], :, [0, 1, 2]] = 1
        _P3[6][[0, 1, 2], :, [2, 1, 0]] = 1
        _P3[7][[0, 1, 2], [0, 1, 2], :] = 1
        _P3[8][[0, 1, 2], [2, 1, 0], :] = 1

        self.register_buffer('morph', torch.Tensor(_P3))

    def forward(self, input, aux, device):
            x = MorphPoolFunction.apply(input, aux, self.morph, device)
            return x
