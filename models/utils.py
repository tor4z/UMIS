import torch


def norm_ip(img, min, max):
    out = torch.clamp(img, min=min, max=max)
    out = (out - min) / (max - min + 1e-5)
    return out


def norm_range(t, range=None):
    if range is not None:
        return norm_ip(t, range[0], range[1])
    else:
        return norm_ip(t, float(t.min()), float(t.max()))
