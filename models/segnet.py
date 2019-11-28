from torch import nn

from .networks import SegNet2D_Seg, SegNet2D_Rec,\
                      SegNet3D_Seg, SegNet3D_Rec,\
                      FeatureExtract


class SegNet2D(nn.Module):
    def __init__(self, opt):
        super(SegNet2D, self).__init__()
        self.feature = FeatureExtract(opt)
        expansion = self.feature.expansion
        self.seg = SegNet2D_Seg(expansion)
        self.rec = SegNet2D_Rec(expansion)

    def forward(self, x):
        c1, c2, c3, c4 = self.feature(x)
        seg = self.seg(c1, c2, c3, c4, x)
        rec = self.rec(c1, c2, c3, c4, x)
        return seg, rec


class SegNet3D(nn.Module):
    def __init__(self, opt):
        super(SegNet3D, self).__init__()
        self.feature = FeatureExtract(opt)
        expansion = self.feature.expansion
        self.seg = SegNet3D_Seg(expansion)
        self.rec = SegNet3D_Rec(expansion)

    def forward(self, x):
        c1, c2, c3, c4 = self.feature(x)
        seg = self.seg(c1, c2, c3, c4)
        rec = self.rec(c1, c2, c3, c4)
        return seg, rec