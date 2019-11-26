from torch import nn

from .networks import SegNet2D_Seg, SegNet2D_Rec,
                      SegNet3D_Seg, SegNet3D_Rec, 
                      FeatureExtract


class SegNet2D(nn.modules):
    def __init__(self, opt):
        super(SegNet2D, self).__init__()
        self.feature = FeatureExtract(opt)
        self.seg = SegNet2D_Seg()
        self.rec = SegNet2D_Rec()

    def forward(self, x):
        c1, c2, c3, c4 = self.feature(X)
        seg = self.seg(c1, c2, c3, c4)
        rec = self.rec(c1, c2, c3, c4)
        return seg, rec


class SegNet3D(nn.modules):
    def __init__(self, opt):
        super(SegNet3D, self).__init__()
        self.feature = FeatureExtract(opt)
        self.seg = SegNet3D_Seg()
        self.rec = SegNet3D_Rec()

    def forward(self, x):
        c1, c2, c3, c4 = self.feature(X)
        seg = self.seg(c1, c2, c3, c4)
        rec = self.rec(c1, c2, c3, c4)
        return seg, rec