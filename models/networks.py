import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet3d import resnet as resnet3d
from .resnet2d import resnet as resnet2d


class FeatureExtract(nn.Module):
    def __init__(self, opt):
        super(FeatureExtract, self).__init__()
        if opt.dim == 2:
            resnet = resnet2d
        else:
            resnet = resnet3d

        self.base = resnet(opt)
        self.drop = nn.Dropout(0.2)
        self.expansion = self.base.expansion

    def forward(self, x):
        c1, c2, c3, c4 = self.base(x)
        c4 = self.drop(c4)
        return c1, c2, c3, c4


class SegNet3D_Seg(nn.Module):
    def __init__(self, expansion):
        super(SegNet3D_Seg, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(512 * expansion, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, (1, 3, 3), 1, padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(256 * expansion + 64, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 32, (1, 3, 3), 1, padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(128 * expansion + 32, 8, 4, 2, 1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 8, (1, 3, 3), 1, padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.out_conv = nn.Sequential(
            nn.ConvTranspose3d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, c1, c2, c3, c4, x):
        dc1 = self.deconv1(c4)
        cat = torch.cat((c3, dc1), 1)
        cat = F.interpolate(cat, size=(x.shape[2] // 8, x.shape[3] // 8, x.shape[4] // 8),
                            mode='trilinear', align_corners=False)
        dc2 = self.deconv2(cat)
        _c2 = F.interpolate(c2, size=dc2.shape[2:], mode='trilinear', align_corners=False)
        cat = torch.cat((_c2, dc2), 1)
        out = self.deconv3(cat)
        out = F.interpolate(out, size=x.shape[2:], mode='trilinear', align_corners=False)
        out = self.out_conv(out)
        return out


class SegNet3D_Rec(nn.Module):
    def __init__(self, expansion):
        super(SegNet3D_Rec, self).__init__()
        self.deconv1_rec = nn.Sequential(
            nn.ConvTranspose3d(512 * expansion, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, 1, 1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.deconv2_rec = nn.Sequential(
            nn.ConvTranspose3d(256 * expansion + 64, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 32, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.deconv3_rec = nn.Sequential(
            nn.ConvTranspose3d(128 * expansion + 32, 8, 4, 2, 1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 8, 1, 1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.rec_conv = nn.Sequential(
            nn.ConvTranspose3d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, c1, c2, c3, c4, x):
        dcr1 = self.deconv1_rec(c4)
        dcr2 = self.deconv2_rec(torch.cat((c3, dcr1), 1))
        rec = self.deconv3_rec(torch.cat((c2, dcr2), 1))
        rec = F.interpolate(rec, size=x.shape[2:], mode='trilinear', align_corners=False)
        rec = self.rec_conv(rec)
        return rec


class SegNet2D_Seg(nn.Module):
    def __init__(self, expansion):
        super(SegNet2D_Seg, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512 * expansion, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (3, 3), 1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256 * expansion + 128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (3, 3), 1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            # nn.ConvTranspose2d(128 * expansion + 32, 8, 4, 2, 1),
            nn.ConvTranspose2d(128, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (3, 3), 1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, c1, c2, c3, c4, x):
        dc1 = self.deconv1(c4.detach())
        cat = torch.cat((c3.detach(), dc1), 1)
        cat = F.interpolate(cat, size=(x.shape[2] // 8, x.shape[3] // 8),
                            mode='bilinear', align_corners=False)
        dc2 = self.deconv2(cat)
        # _c2 = F.interpolate(c2, size=dc2.shape[2:], mode='bilinear', align_corners=False)
        # cat = torch.cat((_c2, dc2), 1)
        out = self.deconv3(dc2)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = self.out_conv(out)
        return out


class SegNet2D_Rec(nn.Module):
    def __init__(self, expansion):
        super(SegNet2D_Rec, self).__init__()
        self.deconv1_rec = nn.Sequential(
            nn.ConvTranspose2d(512 * expansion, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.deconv2_rec = nn.Sequential(
            nn.ConvTranspose2d(256 * expansion + 64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv3_rec = nn.Sequential(
            nn.ConvTranspose2d(128 * expansion + 32, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.rec_conv = nn.Sequential(
            nn.ConvTranspose2d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, c1, c2, c3, c4, x):
        dcr1 = self.deconv1_rec(c4)
        dcr2 = self.deconv2_rec(torch.cat((c3, dcr1), 1))
        rec = self.deconv3_rec(torch.cat((c2, dcr2), 1))
        rec = F.interpolate(rec, size=x.shape[2:], mode='bilinear', align_corners=False)
        rec = self.rec_conv(rec)
        return rec


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        if opt.dim == 2:
            self.pool = nn.GlobalAvgPool2D()
            resnet = resnet2d
        else:
            self.pool = nn.GlobalAvgPool3D()
            resnet = resnet3d

        self.base = resnet(opt)
        expansion = self.base.expansion
        self.flatten = nn.Flatten()
        self.liner = nn.Linear(512 * expansion, 1)
    
    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.liner(x)
