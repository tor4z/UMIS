import torch
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F

from .segnet import SegNet2D, SegNet3D
from .grad import Grad2D, Grad3D
from .morphpool.morphpoollayer import MorphPool2D,\
                                      MorphPool3D


class Trainner(nn.Module):
    def __init__(self, opt, summary, saver):
        super(Trainner, self).__init__()
        self.opt = opt
        self.summary = summary
        self.saver = saver
        print('initialize trianner')

    def setup_model(self):
        if self.opt.dim == 2:
            self.model = SegNet2D(self.opt).cuda(self.opt.device)
            self.morph = MorphPool2D().cuda(self.opt.device)
            self.grad_fn = Grad2D().cuda(self.opt.device)
        elif self.opt.dim == 3:
            self.model = SegNet3D(self.opt).cuda(self.opt.device)
            self.morph = MorphPool3D().cuda(self.opt.device)
            self.grad_fn = Grad3D().cuda(self.opt.device)
        else:
            raise ValueError('dim:{}'.format(opt.dim))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)

        if self.opt.cuda:
            self.model = nn.DataParallel(self.model, device_ids=self.opt.devices)
            self.morph = nn.DataParallel(self.morph, device_ids=self.opt.devices)
            self.grad_fn = nn.DataParallel(self.grad_fn, device_ids=self.opt.devices)

    def train_iter(self, data, label):
            data = data.to(self.opt.device)
            if label: label = label.to(self.opt.device)

            # Misc
            dimsum = list(range(1, len(data.shape)))

            # Network
            seg, rec = self.model(data)
            grad_seg = self.grad_fn(seg)
            grad_rec = self.grad_fn(rec)

            # References
            area = seg.sum(dim=dimsum, keepdim=True)
            area_m = (1 - seg).sum(dim=dimsum, keepdim=True)
            c0 = (data * seg).sum(dim=dimsum, keepdim=True) / (area + 1e-8)
            c1 = (data * (1 - seg)).sum(dim=dimsum, keepdim=True) / (area_m + 1e-8)

            # Image force
            image_force = (self.opt.lmd1 * (data - c0).pow(2) - self.opt.lmd2 * (data - c1).pow(2)) * grad_seg

            # Smooth
            for i in range(self.opt.smooth_iter):
                seg = self.morph(seg)
                seg = self.morph(seg, True)

            # Reconstruction loss
            rec_loss = F.mse_loss(rec, data) + grad_rec.mean()

            # Rank loss
            rank_loss = torch.exp(c1 - c0).mean()

            # Variance loss
            if self.opt.dim == 3:
                # seg_mean = area / (seg.shape[1] * seg.shape[2] * seg.shape[3] * seg.shape[4])
                seg_size = seg.shape[1] * seg.shape[2] * seg.shape[3] * seg.shape[4]
            else:
                # seg_mean = area / (seg.shape[1] * seg.shape[2] * seg.shape[3])
                seg_size = seg.shape[1] * seg.shape[2] * seg.shape[3]

            var_loss = (seg.pow(2).sum(dim=dimsum) / seg_size) - (seg.sum(dim=dimsum) / seg_size).pow(2)
            var_loss = torch.exp(var_loss).mean()

            # Entropy loss
            etropy_loss = (- seg * (seg + 1e-5).log()).mean()

            # Image force loss
            one_opt = image_force[image_force < 0]
            one_opt_seg = seg[image_force < 0]
            zero_opt = image_force[image_force > 0]
            zero_opt_seg = seg[image_force > 0]
            image_foce_loss = 0
            if len(one_opt) > 0:
                image_foce_loss += torch.exp(one_opt * one_opt_seg).mean() * 0.5
            if len(zero_opt) > 0:
                image_foce_loss += torch.exp(- zero_opt * (1 - zero_opt_seg)).mean() * 0.5

            # Compound loss
            loss = self.opt.morph_ratio * image_foce_loss
            loss += self.opt.rank_ratio * rank_loss
            loss += self.opt.entropy_ratio * etropy_loss
            loss += self.opt.var_ratio * var_loss
            loss += self.opt.rec_ratio * rec_loss
            area_mean = area.mean()
            loss += self.opt.area_ratio * area_mean

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.global_steps % self.opt.visual_freq == 0:
                loss_scalars = {
                    'loss': loss.item(),
                    'image_foce_loss': image_foce_loss.item(),
                    'rank_loss': rank_loss.item(),
                    'etropy_loss': etropy_loss.item(),
                    'var_loss': var_loss.item(),
                    'rec_loss': rec_loss.item(),
                    'area_mean': area_mean.item()}

                self.summary.train_image(data, seg, rec, label, self.global_steps)
                self.summary.add_scalars('main_loss', loss_scalars, self.global_steps)

    def train_epoch(self, dataloader, epoch):
        self.epoch = epoch
        self.model.train()
        iterator = tqdm(dataloader,
                        leave=True,
                        dynamic_ncols=True)

        for i, data in enumerate(iterator):
            if isinstance(data, tuple):
                data, label = data
            else:
                label = None

            self.global_steps = epoch * len(dataloader) + i
            self.train_iter(data, label)

            iterator.set_description(
                'Epoch [{epoch}/{epochs}]'.format(
                    epoch=epoch, epochs=self.opt.epochs))

    def run(self, train_dataloader, val_dataloader):
        self.resume()
        for epoch in range(self.opt.epochs):
            self.train_epoch(train_dataloader, epoch)
            self.validate_one(val_dataloader)

    def validate_one(self, val_dataloader):
        with torch.no_grad():
            self.model.eval()

            for data in val_dataloader:
                if isinstance(data, tuple):
                    data, label = data
                    lable = lable.to(self.opt.device)
                else:
                    label = None
                data = data.to(self.opt.device)                    

                # Network
                seg, rec = self.model(data)

                # Smooth
                for i in range(self.opt.smooth_iter):
                    seg = self.morph(seg)
                    seg = self.morph(seg, True)

                self.summary.val_image(data, seg, label, self.global_steps)
        
        self.save_checkpoint(0)

    def save_checkpoint(self, pred):
        self.saver.save_checkpoint({
            'epoch': self.epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'pred': pred})

    def load_checkpoint(self, epoch=None, best=False):
        # load latest model by default
        if epoch is not None and best:
            raise ValueError('Ambiguous: epoch is not None best is true')
        if epoch is None and not best:
            state = self.saver.load_latest()
        elif epoch is not None:
            state = self.saver.load_checkpoint(epoch)
        else:
            state = self.saver.load_best()
        
        self.model.module.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

    def resume(self):
        if self.opt.resume:
            if self.opt.resume_best:
                # resume best
                self.load_checkpoint(best=True)
            elif self.opt.resume_latest:
                # resume latest
                self.load_checkpoint(latest=True)
            elif self.opt.resume_epoch is not None:
                # resume epoch
                self.load_checkpoint(epoch=self.opt.resume_epoch)
            else:
                raise ValueError('resume option error, please check cfg.yaml.')

