import torch
from torch import nn

from .segnet import SegNet2D
from .grad import Grad2D, Grad3D
from .morphpool.morphpoollayer import MorphPool2D,\
                                      MorphPool3D


class Trainner(nn.Module):
    def __init__(self, opt, summary, saver):
        self.opt = opt
        self.summary = summary
        self.saver = saver

    def setup_model(self):
        if self.opt.dim == 2:
            self.model = SegNet2D().to(self.opt.device)
            self.morph = MorphPool2D().to(self.opt.device)
            self.grad_fn = Grad2D().to(self.opt.device)
        elif self.opt.dim == 3:
            self.model = SegNet3D().to(self.opt.device)
            self.morph = MorphPool3D().to(self.opt.device)
            self.grad_fn = Grad3D().to(self.opt.device)
        else:
            raise ValueError('dim:{}'.format(opt.dim))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)

        if self.opt.cuda:
            self.model = nn.DataParallel(self.model, device_ids=self.opt.devices)
            self.morph = nn.DataParallel(self.morph, device_ids=self.opt.devices)
            self.grad_fn = nn.DataParallel(self.grad_fn, device_ids=self.opt.devices)

    def resume(self):
        pass

    def train_iter(self, data):
        data = data.to(self.opt.device)

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
        image_force = (args.lmd1 * (data - c0).pow(2) - args.lmd2 * (data - c1).pow(2)) * grad_seg

        # Smooth
        for i in range(self.opt.smooth_iter):
            seg = self.morph(seg)
            seg = self.morph(seg, True)


        # Reconstruction loss
        rec_loss = F.mse_loss(rec, data) + grad_rec.mean()

        # Rank loss
        rank_loss = torch.exp(c1 - c0).mean()

        # Variance loss
        seg_mean = area / (seg.shape[1] * seg.shape[2] * seg.shape[3] * seg.shape[4])
        seg_size = seg.shape[1] * seg.shape[2] * seg.shape[3] * seg.shape[4]
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
        loss = image_foce_loss
        loss += 1e-2 * rank_loss
        loss += 1e-3 * etropy_loss
        loss += 1e-3 * var_loss
        loss += 1e-6 * rec_loss
        area_mean = area.mean()
        loss += 5e-8 * area_mean

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.global_steps % self.opt.visual_freq == 0:
            self.visualize()

        return loss

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        iterator = tqdm(dataloader,
                        leave=True,
                        dynamic_ncols=True)
        
        for i, (data, _) in enumerate(iterator):
            self.global_steps = epoch * len(dataloader) + i

            loss = self.train_iter(data)

            iterator.set_description(
                'Epoch [{epoch}/{epochs}] :: Train Loss {loss:.4f}'.format(
                    epoch=epoch, epochs=self.opt.epochs, loss=loss.item()))            

    def validate(self):
        if args.validate:
            # Validate
            with torch.no_grad():
                model.eval()
                iterator = tqdm(dataloader_val,
                                leave=True,
                                dynamic_ncols=True,
                                desc='Validation ::')
                input = dataset_val.img[
                        dataset_val.effective_lable_idx[0][0]:dataset_val.effective_lable_idx[0][1],
                        dataset_val.effective_lable_idx[1][0]:dataset_val.effective_lable_idx[1][1],
                        dataset_val.effective_lable_idx[2][0]:dataset_val.effective_lable_idx[2][1]
                        ]
                input_gt = dataset_val.lbl[
                           dataset_val.effective_lable_idx[0][0]:dataset_val.effective_lable_idx[0][1],
                           dataset_val.effective_lable_idx[1][0]:dataset_val.effective_lable_idx[1][1],
                           dataset_val.effective_lable_idx[2][0]:dataset_val.effective_lable_idx[2][1]
                           ]
                input_gt = input_gt // input_gt.max()

                output = np.zeros((1,
                                   dataset_val.effective_lable_shape[0],
                                   dataset_val.effective_lable_shape[1],
                                   dataset_val.effective_lable_shape[2]))
                idx_sum = np.zeros((1,
                                    dataset_val.effective_lable_shape[0],
                                    dataset_val.effective_lable_shape[1],
                                    dataset_val.effective_lable_shape[2]))

                for index, (data, lables) in enumerate(iterator):
                    # To CUDA
                    data = data.cuda(device)
                    lables = lables.cuda(device)

                    # Network
                    seg, _ = model(data)

                    # Smooth
                    seg = mp3d(seg)
                    seg = mp3d(seg, True)
                    seg = mp3d(seg)
                    seg = mp3d(seg, True)
                    # seg = mp3d(seg)
                    # seg = mp3d(seg, True)

                    for batch_idx, val in enumerate(seg[:, 0]):
                        out_i = index * dataloader_val.batch_size + batch_idx
                        z, y, x = np.unravel_index(out_i, (dataset_val.dz, dataset_val.dy, dataset_val.dx))
                        z = z * dataset_val.stride[0]
                        y = y * dataset_val.stride[1]
                        x = x * dataset_val.stride[2]

                        idx_sum[0,
                        z: z + dataset_val.lables_shape[0],
                        y: y + dataset_val.lables_shape[1],
                        x: x + dataset_val.lables_shape[2]] += 1

                        output[0,
                        z: z + dataset_val.lables_shape[0],
                        y: y + dataset_val.lables_shape[1],
                        x: x + dataset_val.lables_shape[2]] += val.cpu().data.numpy()

                output = output / idx_sum
                output = torch.Tensor(output).unsqueeze(0).cuda(device)
                input_gt = torch.Tensor(input_gt).unsqueeze(0)

                # Normalize
                output = norm_range(output)

            # Plot
            input = torch.Tensor(input).unsqueeze(0).unsqueeze(0)
            summary.visualize_image_val(writer, input, output, epoch)
            
        self.save_checkpoint()

    def save_checkpoint(self):
        pass

    def visualize(self):
        pass
