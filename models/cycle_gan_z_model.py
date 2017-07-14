import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from .latent_cycle_gan_model import LatentCycleGANModel
from . import networks, mean_estimator
import sys

class CycleGANZModel(BaseModel):
    def name(self):
        return 'CycleGANZModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        self.size = size = opt.fineSize
        self.latent_z = opt.latent_z
        self.A_nc = opt.input_nc
        self.B_nc = opt.output_nc
        self.image_z_cycle_ratio = opt.image_z_cycle_ratio
        self.z_cycle_multiplier = opt.z_cycle_multiplier
        assert(0 <= self.image_z_cycle_ratio <= 1)
        assert(0 <= self.z_cycle_multiplier)

        self.input_A_im = self.Tensor(nb, opt.input_nc, size, size)
        self.sampled_A_z = self.Tensor(nb, opt.latent_z)
        self.input_B_im = self.Tensor(nb, opt.output_nc, size, size)
        self.sampled_B_z = self.Tensor(nb, opt.latent_z)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc + opt.latent_z,
                                        opt.output_nc, opt.ngf,
                                        opt.which_model_netG, opt.norm,
                                        opt.use_dropout, size,
                                        None, opt.latent_z, opt.norm_first,
                                        gpu_ids = self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc + opt.latent_z,
                                        opt.input_nc, opt.ngf,
                                        opt.which_model_netG, opt.norm,
                                        opt.use_dropout, size,
                                        None, opt.latent_z, opt.norm_first,
                                        gpu_ids = self.gpu_ids)


        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm,
                                            use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm,
                                            use_sigmoid, self.gpu_ids)
            self.netD_z = networks.define_D_1d(opt.latent_z,
                                               opt.which_model_netD,
                                               opt.n_layers_D, use_sigmoid,
                                               self.gpu_ids)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

        else:
            self.netD_A = self.netD_B = self.mean_estimator_A = self.mean_estimator_B = None
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_z_pool = ImagePool(opt.pool_size * 2)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=self.current_lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                                lr=self.current_lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(),
                                                lr=self.current_lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_z = torch.optim.Adam(self.netD_z.parameters(),
                                                lr=self.current_lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            networks.print_network(self.netD_z)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        im_A = input['A' if AtoB else 'B']
        im_B = input['B' if AtoB else 'A']
        self.input_A_im.resize_(im_A.size()).copy_(im_A)
        self.input_B_im.resize_(im_B.size()).copy_(im_B)
        z_A = torch.randn(im_A.size(0), self.latent_z)
        z_B = torch.randn(im_B.size(0), self.latent_z)
        self.sampled_A_z.resize_(z_A.size()).copy_(z_A)
        self.sampled_B_z.resize_(z_B.size()).copy_(z_B)
        self.A_paths = input['A_paths' if AtoB else 'B_paths']
        self.B_paths = input['B_paths' if AtoB else 'A_paths']

    def im_z_to_input(self, im, z):
        return torch.cat((im, z[:, :, None, None].expand(z.size(0), self.latent_z, self.size, self.size)), 1)

    def forward(self, volatile = False):
        self.real_A_im = Variable(self.input_A_im, volatile = volatile)
        self.real_A_z = Variable(self.sampled_A_z, volatile = volatile)
        self.real_A = self.im_z_to_input(self.real_A_im, self.real_A_z)

        self.real_B_im = Variable(self.input_B_im, volatile = volatile)
        self.real_B_z = Variable(self.sampled_B_z, volatile = volatile)
        self.real_B = self.im_z_to_input(self.real_B_im, self.real_B_z)

        self.fake_B_im, self.fake_B_z = self.netG_A.forward(self.real_A)
        self.fake_B = self.im_z_to_input(self.fake_B_im, self.fake_B_z)
        self.rec_A_im, self.rec_A_z = self.netG_B.forward(self.fake_B)

        self.fake_A_im, self.fake_A_z = self.netG_B.forward(self.real_B)
        self.fake_A = self.im_z_to_input(self.fake_A_im, self.fake_A_z)
        self.rec_B_im, self.rec_B_z = self.netG_A.forward(self.fake_A)

    def test(self):
        self.forward(True)

    # get image paths
    def get_image_paths_at(self, i):
        replicate = 4 if self.opt.identity > 0 else 3
        image_paths = []
        if i < self.input_A.size(0):
            image_paths += [self.A_paths[i]] * replicate
        if i < self.input_B.size(0):
            image_paths += [self.B_paths[i]] * replicate
        return image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B_im = self.fake_B_pool.query(self.fake_B_im)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B_im, fake_B_im)

    def backward_D_B(self):
        fake_A_im = self.fake_A_pool.query(self.fake_A_im)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A_im, fake_A_im)

    def backward_D_z(self):
        fake_z = self.fake_z_pool.query(torch.cat((self.fake_A_z, self.fake_B_z), 0))
        real_z = torch.cat((self.real_A_z, self.real_B_z), 0)
        self.loss_D_z = self.backward_D_basic(self.netD_z, real_z, fake_z)

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A_im, self.idt_A_z = self.netG_A.forward(self.real_B)
            self.idt_A = self.im_z_to_input(self.idt_A_im, self.idt_A_z)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B_im, self.idt_B_z = self.netG_B.forward(self.real_A)
            self.idt_B = self.im_z_to_input(self.idt_B_im, self.idt_B_z)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # Loss
        # A
        self.loss_G_A_im, self.loss_G_A_z = self.lerp(
            self.criterionGAN(self.netD_A.forward(self.fake_B_im), True),
            self.criterionGAN(self.netD_z.forward(self.fake_B_z), True),
            self.image_z_cycle_ratio,
        )
        self.loss_cycle_A_im, self.loss_cycle_A_z = self.lerp(
            self.criterionCycle(self.rec_A_im, self.real_A_im),
            self.criterionCycle(self.rec_A_z, self.real_A_z) * self.z_cycle_multiplier,
            self.image_z_cycle_ratio,
            lambda_A,
        )
        # B
        self.loss_G_B_im, self.loss_G_B_z = self.lerp(
            self.criterionGAN(self.netD_B.forward(self.fake_A_im), True),
            self.criterionGAN(self.netD_z.forward(self.fake_A_z), True),
            self.image_z_cycle_ratio,
        )
        self.loss_cycle_B_im, self.loss_cycle_B_z = self.lerp(
            self.criterionCycle(self.rec_B_im, self.real_B_im),
            self.criterionCycle(self.rec_B_z, self.real_B_z) * self.z_cycle_multiplier,
            self.image_z_cycle_ratio,
            lambda_B,
        )
        # combined loss
        self.loss_G = self.loss_G_A_im + self.loss_G_B_im + \
            self.loss_G_A_z + self.loss_G_B_z + \
            self.loss_cycle_A_im + self.loss_cycle_B_im + \
            self.loss_cycle_A_z + self.loss_cycle_B_z + \
            self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def lerp(self, oa, ob, alpha, scale = 1):
        a = oa * alpha * scale
        b = ob * (1 - alpha) * scale
        return a, b


    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
        # D_z
        self.optimizer_D_z.zero_grad()
        self.backward_D_z()
        self.optimizer_D_z.step()

    def get_current_errors(self):
        D_A = self.loss_D_A.data[0]
        G_A_im = self.loss_G_A_im.data[0]
        G_A_z = self.loss_G_A_z.data[0]
        Cyc_A_im = self.loss_cycle_A_im.data[0]
        Cyc_A_z = self.loss_cycle_A_z.data[0]
        D_B = self.loss_D_B.data[0]
        G_B_im = self.loss_G_B_im.data[0]
        G_B_z = self.loss_G_B_z.data[0]
        Cyc_B_im = self.loss_cycle_B_im.data[0]
        Cyc_B_z = self.loss_cycle_B_z.data[0]
        D_z = self.loss_D_z.data[0]
        errors = OrderedDict()
        errors['D_A'] = D_A
        errors['G_A_im'] = G_A_im
        errors['G_A_z'] = G_A_z
        errors['Cyc_A_im'] = Cyc_A_im
        errors['Cyc_A_z'] = Cyc_A_z
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data[0]
            errors['idt_A'] = idt_A
        errors['D_B'] = D_B
        errors['G_B_im'] = G_B_im
        errors['G_B_z'] = G_B_z
        errors['Cyc_B_im'] = Cyc_B_im
        errors['Cyc_B_z'] = Cyc_B_z
        if self.opt.identity > 0.0:
            idt_B = self.loss_idt_B.data[0]
            errors['idt_B'] = idt_B
        errors['D_z'] = D_z
        return errors

    def get_current_visuals_at(self, i):
        visuals = OrderedDict()
        if i < self.real_A.size(0):
            visuals['real_A', LatentCycleGANModel.z_str(self.real_A_z[i])] = \
                util.tensor2im(self.real_A[:, :self.A_nc].data, i)
            visuals['fake_B', LatentCycleGANModel.z_str(self.fake_B_z[i])] = \
                util.tensor2im(self.fake_B_im.data, i)
            visuals['rec_A', LatentCycleGANModel.z_str(self.rec_A_z[i])] = \
                util.tensor2im(self.rec_A_im.data, i)
            if self.opt.identity > 0.0:
                visuals['idt_A', LatentCycleGANModel.z_str(self.idx_A_z[i])] = \
                    util.tensor2im(self.idt_A_im.data, i)
        if i < self.real_B.size(0):
            visuals['real_B', LatentCycleGANModel.z_str(self.real_B_z[i])] = \
                util.tensor2im(self.real_B[:, :self.B_nc].data, i)
            visuals['fake_A', LatentCycleGANModel.z_str(self.fake_A_z[i])] = \
                util.tensor2im(self.fake_A_im.data, i)
            visuals['rec_B', LatentCycleGANModel.z_str(self.rec_B_z[i])] = \
                util.tensor2im(self.rec_B_im.data, i)
            if self.opt.identity > 0.0:
                visuals['idt_B', LatentCycleGANModel.z_str(self.idt_B_z[i])] = \
                    util.tensor2im(self.idt_B_im.data, i)
        return visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netD_z, 'D_z', label, self.gpu_ids)

    def set_learning_rate(self, lr):
        for param_group in self.optimizer_D_z.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
