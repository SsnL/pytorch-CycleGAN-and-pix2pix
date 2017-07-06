import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks, mean_estimator
import sys

class LatentCycleGANModel(BaseModel):
    def name(self):
        return 'LatentCycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        self.size = size = opt.fineSize
        self.latent_nc = opt.latent_nc
        self.latent_z = opt.latent_z

        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.sampled_real_z = self.Tensor(nb * 2, opt.latent_z)

        # load/define networks

        self.netG_A_to_latent, self.netG_A_from_latent = networks.define_G(
                                        opt.input_nc, opt.output_nc, opt.ngf,
                                        opt.which_model_netG, opt.norm,
                                        opt.use_dropout, self.gpu_ids, size,
                                        opt.latent_nc, opt.latent_z)
        self.netG_B_to_latent, self.netG_B_from_latent = networks.define_G(
                                        opt.output_nc, opt.input_nc, opt.ngf,
                                        opt.which_model_netG, opt.norm,
                                        opt.use_dropout, self.gpu_ids, size,
                                        opt.latent_nc, opt.latent_z)

        self.estimate_weight = bool(opt.weight_transform)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_latent = networks.define_D(opt.latent_nc, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netD_z = networks.define_D_1d(opt.latent_z,
                                               opt.which_model_netD,
                                               opt.n_layers_D, use_sigmoid,
                                               self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A_to_latent, 'G_A_to_latent', which_epoch)
            self.load_network(self.netG_B_from_latent, 'G_B_from_latent', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_latent, 'D_latent', which_epoch)
                self.load_network(self.netD_z, 'D_z', which_epoch)

        if self.isTrain:
            self.latent_A_pool = ImagePool(opt.pool_size)
            self.latent_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(
                    self.netG_A_to_latent.parameters(),
                    self.netG_A_from_latent.parameters(),
                    self.netG_B_to_latent.parameters(),
                    self.netG_B_from_latent.parameters(),
                ),
                lr=self.current_lr, betas=(opt.beta1, 0.999),
            )
            self.optimizer_D_latent = torch.optim.Adam(self.netD_latent.parameters(),
                                                lr=self.current_lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_z = torch.optim.Adam(self.netD_z.parameters(),
                                                lr=self.current_lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_A_to_latent)
            networks.print_network(self.netG_A_from_latent)
            networks.print_network(self.netG_B_to_latent)
            networks.print_network(self.netG_B_from_latent)
            networks.print_network(self.netD_latent)
            networks.print_network(self.netD_z)
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        nzs = input_A.size(0) + input_B.size(0)
        real_z = torch.randn(nzs, self.latent_z)
        self.sampled_real_z.resize_(real_z.size()).copy_(real_z)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_z = Variable(self.sampled_real_z)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B  = self.netG_A.forward(self.fake_A)

    #get image paths
    def get_image_paths_at(self, i):
        replicate = 4
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
        pred_fake = netD.forward(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_latent(self):
        latent_A = self.latent_A_pool.query(self.latent_A)
        latent_B = self.latent_B_pool.query(self.latent_B)
        self.loss_D_latent = self.backward_D_basic(self.netD_latent, latent_A.detach(), latent_B.detach())

    def backward_D_z(self):
        fake_z = torch.cat((self.z_A, self.z_B), 0)
        self.loss_D_z =  self.backward_D_basic(self.netD_z, self.real_z, fake_z.detach())

    def backward_G(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Loss
        # A
        self.latent_A, self.z_A = self.netG_A_to_latent.forward(self.real_A)
        self.rec_A = self.forward_from_latent(self.latent_A, self.z_A, self.netG_B_from_latent)
        self.loss_G_latent_A, self.loss_G_z_A, self.loss_cycle_A = self.get_loss( \
            self.real_A, self.latent_A, self.z_A, self.rec_A, False, True, lambda_A)
        # B
        self.latent_B, self.z_B = self.netG_B_to_latent.forward(self.real_B)
        self.rec_B = self.forward_from_latent(self.latent_B, self.z_B, self.netG_A_from_latent)
        self.loss_G_latent_B, self.loss_G_z_B, self.loss_cycle_B = self.get_loss( \
            self.real_B, self.latent_B, self.z_B, self.rec_B, True, True, lambda_B)
        # combined loss
        self.loss_G = \
            self.loss_G_latent_A + self.loss_G_latent_B + \
            self.loss_G_z_A + self.loss_G_z_B + \
            self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()

    def forward_from_latent(self, latent, z, netG_from_latent):
        k = latent.size(0)
        z_expanded = z[:, :, None, None].expand(k, self.latent_z, self.size, self.size)
        input_latent_z = torch.cat((latent, z_expanded), 1)
        return netG_from_latent.forward(input_latent_z)

    def get_loss(self, real, latent, z, rec, target_latent_label, target_z_label, lamda):
        latent_pred_fake = self.netD_latent.forward(latent)
        loss_G_latent = self.criterionGAN(latent_pred_fake, target_latent_label)
        z_pred_fake = self.netD_z.forward(z)
        loss_G_z = self.criterionGAN(z_pred_fake, target_z_label)
        loss_cycle = self.criterionCycle(rec, real) * lamda
        return loss_G_latent, loss_G_z, loss_cycle

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_latent
        self.optimizer_D_latent.zero_grad()
        self.backward_D_latent()
        self.optimizer_D_latent.step()
        # D_z
        self.optimizer_D_z.zero_grad()
        self.backward_D_z()
        self.optimizer_D_z.step()


    def get_current_errors(self):
        D_latent = self.loss_D_latent.data[0]
        D_z = self.loss_D_z.data[0]
        G_latent_A = self.loss_G_latent_A.data[0]
        G_z_A = self.loss_G_z_A.data[0]
        Cyc_A = self.loss_cycle_A.data[0]
        G_latent_B = self.loss_G_latent_B.data[0]
        G_z_B = self.loss_G_z_B.data[0]
        Cyc_B = self.loss_cycle_B.data[0]
        errors = OrderedDict()
        errors['D_latent'] = D_latent
        errors['D_z'] = D_z
        errors['G_latent_A'] = G_latent_A
        errors['G_z_A'] = G_z_A
        errors['Cyc_A'] = Cyc_A
        errors['G_latent_B'] = G_latent_B
        errors['G_z_B'] = G_z_B
        errors['Cyc_B'] = Cyc_B
        return errors

    def z_str(self, z_single, precision = 2):
        return np.array_str(z_single.data.cpu().float().numpy(), precision = precision)
        # return '<span style="font-family: monospace;font-size: 8px;">{}</span>'.format(s)

    def get_current_visuals(self):
        self.real_z_A = self.real_z[:self.latent_A.size(0)]
        self.fake_B = self.forward_from_latent(self.latent_A, real_z_A, \
            self.netG_A_from_latent)
        self.real_z_B = self.real_z[self.latent_A.size(0):]
        self.fake_A = self.forward_from_latent(self.latent_B, real_z_B, \
            self.netG_B_from_latent)
        return super().get_current_visuals(self)

    def get_current_visuals_at(self, i)
        visuals = OrderedDict()
        if i < self.real_A.size(0):
            visuals['real_A{}'.format(self.z_str(self.z_A[i]))] = util.tensor2im(self.real_A.data, i)
            if self.latent_nc == 3:
                visuals['latent_A'] = util.tensor2im(self.latent_A.data, i)
            elif self.latent_nc == 1:
                visuals['latent_A'] = util.tensor2im(self.latent_A.data, i).repeat(3, 2)
            visuals['fake_B{}'.format(self.z_str(self.real_z_A[i]))] = util.tensor2im(fake_B.data, i)
            visuals['rec_A'] = util.tensor2im(self.rec_A.data, i)
        if i < self.real_B.size(0):
            visuals['real_B{}'.format(self.z_str(self.z_B, [i]))] = util.tensor2im(self.real_B.data, i)
            if self.latent_nc == 3:
                visuals['latent_B'] = util.tensor2im(self.latent_B.data, i)
            elif self.latent_nc == 1:
                visuals['latent_B'] = util.tensor2im(self.latent_B.data, i).repeat(3, 2)
            visuals['fake_A{}'.format(self.z_str(real_z_B[i]))] = util.tensor2im(fake_A.data, i)
            visuals['rec_B'] = util.tensor2im(self.rec_B.data, i)
        return visuals

    def save(self, label):
        self.save_network(self.netD_latent, 'D_latent', label, self.gpu_ids)
        self.save_network(self.netD_z, 'D_z', label, self.gpu_ids)
        self.save_network(self.netG_A_to_latent, 'G_A_to_latent', label, self.gpu_ids)
        self.save_network(self.netG_A_from_latent, 'G_A_from_latent', label, self.gpu_ids)
        self.save_network(self.netG_B_to_latent, 'G_B_to_latent', label, self.gpu_ids)
        self.save_network(self.netG_B_from_latent, 'G_B_from_latent', label, self.gpu_ids)

    def set_learning_rate(self, lr):
        for param_group in self.optimizer_D_latent.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_z.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
