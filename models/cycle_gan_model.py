import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys

class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        stochasticity = self.opt.stochasticity
        latent_resnet_block = self.opt.latent_resnet_block

        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        G_input_nc = opt.input_nc
        G_output_nc = opt.output_nc

        if stochasticity and latent_resnet_block < 0:
            # self.random_A = self.Tensor(nb, 1, size, size)
            # self.random_B = self.Tensor(nb, 1, size, size)
            G_input_nc += 1
            G_output_nc += 1
        elif stochasticity and latent_resnet_block >= 0:
            G_input_nc += 1
        elif not stochasticity and latent_resnet_block >= 0:
            pass


        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(G_input_nc, G_output_nc, latent_resnet_block,
                                     opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)
        self.netG_B = networks.define_G(G_input_nc, G_output_nc, latent_resnet_block,
                                    opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)

        if self.isTrain:
            if stochasticity:
                nb *= self.opt.stochasticity_replicate
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(G_output_nc, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(G_output_nc, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            if self.opt.feature_shift_to > 0:
                self.cycle_featurizer_A = networks.define_cycle_featurizer(
                    self.netD_B, opt.cycle_feature_level, self.gpu_ids)
                self.cycle_featurizer_B = networks.define_cycle_featurizer(
                    self.netD_A, opt.cycle_feature_level, self.gpu_ids)
            else:
                self.cycle_featurizer_A = None
                self.cycle_featurizer_B = None
        if not self.isTrain or opt.continue_train:
            self.load_saved_networks(opt.which_epoch[0])

        if self.isTrain:
            self.old_lr = opt.lr
            if self.opt.non_gan_decay_range[1] == 0:
                self.non_gan_weight = self.non_gan_decay_to
            else:
                self.non_gan_weight = 1
            if self.opt.feature_shift_range[1] == 0:
                self.feature_cycle_ratio = self.feature_shift_to
            else:
                self.feature_cycle_ratio = 0
            if self.opt.weight_cycle_niter <= 0:
                self.weight_cycle = False
            else:
                self.weight_cycle = True
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_A)
            networks.print_network(self.netG_B)
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            if self.opt.feature_shift_to > 0:
                networks.print_network(self.cycle_featurizer_A)
                networks.print_network(self.cycle_featurizer_B)
            print('-----------------------------------------------')

    def load_saved_networks(self, which_epoch):
        self.load_network(self.netG_A, 'G_A', which_epoch)
        self.load_network(self.netG_B, 'G_B', which_epoch)
        if self.isTrain:
            self.load_network(self.netD_A, 'D_A', which_epoch)
            self.load_network(self.netD_B, 'D_B', which_epoch)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if self.opt.stochasticity:
            if self.isTrain:
                input_A = torch.cat((input_A,) * self.opt.stochasticity_replicate, 0)
                input_B = torch.cat((input_B,) * self.opt.stochasticity_replicate, 0)
            k, _, w, h = input_A.size()
            # self.random_A.resize_([k, 1, w, h]).copy_(torch.randn(k, 1, w, h))
            if self.opt.latent_resnet_block < 0:
                input_A = torch.cat((input_A, torch.randn(k, 1, w, h)), 1)
            else:
                input_A = torch.cat((input_A, torch.randn(k, 1, 1, 1).expand(k, 1, w, h)), 1)
            k, _, w, h = input_B.size()
            # self.random_B.resize_([k, 1, w, h]).copy_(torch.randn(k, 1, w, h))
            if self.opt.latent_resnet_block < 0:
                input_B = torch.cat((input_B, torch.randn(k, 1, w, h)), 1)
            else:
                input_B = torch.cat((input_B, torch.randn(k, 1, 1, 1).expand(k, 1, w, h)), 1)
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)

        if self.opt.latent_resnet_block >= 0:
            latent_A, self.fake_B = self.netG_A.forward_latent(self.real_A)
            self.rec_A = self.netG_B.forward_from_latent(latent_A)
        else:
            self.fake_B = self.netG_A.forward(self.real_A)
            self.rec_A  = self.netG_B.forward(self.fake_B)

        if self.opt.latent_resnet_block >= 0:
            latent_B, self.fake_A = self.netG_B.forward_latent(self.real_B)
            self.rec_B = self.netG_A.forward_from_latent(latent_B)
        else:
            self.fake_A = self.netG_B.forward(self.real_B)
            self.rec_B  = self.netG_A.forward(self.fake_A)

    #get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        if self.opt.stochasticity and self.opt.latent_resnet_block >= 0:
            real = real[:, :-1]
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
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self, epoch):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_idt * self.non_gan_weight
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_idt * self.non_gan_weight
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))
        if self.opt.latent_resnet_block >= 0:
            latent_A, self.fake_B = self.netG_A.forward_latent(self.real_A)
        else:
            self.fake_B = self.netG_A.forward(self.real_A)
        pred_fake_B = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake_B, True)
        # D_B(G_B(B))
        if self.opt.latent_resnet_block >= 0:
            latent_B, self.fake_A = self.netG_B.forward_latent(self.real_B)
        else:
            self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake_A = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake_A, True)
        if self.opt.latent_resnet_block >= 0:
            self.rec_A = self.netG_B.forward_from_latent(latent_A)
            self.rec_B = self.netG_A.forward_from_latent(latent_B)
            if self.opt.stochasticity:
                # Forward latent loss
                self.loss_rec_A = self.criterionCycle(self.rec_A, self.real_A[:, :-1]) * lambda_A
                # Backward latent loss
                self.loss_rec_B = self.criterionCycle(self.rec_B, self.real_B[:, :-1]) * lambda_B
            else:
                rec_latent_A = self.netG_B.forward_to_latent(self.fake_B)
                rec_latent_B = self.netG_A.forward_to_latent(self.fake_A)
                self.loss_rec_A = \
                    (self.criterionCycle(self.rec_A, self.real_A) + (latent_A - rec_latent_A).abs().mean()) * lambda_A
                self.loss_rec_B = \
                    (self.criterionCycle(self.rec_B, self.real_B) + (latent_A - rec_latent_A).abs().mean()) * lambda_A
        else:
            # Forward cycle loss
            self.rec_A = self.netG_B.forward(self.fake_B)
            self.loss_rec_A = self.cycle_loss(
                self.rec_A,
                self.real_A,
                pred_fake_B,
                self.cycle_featurizer_A,
                lambda_A,
            ) * self.non_gan_weight
            # Backward cycle loss
            self.rec_B = self.netG_A.forward(self.fake_A)
            self.loss_rec_B = self.cycle_loss(
                self.rec_B,
                self.real_B,
                pred_fake_A,
                self.cycle_featurizer_B,
                lambda_B,
            ) * self.non_gan_weight
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_rec_A + self.loss_rec_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def cycle_loss(self, rec, real, pred_fake, featurizer, lamda):
        if lamda == 0:
            return 0
        if self.feature_cycle_ratio > 0:
            rec_feature = featurizer(rec)
            real_feature = featurizer(real).detach()
        k = pred_fake.size()[0]
        l = 0
        for i in range(k):
            li = 0
            if self.feature_cycle_ratio > 0:
                li += self.criterionCycle(rec_feature[i], real_feature[i]) * self.feature_cycle_ratio
            li += self.criterionCycle(rec[i], real[i]) * (1 - self.feature_cycle_ratio)
            if self.weight_cycle:
                li = li * (1 - self.criterionGAN(pred_fake[i], True)).clamp(min = 0.1, max = 1).detach()
            l += li
        return l / k * lamda

    def optimize_parameters(self, epoch):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()


    def get_current_errors(self):
        D_A = self.loss_D_A.data[0]
        G_A = self.loss_G_A.data[0]
        Rec_A = self.loss_rec_A.data[0]
        D_B = self.loss_D_B.data[0]
        G_B = self.loss_G_B.data[0]
        Rec_B = self.loss_rec_B.data[0]
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data[0]
            idt_B = self.loss_idt_B.data[0]
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Rec_A', Rec_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Rec_B', Rec_A), ('idt_B', idt_B)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Rec_A', Rec_A),
                                ('D_B', D_B), ('G_B', G_B), ('Rec_B', Rec_B)])

    def get_current_visuals(self):
        # hardcoded assumption of 3 channel rgb images
        real_A = util.tensor2im(self.real_A[:, :3].data)
        fake_B = util.tensor2im(self.fake_B[:, :3].data)
        rec_A  = util.tensor2im(self.rec_A[:, :3].data)
        real_B = util.tensor2im(self.real_B[:, :3].data)
        fake_A = util.tensor2im(self.fake_A[:, :3].data)
        rec_B  = util.tensor2im(self.rec_B[:, :3].data)
        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_parameters(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
            for param_group in self.optimizer_D_A.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_D_B.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr

            print('update learning rate: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr

        start, end = self.opt.non_gan_decay_range
        to = self.opt.non_gan_decay_to
        if start <= epoch < end and self.non_gan_weight > to:
            old_non_gan_weight = self.non_gan_weight
            self.non_gan_weight -= (1 - to) / (end - start)
            self.non_gan_weight = max(0, to, self.non_gan_weight)
            print('update non_gan_weight: %f -> %f' % (old_non_gan_weight, self.non_gan_weight))

        start, end = self.opt.feature_shift_range
        to = self.opt.feature_shift_to
        if start <= epoch < end and self.feature_cycle_ratio < to:
            old_feature_cycle_ratio = self.feature_cycle_ratio
            self.feature_cycle_ratio += to / (end - start)
            self.feature_cycle_ratio = min(1, to, self.feature_cycle_ratio)
            print('update feature_cycle_ratio: %f -> %f' % (old_feature_cycle_ratio, self.feature_cycle_ratio))

        if epoch == self.opt.weight_cycle_niter:
            self.weight_cycle = False
