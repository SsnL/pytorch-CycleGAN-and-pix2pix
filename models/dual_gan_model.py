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

class DualGANModel(BaseModel):
    def name(self):
        return 'DualGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                        opt.which_model_netG, opt.norm,
                                        opt.use_dropout, gpu_ids = self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf,
                                        opt.which_model_netG, opt.norm,
                                        opt.use_dropout, gpu_ids = self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # real === real_A_fake_B
            self.netD = networks.define_D(opt.output_nc + opt.input_nc, opt.ndf,
                                          opt.which_model_netD, opt.n_layers_D,
                                          opt.norm, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.real_A_fake_B_pool = ImagePool(opt.pool_size)
            self.fake_A_real_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_A)
            networks.print_network(self.netG_B)
            networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)
        # self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        # self.rec_B  = self.netG_A.forward(self.fake_A)

    #get image paths
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
        pred_fake = netD.forward(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        rAfB = self.real_A_fake_B_pool.query(self.rAfB)
        fArB = self.fake_A_real_B_pool.query(self.fArB)
        self.loss_D = self.backward_D_basic(self.netD, rAfB.detach(), fArB.detach())

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # Loss
        # A
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)
        self.rAfB = torch.cat((self.real_A, self.fake_B), 1)
        self.loss_G_A = self.criterionGAN(self.netD.forward(self.rAfB), 0.5)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # B
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B = self.netG_A.forward(self.fake_A)
        self.fArB = torch.cat((self.fake_A, self.real_B), 1)
        self.loss_G_B = self.criterionGAN(self.netD.forward(self.fArB), 0.5)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_A
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()


    def get_current_errors(self):
        D = self.loss_D.data[0]
        G_A = self.loss_G_A.data[0]
        Cyc_A = self.loss_cycle_A.data[0]
        G_B = self.loss_G_B.data[0]
        Cyc_B = self.loss_cycle_B.data[0]
        errors = OrderedDict()
        errors['D'] = D
        errors['G_A'] = G_A
        errors['Cyc_A'] = Cyc_A
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data[0]
            errors['idt_A'] = idt_A
        errors['G_B'] = G_B
        errors['Cyc_B'] = Cyc_B
        if self.opt.identity > 0.0:
            idt_B = self.loss_idt_B.data[0]
            errors['idt_B'] = idt_B
        return errors

    def get_current_visuals_at(self, i):
        visuals = OrderedDict()
        if i < self.real_A.size()[0]:
            visuals['real_A'] = util.tensor2im(self.real_A.data, i)
            visuals['fake_B'] = util.tensor2im(self.fake_B.data, i)
            visuals['rec_A'] = util.tensor2im(self.rec_A.data, i)
            if self.opt.identity > 0.0:
                visuals['idt_A'] = util.tensor2im(self.idt_A.data, i)
        if i < self.real_B.size()[0]:
            visuals['fake_A'] = util.tensor2im(self.fake_A.data, i)
            visuals['real_B'] = util.tensor2im(self.real_B.data, i)
            visuals['rec_B'] = util.tensor2im(self.rec_B.data, i)
            if self.opt.identity > 0.0:
                visuals['idt_B'] = util.tensor2im(self.idt_B.data, i)
        return visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def set_learning_rate(self, lr):
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
