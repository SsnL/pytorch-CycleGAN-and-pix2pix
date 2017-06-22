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
import string

class MultiCycleGANWithHubModel(BaseModel):
    def name(self):
        return 'MultiCycleGANWithHubModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = self.nb = opt.batchSize
        size = opt.fineSize
        num_datasets = self.num_datasets = opt.num_datasets

        if opt.hub.isdigit():
            raise NotImplemented
        else:
            hub = self.hub = opt.hub

        if opt.identity > 0:
            raise NotImplemented

        assert len(opt.ncs) == len(opt.lambdas) == num_datasets

        self.non_hub_multiplier = opt.non_hub_multiplier

        self.lambdas = {}
        self.inputs = {}

        for label, lamda, nc in zip(string.ascii_uppercase, opt.lambdas, opt.ncs):
            self.lambdas[label] = lamda
            self.inputs[label] = self.Tensor(nb, nc, size, size)

        # load/define networks
        # for each (non-hub) dataset D:
        #     Encoder: D => hub
        #     Decoder: hub => D
        self.encoders = {}
        self.decorders = {}

        hub_nc = opt.ncs[string.ascii_uppercase.find(hub)]

        for label, nc in zip(self.inputs, opt.ncs):
            if label == hub:
                continue
            self.encoders[label] = networks.define_G(nc, hub_nc,
                                     opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)
            self.decorders[label] = networks.define_G(hub_nc, nc,
                                     opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.Ds = {}
            for label, nc in zip(self.inputs, opt.ncs):
                self.Ds[label] = networks.define_D(nc, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            for label in self.inputs:
                self.load_network(self.encoders[label], 'enc_' + label, which_epoch)
                self.load_network(self.decorders[label], 'dec_' + label, which_epoch)
            if self.isTrain:
                for label in self.inputs:
                    self.load_network(self.Ds[label], 'D_' + label, which_epoch)

        if self.isTrain:
            self.fake_pools = {}
            for label in self.inputs:
                self.fake_pools[label] = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers
            G_params = list(e.parameters() for e in self.encoders.values()) + list(d.parameters() for d in self.decorders.values())
            self.optimizer_G = torch.optim.Adam(itertools.chain(*G_params),
                                                lr=self.current_lr, betas=(opt.beta1, 0.999))
            self.optimizers_D = {}
            for label in self.inputs:
                self.optimizers_D[label] = torch.optim.Adam(self.Ds[label].parameters(),
                                                lr=self.current_lr, betas=(opt.beta1, 0.999))

            # skip printing for now?
            print('---------- Networks initialized -------------')
            # networks.print_network(self.netG_A)
            # networks.print_network(self.netG_B)
            # networks.print_network(self.netD_A)
            # networks.print_network(self.netD_B)
            # print('-----------------------------------------------')

    def set_input(self, input):
        for label in self.inputs:
            values = input[label]
            self.inputs[label].resize_(values.size()).copy_(values)
        self.image_paths = input['A_paths'] # hacks

    def forward(self, volatile = False):
        self.reals = OrderedDict()
        self.fakes = OrderedDict()
        self.recs = OrderedDict()
        for label in self.inputs:
            real = Variable(self.inputs[label], volatile = volatile)
            self.reals[label] = real
            if label == self.hub:
                for to_label in self.inputs:
                    if to_label == self.hub:
                        continue
                    fake = self.decorders[to_label].forward(real)
                    rec = self.encoders[to_label].forward(fake)
                    self.fakes[(label, to_label)] = fake
                    self.recs[(label, to_label)] = rec
            else:
                fake_hub = self.encoders[label].forward(real)
                for to_label in self.inputs:
                    if to_label == label:
                        continue
                    elif to_label == self.hub:
                        fake = fake_hub
                        rec = self.decorders[label].forward(fake_hub)
                    else:
                        fake = self.decorders[to_label].forward(fake_hub)
                        rec = self.decorders[label].forward(self.encoders[to_label].forward(fake))
                    self.fakes[(label, to_label)] = fake
                    self.recs[(label, to_label)] = rec

    def test(self):
        self.forward(True)

    #get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self, label):
        real = self.reals[label]
        new_fake = torch.cat(tuple(self.fakes[(from_label, label)] for from_label in self.inputs if from_label != label), 0)
        fake = self.fake_pools[label].query(new_fake)
        netD = self.Ds[label]
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
        self.loss_Ds[label] = loss_D

    def backward_G(self, label):
        # lambda_idt = self.opt.identity
        # Identity loss
        # assumes no idt loss for now

        loss_G = 0
        loss_cycle = 0

        real = self.reals[label]
        for to_label in self.inputs:
            if to_label == label:
                continue
            pred_fake = self.Ds[to_label].forward(self.fakes[(label, to_label)])
            loss_G_one = self.criterionGAN(pred_fake, True)
            loss_cycle_one = self.criterionCycle(self.recs[(label, to_label)], real)
            if label == self.hub or to_label == self.hub:
                loss_G += loss_G_one
                loss_cycle += loss_cycle_one
            else:
                loss_G += loss_G_one * self.non_hub_multiplier
                loss_cycle += loss_cycle_one * self.non_hub_multiplier

        loss_G /= self.num_datasets - 1
        loss_cycle *= self.lambdas[label]

        self.loss_cycles[label] = loss_cycle
        self.loss_Gs[label] = loss_G

        total_loss = loss_G + loss_cycle
        total_loss.backward()

    def optimize_parameters(self):
        self.loss_Ds = {}
        self.loss_cycles = {}
        self.loss_Gs = {}
        # forward
        self.forward()
        # Gs
        self.optimizer_G.zero_grad()
        for label in self.inputs:
            self.backward_G(label)
        self.optimizer_G.step()
        # Ds
        for label in self.inputs:
            self.optimizers_D[label].zero_grad()
            self.backward_D(label)
            self.optimizers_D[label].step()


    def get_current_errors(self):
        errors = OrderedDict()

        # D GAN loss
        for label in self.inputs:
            errors['D_' + label] = self.loss_Ds[label].data[0]

        # G GAN loss
        for label in self.inputs:
            errors['G_' + label] = self.loss_Gs[label].data[0]

        # G cycle loss
        for label in self.inputs:
            errors['Cyc_' + label] = self.loss_cycles[label].data[0]

        # G identity loss
        if self.opt.identity > 0.0:
            raise NotImplemented
            for label in self.inputs:
                errors['idt_' + label] = self.loss_idts[label].data[0]

        return errors

    def get_current_visuals(self):
        visuals = OrderedDict()

        for label in self.inputs:
            # real images
            visuals['real_' + label] = util.tensor2im(self.reals[label].data)
            for to_label in self.inputs:
                if label == to_label:
                    continue
                # fake images
                visuals['fake_' + label + to_label] = util.tensor2im(self.fakes[(label, to_label)].data)
                # rec images
                visuals['rec_' + label + to_label] = util.tensor2im(self.recs[(label, to_label)].data)
                # identity rec images
                if self.opt.identity > 0.0:
                    raise NotImplemented
                    for label in self.inputs:
                        visuals['idt_' + label + to_label] = util.tensor2im(self.idt_recs[(label, to_label)].data)

        return visuals

    def save(self, which_epoch):
        for label in self.inputs:
            if label != self.hub:
                self.save_network(self.encoders[label], 'enc_' + label, which_epoch, self.gpu_ids)
                self.save_network(self.decorders[label], 'dec_' + label, which_epoch, self.gpu_ids)
            self.save_network(self.Ds[label], 'D_' + label, which_epoch, self.gpu_ids)

    def set_learning_rate(self, lr):
        for optimizer in self.optimizers_D.values():
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
