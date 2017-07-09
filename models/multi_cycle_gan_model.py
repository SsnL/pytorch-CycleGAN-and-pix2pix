import numpy as np
import torch
import os
from collections import OrderedDict, defaultdict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
import string

class MultiCycleGANModel(BaseModel):
    def name(self):
        return 'MultiCycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = self.nb = opt.batchSize
        size = opt.fineSize
        num_datasets = self.num_datasets = opt.num_datasets

        if opt.identity > 0:
            raise NotImplemented

        self.display_single_pane_ncols = opt.display_single_pane_ncols
        # Some cycles can be sampled with replacement. If displaying in single
        # image pane, we need filler image so that results from certain dataset
        # are placed in beginning of new rows.
        if self.display_single_pane_ncols > 0:
            self.visual_filler = np.ones((size, size, 3), dtype = np.uint8) * 255

        assert len(opt.ncs) == num_datasets

        self.inputs = OrderedDict()

        for label, nc in zip(string.ascii_uppercase, opt.ncs):
            self.inputs[label] = self.Tensor(nb, nc, size, size)

        if self.isTrain:
            self.lambdas = OrderedDict()

            for label, lamda in zip(string.ascii_uppercase, opt.lambdas):
                self.lambdas[label] = lamda

        # preprocess with cycles
        assert len(opt.cycle_lengths) == len(opt.cycle_weights) == len(opt.cycle_num_samples)

        cl, cw, cn = zip(*list(sorted(zip(opt.cycle_lengths, opt.cycle_weights, opt.cycle_num_samples))))

        self.sample_cycle_lengths = []
        self.sample_cycle_weights = []
        self.sample_cycle_num = []
        self.exact_cycles = {label: [] for label in self.inputs}

        all_mids = {l: {l: [[]]} for l in self.inputs}
        last_exact_l = 1

        # exact cycles
        for l, w, n in zip(cl, cw, cn):
            if l < 2:
                raise ValueError("Cycle length must be at least 2.")
            if l == last_exact_l:
                raise ValueError("Cycle lengths must be distinct.")
            if n > 0:
                self.sample_cycle_lengths.append(l)
                self.sample_cycle_weights.append(w)
                self.sample_cycle_num.append(n)
                continue
            total = (num_datasets - 1) * (num_datasets - 2) ** (l - 2)
            for label in self.inputs:
                mids = all_mids[label]
                for _ in range(last_exact_l - 1, l - 1):
                    next_mids = defaultdict(list)
                    for last_label, so_fars in mids.items():
                        for next_label in self.inputs:
                            if next_label == last_label or next_label == label:
                                continue
                            next_mids[next_label] += [so_far + [next_label] for so_far in so_fars]
                    mids = next_mids
                self.exact_cycles[label] += [((label,) + tuple(mid) + (label,), w / total) for mid in itertools.chain(*mids.values())]
                all_mids[label] = mids
            last_exact_l = l

        print('----------- Exact cycles --------------')
        for cycles in self.exact_cycles.values():
            for cycle, weight in cycles:
                print('%s\t%.4f' % ('->'.join(cycle), weight))
        print('---------------------------------------')

        # load/define networks
        self.Gs = {}

        for (label, nc), (to_label, to_nc) in itertools.permutations(zip(self.inputs, opt.ncs), 2):
            self.Gs[(label, to_label)] = networks.define_G(nc, to_nc,
                                     opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, gpu_ids = self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.Ds = {}
            for label, nc in zip(self.inputs, opt.ncs):
                self.Ds[label] = networks.define_D(nc, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            for label, to_label in self.Gs:
                self.load_network(self.Gs[(label, to_label)], 'G_' + label + to_label, which_epoch)
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
            G_params = list(netG.parameters() for netG in self.Gs.values())
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

    # returns a dict of sampled {end_label: {cycle path tuple: weights, ...}, ...}
    def sample_cycles(self):
        samples = {label: defaultdict(float) for label in self.inputs}
        for l, w, n in zip(self.sample_cycle_lengths, self.sample_cycle_weights, self.sample_cycle_num):
            for label in self.inputs:
                for _ in range(n):
                    cycle = [label]
                    last_label = label
                    for __ in range(l - 1):
                        avails = list(o for o in self.inputs if o != label and o != last_label)
                        last_label = np.random.choice(avails)
                        cycle.append(last_label)
                    samples[label][tuple(cycle) + (label,)] += w / n
        return {label: list(samples[label].items()) for label in samples}

    def set_input(self, input):
        for label in self.inputs:
            values = input[label]
            self.inputs[label].resize_(values.size()).copy_(values)
        self.image_paths = {l: input['{}_paths'.format(l)] for l in self.inputs}
        # sample cycles
        self.sampled_cycles = self.sample_cycles()

    #get image paths
    def get_image_paths_at(self, i):
        image_paths = []
        no_rec_replicate = 1 + (2 if self.opt.identity > 0 else 1) * (self.num_datasets - 1)
        for label in self.inputs:
            if i < self.inputs[label].size(0):
                num_cycles = len(self.exact_cycles[label] + self.sampled_cycles[label])
                image_paths += [self.image_paths[label][i]] * (no_rec_replicate + num_cycles)
        return image_paths

    # return the rec images
    def forward_path(self, images, path, current_idx = 0):
        current = path[current_idx]
        target_idx = current_idx + 1
        while target_idx < len(path):
            target = path[target_idx]
            images = self.Gs[(current, target)].forward(images)
            current = target
            target_idx += 1
        return images

    def forward(self, volatile = False):
        self.reals = OrderedDict()
        self.fakes = OrderedDict()
        self.recs = OrderedDict()
        for label in self.inputs:
            real = Variable(self.inputs[label], volatile = volatile)
            self.reals[label] = real
            for to_label in self.inputs:
                if label == to_label:
                    continue
                self.fakes[(label, to_label)] = self.Gs[(label, to_label)].forward(real)
        for label in self.inputs:
            for cycle, weight in itertools.chain(self.exact_cycles[label], self.sampled_cycles[label]):
                rec = self.forward_path(self.fakes[(cycle[:2])], cycle, 1)
                self.recs[cycle] = rec

    def test(self):
        self.forward(True)

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
            loss_G += self.criterionGAN(pred_fake, True)

        for cycle, weight in itertools.chain(self.exact_cycles[label], self.sampled_cycles[label]):
            loss_cycle += self.criterionCycle(self.recs[cycle], real) * weight

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
        ncols = self.display_single_pane_ncols
        for label in self.inputs:
            nimgs = 0
            # real images
            visuals['real_' + label] = util.tensor2im(self.reals[label].data)
            nimgs += 1
            for to_label in self.inputs:
                if label == to_label:
                    continue
                # fake images
                visuals['fake_' + label + to_label] = util.tensor2im(self.fakes[(label, to_label)].data)
                nimgs += 1
                # identity rec images
                if self.opt.identity > 0.0:
                    raise NotImplemented
                    for label in self.inputs:
                        visuals['idt_' + label + to_label] = util.tensor2im(self.idt_recs[(label, to_label)].data)
                        nimgs += 1
            # rec images
            for cycle, weight in self.exact_cycles[label] + self.sampled_cycles[label]:
                visuals['rec_' + ''.join(cycle)] = util.tensor2im(self.recs[cycle].data)
                nimgs += 1
            # if necessary, add fillers till new row
            # when testing, outputing to webpage, so skip
            while self.isTrain and ncols > 0 and nimgs % ncols != 0:
                visuals['filler_' + label + '_' + str(nimgs)] = self.visual_filler
                nimgs += 1
        return visuals

    def save(self, which_epoch):
        for label, to_label in self.Gs:
            self.save_network(self.Gs[(label, to_label)], 'G_' + label + to_label, which_epoch, self.gpu_ids)
        for label in self.inputs:
            self.save_network(self.Ds[label], 'D_' + label, which_epoch, self.gpu_ids)

    def set_learning_rate(self, lr):
        for optimizer in self.optimizers_D.values():
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
