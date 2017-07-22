import numpy as np
import os
import torch
from torch import nn
from collections import OrderedDict, Sequence

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.display_num = opt.display_num
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if self.isTrain:
            if opt.niter_warmup > 0:
                self.current_lr = opt.initial_lr
            else:
                self.current_lr = opt.lr

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths_at(self, i):
        pass

    def get_image_paths(self):
        image_paths = []
        for i in range(self.display_num):
            image_paths += self.get_image_paths_at(i)
        return image_paths

    def optimize_parameters(self):
        pass

    def get_current_visuals_at(self, i):
        if i < self.input.size(0):
            return self.input[i]
        else:
            return {}

    def get_current_visuals(self):
        if self.display_num <= 0:
            return {}
        elif self.display_num == 1:
            return self.get_current_visuals_at(0)
        else:
            visuals = OrderedDict()
            for i in range(self.display_num):
                for k, v in self.get_current_visuals_at(i).items():
                    if isinstance(k, Sequence) and not isinstance(k, str):
                        label, supl = k
                        visuals['{}_{}'.format(label, i), supl] = v
                    else:
                        visuals['{}_{}'.format(k, i)] = v
            return visuals

    def get_current_errors(self):
        return {}

    @staticmethod
    def z_str(z_single, precision = 2):
        return np.array_str(z_single.data.cpu().float().numpy(), precision = precision, suppress_small = True)

    def activate(self, net, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(net, input, self.gpu_ids)
        else:
            return net(input)

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device_id=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self, epoch):
        if epoch <= self.opt.niter_warmup:
            lrd = (self.opt.lr - self.opt.initial_lr) / self.opt.niter_warmup
            new_lr = self.current_lr + lrd
        elif epoch > self.opt.niter_warmup + self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.current_lr - lrd
        else:
            return
        self.set_learning_rate(new_lr)
        print('update learning rate: %f -> %f' % (self.current_lr, new_lr))
        self.current_lr = new_lr
