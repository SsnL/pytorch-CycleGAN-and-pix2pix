import random
import torch.utils.data
import torchvision.transforms as transforms
from data.base_data_loader import BaseDataLoader
from data.image_folder import ImageFolder
from PIL import Image
# pip install future --upgrade
from builtins import object
import string
from pdb import set_trace as st
import os

class PairedData(object):
    def __init__(self, data_loaders, num_datasets, max_dataset_size, flip):
        self.data_loaders = data_loaders
        self.stop_flags = [False for _ in self.data_loaders]
        self.num_datasets = num_datasets
        self.max_dataset_size = max_dataset_size
        self.flip = flip

    def __iter__(self):
        self.stop_flags = [False for _ in range(self.num_datasets)]
        self.data_loader_iters = list(iter(loader) for loader in self.data_loaders)
        self.iter = 0
        return self

    def __next__(self):
        data = [None for _ in range(self.num_datasets)]
        paths = [None for _ in range(self.num_datasets)]
        for i in range(self.num_datasets):
            try:
                data[i], paths[i] = next(self.data_loader_iters[i])
            except StopIteration:
                if data[i] is None or paths[i] is None:
                    self.stop_flags[i] = True
                    self.data_loader_iters[i] = iter(self.data_loaders[i])
                    data[i], paths[i] = next(self.data_loader_iters[i])

        if all(self.stop_flags) or self.iter > self.max_dataset_size:
            self.stop_flags = [False for _ in range(self.num_datasets)]
            raise StopIteration()
        else:
            self.iter += 1
            if self.flip and random.random() < 0.5:
                idx = [i for i in range(data[0].size(3) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                data = [d.index_select(3, idx) for d in data]
            rv = {}
            for i in range(self.num_datasets):
                label = string.ascii_uppercase[i]
                rv[label] = data[i]
                rv[label + '_paths'] = paths[i]
            return rv

class UnalignedMultiDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transformations = [transforms.Scale(opt.loadSize, interpolation = Image.BICUBIC),
                           transforms.RandomCrop(opt.fineSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        transform = transforms.Compose(transformations)

        # Datasets
        datasets = []
        data_loaders = []

        for i in range(opt.num_datasets):
            dataset = ImageFolder(root=opt.dataroot + '/' + opt.phase + string.ascii_uppercase[i],
                                    transform=transform, return_paths=True)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.opt.batchSize,
                shuffle=not self.opt.serial_batches,
                num_workers=int(self.opt.nThreads))

            datasets.append(dataset)
            data_loaders.append(data_loader)

        self.datasets = datasets
        flip = opt.isTrain and not opt.no_flip
        self.paired_data = PairedData(data_loaders, opt.num_datasets,
                                      self.opt.max_dataset_size, flip)

    def name(self):
        return 'UnalignedMultiDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max([len(ds) for ds in self.datasets]), self.opt.max_dataset_size)
