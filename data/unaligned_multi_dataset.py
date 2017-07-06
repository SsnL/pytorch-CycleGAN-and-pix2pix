import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import string


class UnalignedMultiDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.num_datasets = opt.num_datasets

        self.dirs = []
        self.pathss = []
        self.sizes = []

        for i, label in zip(range(self.num_datasets), string.ascii_uppercase):
            dir = os.path.join(opt.dataroot, opt.phase + label)
            paths = make_dataset(dir)

            self.dirs.append(dir)
            self.pathss.append(sorted(paths))
            self.sizes.append(len(paths))

        self.transform = self._get_transformation(opt)

    def __getitem__(self, index):
        rv = {}

        for i, label in zip(range(self.num_datasets), string.ascii_uppercase):
            path = self.pathss[i][index % self.sizes[i]]
            img = Image.open(path).convert('RGB')
            rv[label] = self.transform(img)
            rv[label + '_paths'] = path

        return rv

    def __len__(self):
        return max(self.sizes)

    def name(self):
        return 'UnalignedMultiDataset'
