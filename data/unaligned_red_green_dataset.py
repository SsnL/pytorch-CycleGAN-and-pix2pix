import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class UnalignedRedGreenDataset(BaseDataset):
    def initialize(self, opt, dataset_size = 1000):
        self.opt = opt
        self.dataset_size = dataset_size
        self.size = opt.fineSize
        self.transform = get_transform(opt)

    def __getitem__(self, index):

        A_arr = np.uint8(np.tile(np.array([np.random.rand() * 0.5 + 0.5, 0, np.random.rand() * 0.6]), (self.size, self.size, 1)) * 255)
        B_arr = np.uint8(np.tile(np.array([0, np.random.rand() * 0.5 + 0.5, np.random.rand() * 0.6]), (self.size, self.size, 1)) * 255)

        A_img = Image.fromarray(A_arr, 'RGB')
        B_img = Image.fromarray(B_arr, 'RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img,
                'A_paths': 'gen', 'B_paths': 'gen'}

    def __len__(self):
        return self.size

    def name(self):
        return 'UnalignedRedGreenDataset'
