import torch
from torch.utils.data import Dataset
import numpy as np
import os
import config
import random

class H36M_Dataset(Dataset):
    def __init__(self, root_dir, mode="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        if mode != "train" and mode != "test":
            raise ValueError("mode must be train or test")
        self.data = []
        self.used_joint_indexes = np.array([0,1,2,6,7,11,12,13,14,15,16,17,18,24,25,26]).astype(np.int64)
        self.load_data()

    def load_data(self):
        if self.mode == "train":
            folders = [1, 6, 7, 8, 9, 11]
        else:
            folders = [5]
        for i in folders:
            folder = os.path.join(self.root_dir, f'S{i}')
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                content = np.loadtxt(file_path, delimiter=',')
                for j in range(len(content) - config.frame_num):
                    self.data.append(content[j:j+config.frame_num])
        self.data = np.array(self.data)
        self.data = torch.from_numpy(self.data).float()

    def __len__(self):
        return len(self.data) - config.frame_num
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        sample = sample.reshape(-1, 33, 3)
        # drop the root joint first
        sample = sample[:, 1:, :]
        sample = sample[:, self.used_joint_indexes, :]
        return sample
