"""
Point cloud dataset
"""
import os
import os.path

import numpy as np
import pandas
import torch
import torch.utils.data as data


class point_cloud(data.Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform

        self.data = pandas.read_csv(self.path).as_matrix()
        self.min_coords = np.min(self.data[:,0]), np.min(self.data[:,1])
        self.max_coords = np.max(self.data[:, 0]), np.max(self.data[:, 1])
        self.num_classes = len(np.unique(self.data[:,2]))

    def __getitem__(self, index):

        x, y, target = self.data[index]

        input = np.array([x,y])
        target = target.astype(np.int64)

        if self.transform is not None:
            input = self.transform(input)
        else:
            input = torch.from_numpy(input).type(torch.FloatTensor)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.data)
