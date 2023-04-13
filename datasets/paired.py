import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class PairedPointCloudDataset(Dataset):

    def __init__(self, root, subset, cat_low, cat_high, transform=None):
        super().__init__()
        self.dir = os.path.join(root, subset)
        self.dir_cat_low = os.path.join(self.dir, cat_low)
        self.dir_cat_high = os.path.join(self.dir, cat_high)
        self.transform = transform
        self.pointclouds_cat_low = []
        self.pointclouds_cat_high = []
        self.pointcloud_names = []
        for fn in tqdm(os.listdir(self.dir_cat_low), desc='Loading'):
            if fn[-3:] != 'xyz':
                continue
            pc1_path = os.path.join(self.dir_cat_low, fn)
            pc2_path = os.path.join(self.dir_cat_high, fn)
            if not os.path.exists(pc2_path):
                raise FileNotFoundError('File not found: %s' % pc2_path)
            pc1 = torch.FloatTensor(np.loadtxt(pc1_path, dtype=np.float32))
            pc2 = torch.FloatTensor(np.loadtxt(pc2_path, dtype=np.float32))
            self.pointclouds_cat_low.append(pc1)
            self.pointclouds_cat_high.append(pc2)
            self.pointcloud_names.append(fn[:-4])

    def __len__(self):
        assert len(self.pointclouds_cat_low) == len(self.pointclouds_cat_high)
        return len(self.pointclouds_cat_low)

    def __getitem__(self, idx):
        pair = {
            'pcl_low': self.pointclouds_cat_low[idx].clone(), 
            'pcl_high': self.pointclouds_cat_high[idx].clone(),
            'name': self.pointcloud_names[idx]
        }
        if self.transform is not None:
            pair = self.transform(pair)
        return pair

