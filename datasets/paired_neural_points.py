import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class PairedPointCloudDatasetNeuralPoints(Dataset):

    def __init__(self, args, root, subset, cat_low, cat_high, transform=None):
        super().__init__()
        self.args = args,
        self.dir = os.path.join(root, subset)
        self.dir_cat_low = os.path.join(self.dir, cat_low)
        self.dir_cat_high = os.path.join(self.dir, cat_high)
        self.transform = transform
        self.pointcloud_names = []
        self.pointclouds_low = torch.FloatTensor(np.fromfile(self.dir_cat_low, dtype=np.float32)).reshape(-1,args.patch_size,6)[:,:,:3]
        self.pointclouds_high = torch.FloatTensor(np.fromfile(self.dir_cat_high, dtype=np.float32)).reshape(-1,args.patch_size*args.upsample_rate,6)[:,:,:3]

    def __len__(self):
        assert len(self.pointclouds_low) == len(self.pointclouds_high)
        return len(self.pointclouds_low)

    def __getitem__(self, idx):
        pair = {
            'pcl_low': self.pointclouds_low[idx,:,:].clone(),
            'pcl_high': self.pointclouds_high[idx,:,:].clone(),
        }
        return pair

class PairedPointCloudDatasetTest(Dataset):

    def __init__(self, args, root, subset, cat_low, cat_high, scale, transform=None):
        super().__init__()
        self.args = args,
        self.dir = os.path.join(root, subset)
        self.dir_cat_low = os.path.join(self.dir, cat_low)
        self.dir_cat_high = os.path.join(self.dir, cat_high)
        self.dir_scale = os.path.join(self.dir, scale)
        self.transform = transform
        self.pointcloud_names = []
        self.pointclouds_low = torch.FloatTensor(np.fromfile(self.dir_cat_low, dtype=np.float32)).reshape(-1,args.res_input,3)[:,:,:3]
        self.pointclouds_high = torch.FloatTensor(np.fromfile(self.dir_cat_high, dtype=np.float32)).reshape(-1,args.res_gts,3)[:,:,:3]
        self.pointclouds_scale = torch.FloatTensor(np.fromfile(self.dir_scale, dtype=np.float32)).reshape(-1,4)


    def __len__(self):
        assert len(self.pointclouds_low) == len(self.pointclouds_high)
        return len(self.pointclouds_low)

    def __getitem__(self, idx):
        pair = {
            'pcl_low': self.pointclouds_low[idx,:,:].clone(),
            'pcl_high': self.pointclouds_high[idx,:,:].clone(),
            'scale': self.pointclouds_scale[idx]
        }
        return pair

