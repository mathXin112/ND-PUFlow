import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torch_cluster import knn_graph
from kmeans_pytorch import kmeans
import pytorch3d
from models.data_loss import *
from models.net_utils import *

def make_patches_for_pcl_pair(pcl_low, pcl_high, patch_size, num_patches, ratio):
    """
    Args:
        pcl_low:  Low-resolution point cloud, (N, 3).
        pcl_high: High-resolution point cloud, (rN, 3).
        patch_size:   Patch size M.
        num_patches, self.args.ratio:  Number of patches P. 1
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """
    N = pcl_low.size(0)
    seed_idx = torch.randperm(N)[:num_patches]   # (P, )
    seed_pnts = pcl_low[seed_idx].unsqueeze(0)   # (1, P, 3)
    _, _, pat_low = pytorch3d.ops.knn_points(seed_pnts, pcl_low.unsqueeze(0), K=patch_size, return_nn=True)
    pat_low = pat_low[0]    # (P, M, 3)
    _, _, pat_high = pytorch3d.ops.knn_points(seed_pnts, pcl_high.unsqueeze(0), K=patch_size*ratio, return_nn=True)
    pat_high = pat_high[0]
    return pat_low, pat_high


class Make_Patch(Dataset):
    """
    Args:
        pc: (N * 3)
    return:
        pcl_low: (N1 * 3)
    """

    def __init__(self, args, root, subset, cat_low, transform=None):
        super().__init__()
        self.dir = os.path.join(root, subset)
        self.dir_cat_low = os.path.join(self.dir, cat_low)
        self.args = args
        self.transform = transform
        self.patches_low = []
        self.patches_high =[]
        self.patches = []
        for fn in tqdm(os.listdir(self.dir_cat_low), desc='Loading'):
            if fn[-3:] != 'xyz':
                continue
            pc1_path = os.path.join(self.dir_cat_low, fn)
            pc1 = torch.FloatTensor(np.loadtxt(pc1_path, dtype=np.float32))
            N, _ = pc1.shape

           ## fast point sample
            pcl_1 = farthest_point_sampling(pc1.unsqueeze(0), int(N / 4)).squeeze(0)
            pcl_1 = pcl_1.squeeze(0)
            self.name = fn[:-4]
            pat1, pat2 = make_patches_for_pcl_pair(pcl_1, pc1, self.args.patch_size, self.args.num_patches, self.args.upsample_rate)

            for i in range(pat1.size(0)):
                self.patches.append((pat1[i], pat2[i]))  # + '_ %s' % i))

    def __len__(self):
        # assert len(self.pointclouds_cat_low) == len(self.pointclouds_cat_high)
        return len(self.patches)

    def __getitem__(self, idx):
        patches = {
            'pcl_low': self.patches[idx][0].clone(),
            'pcl_high': self.patches[idx][1].clone(),
        }
        if self.transform is not None:
            patches = self.transform(patches)
        return patches

class Make_Patch_Supervised(Dataset):
    """
    Args:
        pc: (N * 3)
    return:
        pcl_low: (N1 * 3)
    """

    def __init__(self, args, root, subset, cat_low, cat_high, transform=None):
        super().__init__()
        self.dir = os.path.join(root, subset)
        self.dir_cat_low = os.path.join(self.dir, cat_low)
        self.dir_cat_high = os.path.join(self.dir, cat_high)
        self.args = args
        self.transform = transform
        self.patches = []
        self.patches_low = []
        self.patches_middle = []
        self.patches_high = []
        for fn in tqdm(os.listdir(self.dir_cat_low), desc='Loading'):
            if fn[-3:] != 'xyz':
                continue
            pc1_path = os.path.join(self.dir_cat_low, fn)
            pc1 = torch.FloatTensor(np.loadtxt(pc1_path, dtype=np.float32))
            pc2_path = os.path.join(self.dir_cat_high, fn)
            pc2 = torch.FloatTensor(np.loadtxt(pc2_path, dtype=np.float32))
            N, _ = pc1.shape

            ## random select
            self.name = fn[:-4]
            pat1, pat2 = make_patches_for_pcl_pair(pc1, pc2, self.args.patch_size, self.args.num_patches, args.upsample_rate)

            for i in range(pat1.size(0)):
                self.patches.append((pat1[i], pat2[i]))


    def __len__(self):
        # assert len(self.pointclouds_cat_low) == len(self.pointclouds_cat_high)
        return len(self.patches)

    def __getitem__(self, idx):
        patches = {
            'pcl_low': self.patches[idx][0].clone(),
            'pcl_high': self.patches[idx][1].clone(),
        }
        if self.transform is not None:
            patches = self.transform(patches)
        return patches