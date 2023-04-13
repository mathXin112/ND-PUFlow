import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torch_cluster import knn_graph
from kmeans_pytorch import kmeans
NUM_CLUSTERS = 5


def density(pc, k=10):
    num = pc.shape[0]
    knn = knn_graph(pc[:,:3], k, loop=False)
    knn_indx, _ = knn.view(2, num, k)
    knn_data = pc[knn_indx, : ]
    max_distance, _ = (knn_data[:, :, :3] - pc[:, None, :3]).norm(dim=-1).max(dim=-1)
    dense = k /(max_distance ** 3)
    inf_mask = torch.isinf(dense)
    max_val = dense[~inf_mask].max()
    dense[inf_mask] = max_val
    return dense

def double_sub_sample(pc1, pc2, p1, p2, n1, n2, allow_residual=True):
    high_perm = torch.randperm(pc1.shape[0])
    low_perm = torch.randperm(pc2.shape[0])

    d1np1 = round(n1 * p1)
    d1np2 = round(n1 * (1 - p1))
    if d1np1 > pc1.shape[0]:
        d1np2 += (d1np1 - pc1.shape[0])
        d1np1 = pc1.shape[0]
    if d1np2 > pc2.shape[0]:
        d1np1 += (d1np2 - pc2.shape[0])
        d1np2 = pc2.shape[0]

    d2np1 = round(n2 * p2)
    d2np2 = round(n2 * (1 - p2))
    if d2np1 > pc1.shape[0]:
        d2np2 += (d2np1 - pc1.shape[0])
        d2np1 = pc1.shape[0]
    if d2np2 > pc2.shape[0]:
        d2np1 += (d2np2 - pc2.shape[0])
        d2np2 = pc2.shape[0]

    d1idx1, d2idx1 = disjoint_select(high_perm, d1np1, d2np1, allow_residue=allow_residual)

    d2np2 += max(0, d2np1 - d2idx1.shape[0])

    d1idx2, d2idx2 = disjoint_select(low_perm, d1np2, d2np2, allow_residue=allow_residual)

    return torch.cat([pc1[d1idx1, :], pc2[d1idx2, :]], dim=0), torch.cat([pc1[d2idx1, :], pc2[d2idx2, :]], dim=0)

def disjoint_select(pc, n1, n2, allow_residue=True):
    idx1 = pc[:n1]
    if allow_residue:
        residual = max(n1 + n2 - pc.shape[0], 0)
    else:
        residual = 0
    idx2 = pc[max(n1 - residual, 0): n1 + n2]
    return idx1, idx2


def SubsetData(pc, weight, args):
    criterion_mask = weight < weight.mean()

    high_pc = pc[criterion_mask, :]
    low_pc = pc[~criterion_mask, :]

    # subset ratios
    p1 = args.p1
    p2 = args.p2

    size1, size2 = int(pc.shape[0]/4), int(pc.shape[0]/4)
    D1, D2 = double_sub_sample(high_pc, low_pc, p1, p2, size1, size2, allow_residual=True)

    return D1[:, :3], D2[:, :3]

def DensityData(pc, args):
    # device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    dense = density(pc)
    pcl_low, pcl_high = SubsetData(pc, torch.log(dense), args)
    return pcl_low, pcl_high


class Pair_Density_Data(Dataset):

    def __init__(self, args, root, subset, cat_low, transform=None):
        super().__init__()
        self.dir = os.path.join(root, subset)
        self.dir_cat_low = os.path.join(self.dir, cat_low)
        # self.dir_cat_high = os.path.join(self.dir, cat_high)
        self.transform = transform
        self.pointclouds_cat_low = []
        self.pointclouds_cat_high = []
        self.pointclouds_cat_origin = []
        self.pointcloud_names = []
        for fn in tqdm(os.listdir(self.dir_cat_low), desc='Loading'):
            if fn[-3:] != 'xyz':
                continue
            pc1_path = os.path.join(self.dir_cat_low, fn)
            # pc2_path = os.path.join(self.dir_cat_high, fn)
            # if not os.path.exists(pc2_path):
            #     raise FileNotFoundError('File not found: %s' % pc2_path)
            pc1 = torch.FloatTensor(np.loadtxt(pc1_path, dtype=np.float32))
            # pc2 = torch.FloatTensor(np.loadtxt(pc2_path, dtype=np.float32))
            pc1_low, pc1_high = DensityData(pc1, args)
            self.pointclouds_cat_low.append(pc1_low)
            self.pointclouds_cat_high.append(pc1_high)
            self.pointclouds_cat_origin.append(pc1)
            self.pointcloud_names.append(fn[:-4])

    def __len__(self):
        assert len(self.pointclouds_cat_low) == len(self.pointclouds_cat_high)
        return len(self.pointclouds_cat_low)

    def __getitem__(self, idx):
        pair = {
            'pcl_low': self.pointclouds_cat_low[idx].clone(),
            'pcl_high': self.pointclouds_cat_high[idx].clone(),
            'pcl_ori': self.pointclouds_cat_origin[idx].clone(),
            'name': self.pointcloud_names[idx]
        }
        if self.transform is not None:
            pair = self.transform(pair)
        return pair
