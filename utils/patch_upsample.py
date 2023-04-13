import math
import os
import time

import torch
import pytorch3d.ops
from tqdm.auto import tqdm

# from datasets.resample import pcl_save
from models.data_loss import *
from models.net_utils import *
from utils.transforms_convar import standard_train_transforms, pcl_transform

def remove_outlier(xyz):
    device = xyz.device
    B, N, C = xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, N).repeat(B, 1)
    dst, _, _ = pytorch3d.ops.knn_points(xyz, xyz, K=5)
    dst_0 = dst.sum(-1)
    group_idx[dst_0 > 0.05] = N
    group_first = group_idx[:, 0].repeat(1,N)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    new_xyz = pytorch3d.ops.knn_gather(xyz, group_idx.reshape(B,N,1)).squeeze(-2)
    return new_xyz[0]

def patch_based_upsample(args, model, pcl, patch_size, seed_k=3, flow_batch_size=12):
    """
    Args:
        pcl:  Input point cloud, (N, 3)
    """
    assert pcl.dim() == 2, 'The shape of input point cloud must be (N, 3).'

    N, d = pcl.size()
    pcl = pcl.unsqueeze(0)  # (1, N, 3)
    seed_pnts = farthest_point_sampling(pcl, max(int(seed_k * N / patch_size), 1))

    dst, _, patches = pytorch3d.ops.knn_points(seed_pnts, pcl, K=patch_size, return_nn=True)
    patches = patches[0]

    new_patches = []
    for i in range(patches.size(0)):
        new_patch = remove_outlier(patches[i].unsqueeze(0))
        new_patches.append(new_patch)
    new_patches = torch.stack(new_patches)
    patches_queue = [new_patches[i * flow_batch_size:(i + 1) * flow_batch_size] for i in range(math.ceil(patches.size(0) / flow_batch_size))]

    patches_up_queue = []

    with torch.no_grad():
        model.eval()    # Important!
        for pat in patches_queue:
            pcl_up = model.upsample(pcl_low=pat, pcl_high=pat, rate=args.upsample_rate, fps=False,
                                                        rate_mult=args.rate_mult, state='test')
            patches_up_queue.append(pcl_up)
        patches_up = torch.cat(patches_up_queue, dim=0)
        assert patches_up.size(0) == patches.size(0)
    patches_up = torch.cat([patches_up.view(1, -1, d), pcl], 1)
    pcl_up = farthest_point_sampling(patches_up, int(args.upsample_rate * N))[0]
    return pcl_up
