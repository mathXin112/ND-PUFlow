import torch
from torch import nn
import pytorch3d.ops
import random
import math
import matplotlib.pyplot as plt
from torch.nn import ModuleList
import loss.emd
from utils.patch_upsample import patch_based_upsample
from .feature import *
from .cnf import build_flow as build_cnf_flow
from .acl import build_flow as build_acl_flow
from .data_loss import *
from .net_utils import *
from .coder import *


def get_local_frames(pcl_source, pcl_target, feat, k, scale):
    """
    Args:
        pcl_source: (B, N, 3)
        pcl_target: (B, M, 3)
    Returns:
        (B, N, K, 3)
    """
    _, idx, frames = pytorch3d.ops.knn_points(pcl_source, pcl_target, K=k, return_nn=True)  # frames:(B, N, K, 3)
    return frames, feat

def local_frames_to_pcl(frames, pcl_source, scale):
    """
    Args:
        frames:     (B, N, K, 3)
        pcl_source: (B, N, 3)
    Returns:
        (B, kN, 3)
    """
    B, N, K, d = frames.size()
    frames_n = (frames * scale).reshape(-1, 3)
    frames_denorm = (frames * scale) + pcl_source.unsqueeze(-2)  # (B, N, K, 3)
    pcl_target = frames_denorm.reshape(B, -1, d)
    return pcl_target


def resample_for_frames(pnts, probs, size):
    """
    Args:
        pnts:  Sampled points in the frames, (B, N, k, 3).
        probs: Log probabilities, (B, N, k [, 1])
    Returns:
        (B, N, size, 3)
    """
    B, N, K, _ = pnts.size()
    probs = probs.view(B, N, K)

    idx_top = torch.argsort(probs, dim=-1, descending=True)
    idx_top_p = idx_top[:, :, :size]

    idx_top = idx_top.unsqueeze(-1).expand_as(pnts)  # (B, N, k, 3)
    idx_top = idx_top[:, :, :size, :]  # (B, N, size, 3)

    probs_sel = torch.gather(probs, dim=2, index=idx_top_p)
    pnts_sel = torch.gather(pnts, dim=2, index=idx_top)
    return pnts_sel, probs_sel


def point_dist(value0):
    dist_matrix0 = torch.sqrt(torch.sum(value0 ** 2, dim=1))
    idx_top = torch.argsort(dist_matrix0, dim=-1, descending=False)
    dist_matrix = dist_matrix0[idx_top[1]]
    return dist_matrix


def pair_wise_distance(pcl, target):
    D = pcl.shape
    if len(D) == 2:
        min_dist = 100
        for i in range(D[0]):
            seed = pcl[i].unsqueeze(0)
            _, _, target_0 = pytorch3d.ops.knn_points(seed.unsqueeze(0), target.unsqueeze(0), K=16, return_nn=True)
            value0 = target_0.squeeze(0).squeeze(0) - seed
            dist_matrix = point_dist(value0)
            min_dist = dist_matrix if dist_matrix < min_dist else min_dist
    elif len(D) == 3:
        min_dist = []
        for i in range(D[0]):
            min_dist_1 = 100
            for j in range(D[1]):
                seed = pcl[i][j].unsqueeze(0)
                _, _, target_0 = pytorch3d.ops.knn_points(seed.unsqueeze(0), target[i].unsqueeze(0), K=16,
                                                          return_nn=True)
                value0 = target_0.squeeze(0).squeeze(0) - seed
                dist_matrix = point_dist(value0)
                min_dist_1 = dist_matrix if dist_matrix < min_dist_1 else min_dist_1
            min_dist.append(min_dist_1)
        min_dist = torch.stack(min_dist)
    else:
        print("Input Error!!!")
    return min_dist

class UpsampleNet(nn.Module):

    def __init__(self, args, conv_knns=[8, 32], k=16):
        super().__init__()
        self.args = args
        self.frame_knn = args.frame_knn
        self.frame_scale = args.frame_scale
        self.feature_net = DGCNN(8, 512)
        self.feats = ModuleList()
        self.feat_dim = 0
        self.k = args.k
        self.cho_k = args.cho_k
        self.scale = args.rate_mult * args.upsample_rate
        self.out_channels = 512
        self.edge_sim_conv_chazhi_v22 = edge_sim_conv_chazhi_v22(out_channel=self.out_channels, k=self.k,
                                                                 cho_k=self.cho_k)
        self.mlp = MLP_CONV_1d(self.out_channels, [384, 256, 128, 64, 3], bn=True)
        self.mlp_refine = MLP_CONV_1d(10, [16, 32, 16, 9, 3], bn=True)
        if args.flow == 'cnf':
            self.sr_flow = build_cnf_flow(
                args,
                input_dim=3,
                hidden_dims=args.flow_hidden_dims,
                context_dim=self.out_channels,
                num_blocks=args.flow_num_blocks,
                conditional=True
            )
        elif args.flow == 'acl':
            self.sr_flow = build_acl_flow(
                context_dim=self.out_channels,
            )

    def get_loss(self, pcl_low, pcl_high):
        """
        Args:
            pcl_low:  Low-resolution point clouds, (B, N, 3)
            pcl_high: High-resolution point clouds, (B, N, 3)
        """
        B, N, _ = pcl_low.size()

        # Normalize
        pcl_low, center, scale = normalize_sphere(pcl_low)
        pcl_high = normalize_pcl(pcl_high, center=center, scale=scale)

        feat = self.feature_net(pcl_low.permute(0,2,1)).permute(0,2,1).repeat(1,self.frame_knn,1)  # (B, N, k, F)
        points, _ = get_local_frames(pcl_low, pcl_high, feat, self.frame_knn, self.frame_scale)  # (B, N, k, 3)
        F, D = feat.size(-1), pcl_high.size(-1)
        feat = feat.reshape(-1, F)
        points = points.reshape(-1, D)

        z, delta_logpz = self.sr_flow(points, context=feat, logpx=torch.zeros(points.size(0), 1).to(points))
        frames = (z - points) / self.args.aug_noise
        log_pz = standard_normal_logprob(frames).sum(dim=1, keepdim=True)
        log_px = log_pz - delta_logpz
        loss = - torch.mean(log_px)
        return loss, frames.reshape(B, -1, D)


    def upsample(self, pcl_low, pcl_high, rate, normalize=True, fps=True, rate_mult=2, state='test', it=1):
        B, N, C = pcl_low.size()  ## B:12, N:256, _:3
        R = int(rate * rate_mult)  ## 64

        import time
        t1 = time.time()
        # Normalize
        if normalize:
            pcl_low, pcl_high, center, scale = normalize_sphere_v2(pcl_low, pcl_high)

        # Noise Point
        noise = torch.randn(B, N * R, 3)
        while sum(noise[noise > 4]) != 0 or sum(noise[noise < -3]) != 0:
            s1 = noise[noise > 4].shape
            noise[noise > 4] = torch.randn(s1)
            s2 = noise[noise < -3].shape
            noise[noise < -3] = torch.randn(s2)
        noise=noise.to(pcl_low) * self.args.aug_noise

        pcl_high = pcl_low.repeat(1, R, 1) + noise

        # Feature extraction
        feat = self.feature_net(pcl_low.permute(0,2,1)).permute(0,2,1)

        feat_flow = self.edge_sim_conv_chazhi_v22(pcl_low, pcl_high, feat, R)

        feat_flow_re = feat_flow.reshape(-1, feat.size(-1))  ## 196608, 60
        pcl_high_re = pcl_high.reshape(-1, pcl_low.size(-1))

        points_smp = self.sr_flow(pcl_high_re, context=feat_flow_re, reverse=True)


        # Reshape and resample
        frames = points_smp.reshape(B, N * R, -1)

        # Denormalize
        if normalize:
            pcl_up_flow = denormalize_pcl(frames, center, scale)

        if fps:
            pcl_up_flow = farthest_point_sampling(pcl_up_flow, rate*pcl_low.size(1))

        return pcl_up_flow



