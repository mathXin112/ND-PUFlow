import torch
from torch.nn import Module, Linear, ModuleList, Conv2d, Conv1d
import pytorch3d.ops
import random
from .net_utils import *
from .data_loss import *


def get_knn_idx(x, y, k, offset=0):
    """
    Args:
        x: (B, N, d)
        y: (B, M, d)
    Returns:
        (B, N, k)
    """
    _, knn_idx, _ = pytorch3d.ops.knn_points(x, y, K=k + offset)
    return knn_idx[:, :, offset:]


def knn_group(x: torch.FloatTensor, idx: torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)

    return torch.gather(x, dim=2, index=idx)


class Aggregator(torch.nn.Module):

    def __init__(self, oper):
        super().__init__()
        assert oper in ('mean', 'sum', 'max')
        self.oper = oper

    def forward(self, x, dim=2):
        if self.oper == 'mean':
            return x.mean(dim=dim, keepdim=False)
        elif self.oper == 'sum':
            return x.sum(dim=dim, keepdim=False)
        elif self.oper == 'max':
            ret, _ = x.max(dim=dim, keepdim=False)
            return ret



def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(xyz, new_xyz, radius, nsample):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def dil_knn(x, y, k=16, d=1, use_fsd=False):
    if len(x.shape) > 3 or len(y.shape) > 3:
        x = x.squeeze(2)
        y = y.squeeze(2)
    dst, idx, _ = pytorch3d.ops.knn_points(x, y, K=k * d, return_nn=True)  # [B N K 2]
    if d > 1:
        if use_fsd:
            idx = idx[:, :, k * (d - 1):k * d]
        else:
            idx = idx[:, :, ::d]
    return dst, idx


def get_graph_features(x, idx, return_central=True):
    """
    get the features for the neighbors and center points from the x and inx
    :param x: input features
    :param idx: the index for the neighbors and center points
    :return:
    """
    if len(x.shape) > 3:
        x = x.squeeze(2)
    pc_neighbors = pytorch3d.ops.knn_gather(x, idx)
    if return_central:
        pc_central = x.unsqueeze(-2).repeat(1, 1, idx.shape[2], 1)
        return pc_central, pc_neighbors
    else:
        return pc_neighbors


class edge_conv(Module):
    def __init__(self, out_channel=64, scale=2, k=16, d=1, n_layer=1):
        super(edge_conv, self).__init__()
        self.out_channel = out_channel
        self.k = k
        self.d = d
        self.n_layer = n_layer
        self.conv1 = Conv2d(out_channel, out_channel * scale, 1)
        self.conv2 = Conv2d(out_channel, out_channel * scale, 1)
        for i in range(n_layer - 1):
            self.add_module('d' + str(i), Conv2d(out_channel * scale, out_channel * scale, kernel_size=1))

    def forward(self, x, idx=None):
        if idx is None:
            idx = dil_knn(x, self.k, self.d)
        central, neighbors = get_graph_features(x, idx)
        message = self.conv1((neighbors - central).permute(0, 3, 1, 2))
        x_center = self.conv2(x.permute(0, 3, 1, 2))
        edge_features = x_center + message
        edge_features = torch.relu(edge_features)
        for i in range(self.n_layer - 1):
            edge_features = self._modules['d' + str(i)](edge_features)
        y = edge_features.max(-1)[0]
        y = y.transpose(1, 2).unsqueeze(-2)
        return y


def point_shuffler(input, scale=2):
    B, N, M, C = input.shape
    outputs = input.reshape(B, N, 1, C // scale, scale)
    outputs = outputs.permute(0, 1, 4, 3, 2)
    outputs = outputs.reshape(B, N * scale, 1, C // scale)
    return outputs

class densgcn(Module):
    def __init__(self, n_layers=3, d=1, out_channel=128, return_idx=True):
        super(densgcn, self).__init__()
        self.n_layers = n_layers
        self.d = d
        self.return_idx = return_idx
        self.conv1 = Conv2d(out_channel, out_channel, 1)
        self.conv2 = Conv2d(out_channel, out_channel, 1)
        for i in range(n_layers - 1):
            self.add_module('d' + str(i), Conv2d(out_channel * 2, out_channel, kernel_size=1))

    def forward(self, f, k, idx=None):
        if idx is None:
            _, idx = dil_knn(f, f, k, self.d)
        central, neighbors = get_graph_features(f, idx, return_central=True)
        message = self.conv1((neighbors - central).permute(0, 3, 1, 2))
        f_center = self.conv2(central.permute(0, 3, 1, 2))
        y = f_center + message
        y = torch.relu(y)

        features = y
        for i in range(self.n_layers - 1):
            feature = torch.cat([features, y], dim=1)
            features = self._modules['d' + str(i)](feature)

        if self.return_idx:
            return features, idx
        else:
            return features


class edge_sim_conv_chazhi_v22(Module):
    def __init__(self, out_channel=64, scale=2, k=4, cho_k=4, d=1, n_layer=2):
        super(edge_sim_conv_chazhi_v22, self).__init__()
        self.out_channel = out_channel
        self.k = k
        self.d = d
        self.n_layer = n_layer
        self.cho_k = cho_k
        self.conv1 = Conv1d(2 * 3 * (k), out_channel, 1)
        self.conv2 = Conv1d(out_channel, out_channel, 1)
        self.conv3 = Conv1d(out_channel, out_channel, 1)
        self.conv4 = Conv1d(2 * out_channel, out_channel, 1)
        self.conv5 = Conv1d(2 * out_channel, out_channel, 1)
        self.conv6 = Conv2d(k, 1, 1)
        self.conv7 = Conv1d(out_channel + 6, out_channel, 1)
        self.bn = nn.BatchNorm1d(out_channel)
        self.mlp_conv = MLP_CONV(3, [128, 256, 384, out_channel], bn=True)
        self.mlp_conv2 = MLP_CONV_1d(3 * out_channel, [1280, 1024, 768, out_channel], bn=True)
        self.mlp_conv3 = MLP_CONV(3, [12, 24, 48, out_channel], bn=True)
        self.mlp_conv4 = MLP_CONV_1d(2 * out_channel, [896, 768, 640, out_channel], bn=True)
        self.mlp_conv5 = MLP_CONV_1d(out_channel, [640, 768, 640, out_channel], bn=True)
        self.mlp_conv6 = MLP_CONV_1d(out_channel + 3, [640, 768, 640, out_channel], bn=True)  # 2022_07_17__21_36_09 无
        self.mlp_conv7 = MLP_CONV(6, [128, 256, 384, out_channel], bn=True)
        self.gcn = densgcn(out_channel=out_channel, n_layers=3, return_idx=False)
        for i in range(n_layer - 1):
            self.add_module('d' + str(i), Conv2d(out_channel * scale, out_channel * scale, kernel_size=1))

    def forward(self, pcl, pcl_noise, feature, R=1, idx=None):
        B, N, C = pcl_noise.shape
        _, M, _ = pcl.shape
        R = int(N / M)
        if idx is None:
            _, close_idx, close_point = pytorch3d.ops.knn_points(pcl_noise, pcl, K=1, return_nn=True)
            _, idx = dil_knn(close_point.squeeze(-2), pcl, self.k, self.d)
        knn_points = pytorch3d.ops.knn_gather(pcl, idx)[:, :, 1:, :]
        knn_feat = pytorch3d.ops.knn_gather(feature, idx)[:, :, 1:, :]
        close_feat = pytorch3d.ops.knn_gather(feature, close_idx).squeeze(-2)
        delta_knn_points = knn_points - pcl_noise.unsqueeze(-2).repeat(1, 1, self.k - 1, 1)
        dst = torch.norm(delta_knn_points, p=2, dim=-1)
        weight = torch.exp(- 10 * dst) / (torch.exp(- 10 * dst).sum(dim=-1, keepdim=True) + 1e-7)
        delta_points = torch.cat([delta_knn_points, close_point.repeat(1, 1, self.k - 1, 1)], dim=-1)
        point_feature = self.mlp_conv7(delta_points.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        point_feature = torch.matmul(weight.unsqueeze(-1).permute(0, 1, 3, 2), point_feature).squeeze(-2)
        co_feature = torch.matmul(weight.unsqueeze(-1).permute(0, 1, 3, 2), knn_feat).squeeze(-2)
        delta_feature = torch.cat([close_feat, co_feature, point_feature], dim=-1)
        new_feature = self.mlp_conv2(delta_feature.permute(0, 2, 1)).permute(0, 2, 1)
        return new_feature
