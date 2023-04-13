import math
import random
import numbers
import torch
from torchvision.transforms import Compose


class NormalizeUnitSphere(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def normalize(pcl, center=None, scale=None):
        """
        Args:
            pcl:  The point cloud to be normalized, (N, 3)
        """
        if center is None:
            p_max = pcl.max(dim=0, keepdim=True)[0]
            p_min = pcl.min(dim=0, keepdim=True)[0]
            center = (p_max + p_min) / 2    # (1, 3)
        pcl = pcl - center
        if scale is None:
            scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
        pcl = pcl / scale
        return pcl, center, scale

    def __call__(self, data):
        assert 'pcl_low' in data
        data['pcl_low'], center, scale = self.normalize(data['pcl_low'])
        if 'pcl_high' in data:
            data['pcl_high'], _, _ = self.normalize(data['pcl_high'], center=center, scale=scale)
        if 'pcl_ori' in data:
            data['pcl_ori'], _, _ = self.normalize(data['pcl_ori'], center=center, scale=scale)
        if 'pcl_ori_1' in data:
            data['pcl_ori_1'], _, _ = self.normalize(data['pcl_ori_1'], center=center, scale=scale)
        data['center'] = center
        data['scale'] = scale
        return data


class RandomScale(object):

    def __init__(self, scales):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        data['pcl_low'] = data['pcl_low'] * scale
        if 'pcl_high' in data:
            data['pcl_high'] = data['pcl_high'] * scale
        if 'pcl_ori' in data:
            data['pcl_ori'] = data['pcl_ori'] * scale
        if 'pcl_ori_1' in data:
            data['pcl_ori_1'] = data['pcl_ori_1'] * scale
        return data


class RandomRotate(object):

    def __init__(self, degrees=180.0, axis=0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        matrix = torch.tensor(matrix)

        data['pcl_low'] = torch.matmul(data['pcl_low'], matrix)
        if 'pcl_high' in data:
            data['pcl_high'] = torch.matmul(data['pcl_high'], matrix)
        if 'pcl_ori' in data:
            data['pcl_ori'] = torch.matmul(data['pcl_ori'], matrix)
        if 'pcl_ori_1' in data:
            data['pcl_ori_1'] = torch.matmul(data['pcl_ori_1'], matrix)

        return data


class AddInputNoise(object):

    def __init__(self, std, label_noise=False):
        self.std = std
        self.label_noise = label_noise

    def __call__(self, data):
        # d = self.std * 0.5
        # noise_std = random.uniform(0 - d, self.std + d)
        # noise_std = min(self.std, max(0, noise_std))
        noise_std = self.std
        data['pcl_low'] = data['pcl_low'] + torch.randn_like(data['pcl_low']) * noise_std
        if self.label_noise:
            data['pcl_high'] = data['pcl_high'] + torch.randn_like(data['pcl_high']) * noise_std
        if 'pcl_ori' in data:
            data['pcl_ori'] = data['pcl_ori'] + torch.randn_like(data['pcl_ori']) * noise_std
        if 'pcl_ori_1' in data:
            data['pcl_ori_1'] = data['pcl_ori_1'] + torch.randn_like(data['pcl_ori_1']) * noise_std
        return data


def standard_train_transforms(noise_std=0.005, scale_d=0.3, label_noise=False):
    return Compose([
        NormalizeUnitSphere(),
        RandomScale([1.0-scale_d, 1.0+scale_d]),
        RandomRotate(axis=0),
        RandomRotate(axis=1),
        RandomRotate(axis=2),
        # AddInputNoise(std=noise_std, label_noise=label_noise),
    ])


def standard_patch_train_transforms(scale_d=0.3):
    return Compose([
        NormalizeUnitSphere(),
        RandomScale([1.0-scale_d, 1.0+scale_d]),
        RandomRotate(axis=0),
        RandomRotate(axis=1),
        RandomRotate(axis=2),
    ])

def pcl_transform(batch, args):
    transform_rotate0 = RandomRotate(axis=0)
    transform_rotate1 = RandomRotate(axis=1)
    transform_rotate2 = RandomRotate(axis=2)
    transform_scale = RandomScale([1.0-args.aug_scale_d, 1.0+args.aug_scale_d])
    pcl_low = batch['pcl_low'].clone()
    pcl_high = batch['pcl_high'].clone()
    batch, matrix0 = transform_rotate0(batch)
    batch, matrix1 = transform_rotate1(batch)
    batch, matrix2 = transform_rotate2(batch)
    matrix = torch.matmul(matrix0, torch.matmul(matrix1, matrix2))
    pcl_low_ro = torch.cat([pcl_low, batch["pcl_low"]], dim=0)
    pcl_high_ro = torch.cat([pcl_high, batch["pcl_high"]], dim=0)
    batch_s = {
        "pcl_low": pcl_low_ro.clone(),
        "pcl_high": pcl_high_ro.clone(),
        "name": batch["name"],
        "matrix": matrix
    }
    batch_s, scale = transform_scale(batch_s)
    batch_r = {
        "pcl_low": pcl_low_ro,
        "pcl_high": pcl_high_ro,
        "name": batch["name"],
        "matrix": matrix,
        "scale": scale
    }
    return batch_r, batch_s