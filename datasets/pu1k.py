import torch
import h5py
import numpy as np
from torch.utils.data import Dataset,DataLoader
import datasets.point_operation as point_operation
from tqdm.auto import tqdm
import os


def load_h5_data( h5_filename, num_point, up_ratio, skip_rate, use_randominput):
    # logging.info('==========================    Loading H5data    ==================================')
    num_point = num_point
    num_4X_point = int(num_point * 4)
    num_out_point = int(num_point * up_ratio)
    # logging.info('loading data from: {}'.format(h5_filename))
    if use_randominput:
        # logging.info('use random input')
        with h5py.File(h5_filename,'r') as f:
            input = f['poisson_%d' % num_4X_point][:]
            gt = f['poisson_%d'% num_out_point][:]
    else:
        # logging.info('do not use random input')
        with h5py.File(h5_filename, 'r') as f:
            input = f['poisson_%d' % num_point][:]
            gt = f['poisson_%d' % num_out_point][:]

    len_input = len(input)
    len_gt = len(gt)
    assert len_input == len_gt

    data_radius = np.ones(shape=(len(input)))
    centroid = np.mean(input[:, :, 0:3], axis=1, keepdims=True)
    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
    input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]
    # logging.info("total %d samples" % (len(input)))
    # logging.info("========== Finish Data Loading ========== \n")
    return input, gt, data_radius

class PU1KDataset(Dataset):
    def __init__(self, data_path, num_point, up_ratio, skip_rate=1, use_randominput=False, transform=False):
        super().__init__()
        self.transform = transform
        self.num_point =num_point
        self.up_ratio = up_ratio
        self.use_randominput = use_randominput
        self.input_data, self.gt_data, self.radius_data = load_h5_data(data_path,num_point,up_ratio,skip_rate,use_randominput)
    def __len__(self):
        return len(self.input_data)
    def __getitem__(self, idx):
        input_data = self.input_data[idx]
        gt_data = self.gt_data[idx]
        radius_data = self.radius_data[idx]
        if self.use_randominput:
            idx = point_operation.nonuniform_sampling(input_data.shape[0],sample_num=self.num_point)
            input_data = input_data[idx]
        if self.transform:
            input_data = point_operation.jitter_perturbation_point_cloud(input_data,sigma=0.01,clip=0.03)
            input_data,gt_data = point_operation.rotate_point_cloud_and_gt(input_data,gt_data)
            input_data,gt_data,scales = point_operation.random_scale_point_cloud_and_gt(input_data,gt_data,scale_low=0.8,scale_high=1.2)
            radius_data = radius_data * scales
        # data = {
        #     'pcl_low': input_data.clone(),
        #     'pcl_high': gt_data.clone(),
        #     "radius_data": radius_data,
        # }
        return input_data, gt_data, radius_data

class PU1K_self_Dataset(Dataset):
    def __init__(self, data_path, num_point, up_ratio, skip_rate=1, use_randominput=False, transform=False):
        super().__init__()
        self.transform = transform
        self.num_point =num_point
        self.up_ratio = up_ratio
        self.use_randominput = use_randominput
        self.input_data, self.gt_data, self.radius_data = load_h5_data(data_path,num_point,up_ratio,skip_rate,use_randominput)
    def __len__(self):
        return len(self.input_data)
    def __getitem__(self, idx):
        input_data = self.input_data[idx]
        gt_data = self.gt_data[idx]
        radius_data = self.radius_data[idx]
        if self.use_randominput:
            idx = point_operation.nonuniform_sampling(input_data.shape[0],sample_num=self.num_point)
            input_data = input_data[idx]
        if self.transform:
            input_data = point_operation.jitter_perturbation_point_cloud(input_data,sigma=0.01,clip=0.03)
            input_data,gt_data = point_operation.rotate_point_cloud_and_gt(input_data,gt_data)
            input_data,gt_data,scales = point_operation.random_scale_point_cloud_and_gt(input_data,gt_data,scale_low=0.8,scale_high=1.2)
            radius_data = radius_data * scales
        return input_data,gt_data,radius_data

class PUNET_Dataset_Whole(Dataset):
    def __init__(self, data_dir='./PU1K/test/input_2048'):
        super().__init__()

        file_list = os.listdir(data_dir)
        # self.names = [x.split('.')[-2] for x in file_list]
        self.names = [x[ :-4] for x in file_list]
        self.last_names = [x[-4 : -1] for x in file_list]
        self.sample_path = [os.path.join(data_dir, x) for x in file_list]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        points = np.loadtxt(self.sample_path[index])
        return points

# class PairedPointCloudDataset(Dataset):
#
#     def __init__(self, input_dir, gt_dir, transform=None):
#         super().__init__()
#         self.dir_input = os.path.join(input_dir)
#         self.dir_gt = os.path.join(gt_dir)
#         self.transform = transform
#         self.pointclouds_input = []
#         self.pointclouds_gt = []
#         self.pointcloud_names = []
#         for fn in tqdm(os.listdir(self.dir_input), desc='Loading'):
#             if fn[-3:] != 'xyz':
#                 continue
#             pc1_path = os.path.join(self.dir_input, fn)
#             pc2_path = os.path.join(self.dir_gt, fn)
#             if not os.path.exists(pc2_path):
#                 raise FileNotFoundError('File not found: %s' % pc2_path)
#             # print('pc1', pc1_path)
#             pc1 = torch.FloatTensor(np.loadtxt(pc1_path, dtype=np.float32))
#             # print('pc2', pc2_path)
#             pc2 = torch.FloatTensor(np.loadtxt(pc2_path, dtype=np.float32))
#             self.pointclouds_input.append(pc1)
#             self.pointclouds_gt.append(pc2)
#             self.pointcloud_names.append(fn[:-4])
#
#     def __len__(self):
#         assert len(self.pointclouds_input) == len(self.pointclouds_gt)
#         return len(self.pointclouds_input)
#
#     def __getitem__(self, idx):
#         pair = {
#             'pcl_input': self.pointclouds_input[idx].clone(),
#             'pcl_gt': self.pointclouds_gt[idx].clone(),
#             'name': self.pointcloud_names[idx]
#         }
#         if self.transform is not None:
#             pair = self.transform(pair)
#         return pair

if __name__ =='__main__':
    mm = PU1KDataset('/media/huxin/MyDisk/upsample_history/upsample_edit8/data/PU1K/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5',256,4,1,True)
    ss = DataLoader(mm, batch_size=2, num_workers=0, shuffle=True)
    for _,data in enumerate(ss):
        input_data = data[0]
        gt_data = data[1]
        radius = data[2]
        m=0