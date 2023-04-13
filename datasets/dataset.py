import torch
import h5py
import numpy as np
from torch.utils.data import Dataset,DataLoader
from datasets.point_operation import *
from tqdm.auto import tqdm
import os
import torch.utils.data as torch_data
# from utils import utils
import os

def load_h5_data(h5_filename,num_point,up_ratio,skip_rate=1,use_randominput=False):
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
    assert len(input) == len(gt)
    # print("load_h5_input", input.shape)  ##(6900, 1024, 3)
    # print("load_h5_gt", gt.shape) ##(6900,1024,3)
    # logging.info("Normalize the data")
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
    # def __init__(self, data_path,num_point,up_ratio,skip_rate=1,use_randominput=False,transform=False):
    def __init__(self, data_path, num_point, up_ratio, skip_rate=1, use_randominput=False): ##, transform=False): ##, type='train'):
        super().__init__()
        # self.transform = transform
        self.num_point =num_point
        self.up_ratio = up_ratio
        # self.use_randominput = use_randominput
        # self.type = type ##胡鑫添加
        # self.input_data, self.gt_data = load_h5_data(data_path,num_point,up_ratio,skip_rate, use_randominput)
        self.input_data, self.gt_data, self.radius_data = load_h5_data(data_path, num_point, up_ratio, skip_rate, use_randominput)
    def __len__(self):
        return len(self.input_data)
    def __getitem__(self, idx):
        input_data = self.input_data[idx]
        # print("PU1K", input_data.shape)
        gt_data = self.gt_data[idx]
        # radius_data = self.radius_data[idx]
        # if self.use_randominput:
        #     idx = point_operation.nonuniform_sampling(input_data.shape[0],sample_num=self.num_point)
        #     input_data = input_data[idx]
        # if self.transform:
        #     input_data = point_operation.jitter_perturbation_point_cloud(input_data,
        #                                                                        sigma=0.01,
        #                                                                        clip=0.03)
        #     input_data,gt_data = point_operation.rotate_point_cloud_and_gt(input_data,gt_data)
        #     input_data,gt_data,scales = point_operation.random_scale_point_cloud_and_gt(input_data,gt_data,
        #         scale_low=0.8,
        #         scale_high=1.2)
            # radius_data = radius_data * scales
        pair = {
            'pcl_low': torch.Tensor(input_data).clone(),
            'pcl_high': torch.Tensor(gt_data).clone(),
            # 'radius': radius_data.clone()
        }
        # print("pair", pair.shape)
        return pair
        # return input_data,gt_data,radius_data

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

if __name__ =='__main__':
    mm = PU1KDataset('/home/huxin/Documents/code/upsample/data/PU1K/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5',256,4,1,True)##, type='train')
    print(mm)
    ss = DataLoader(mm, batch_size=2, num_workers=0, shuffle=True)
    for _,data in enumerate(ss):
        input_data = data['pcl_low']
        gt_data = data['pcl_high']
        # print('input_Data', input_data)
        # print("gt_data", gt_data)
        # radius = data[2]
        m=0