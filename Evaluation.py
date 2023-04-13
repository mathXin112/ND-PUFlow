import os
import torch
import pytorch3d
import pytorch3d.loss
import numpy as np
import pandas as pd
# import point_cloud_utils as pcu
import trimesh
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

# from datasets.resample import pcl_save
from models.data_loss import *
from utils.misc import BlackHole


def load_xyz(xyz_dir):
    all_pcls = {}
    for fn in tqdm(os.listdir(xyz_dir), desc='Loading'):
        if fn[-3:] != 'xyz':
            continue
        name = fn[:-4]
        path = os.path.join(xyz_dir, fn)
        all_pcls[name] = torch.FloatTensor(np.loadtxt(path, dtype=np.float32))
    return all_pcls

def load_off(off_dir):
    all_meshes = {}
    for fn in tqdm(os.listdir(off_dir), desc='Loading'):
        if fn[-3:] != 'off':
            continue
        name = fn[:-4]
        path = os.path.join(off_dir, fn)
        obj = trimesh.load(path)
        verts = obj.vertices
        faces = obj.faces
        # verts, faces, _ = pcu.read_off(path)
        verts = torch.FloatTensor(verts)
        faces = torch.LongTensor(faces)
        all_meshes[name] = {'verts': verts, 'faces': faces}
    return all_meshes


class Evaluator(object):
    def __init__(self, output_pcl_dir, gts_pcl_dir, gts_mesh_dir, device='cuda',
                 res_gts='8192_poisson', logger=BlackHole()):
        super().__init__()
        self.output_pcl_dir = output_pcl_dir
        self.gts_pcl_dir = gts_pcl_dir
        self.gts_mesh_dir = gts_mesh_dir
        self.res_gts = res_gts
        self.device = device
        self.pcls_up = load_xyz(self.output_pcl_dir)
        self.pcls_high = load_xyz(self.gts_pcl_dir)
        self.meshes = load_off(self.gts_mesh_dir)
        self.pcls_name = list(self.pcls_up.keys())

    def run(self):
        pcls_up, pcls_high, pcls_name = self.pcls_up, self.pcls_high, self.pcls_name
        results = {}
        for name in tqdm(pcls_name, desc='Evaluate'):
            if name not in self.pcls_high:
                continue

            pcl_up = pcls_up[name][:,:3].unsqueeze(0).to(self.device)
            if pcl_up.size(-1) == 6:
                pcl_up = pcl_up[:, :, :3]

            pcl_high = pcls_high[name][:,:3].unsqueeze(0).to(self.device)
            verts = self.meshes[name]['verts'].to(self.device)
            faces = self.meshes[name]['faces'].to(self.device)
            # verts = self.meshes['wawa']['verts'].to(self.device)
            # faces = self.meshes['wawa']['faces'].to(self.device)

            cd = pytorch3d.loss.chamfer_distance(pcl_up, pcl_high)[0].item()
            cd_sph = chamfer_distance_unit_sphere(pcl_up, pcl_high)[0].item()
            hd_sph = hausdorff_distance_unit_sphere(pcl_up, pcl_high).item()

            # p2f = point_to_mesh_distance_single_unit_sphere(
            #     pcl=pcl_up[0],
            #     verts=verts,
            #     faces=faces
            # ).sqrt().mean().item()
            p2f = point_mesh_bidir_distance_single_unit_sphere(
                pcl=pcl_up[0],
                verts=verts,
                faces=faces
            ).item()

            results[name] = {
                'cd': cd,
                'cd_sph': cd_sph,
                'p2f': p2f,
                'hd_sph': hd_sph,
            }

        results = pd.DataFrame(results).transpose()
        res_mean = results.mean(axis=0)
        res_std = results.std(axis=0)
        # self.logger.info("\n" + repr(results))
        # self.logger.info("\nMean\n" + '\n'.join([
        #     '%s\t%.12f' % (k, v) for k, v in res_mean.items()
        # ]))
        for k, v in res_mean.items():
            print(k, v)
        # print('cd:', cd)
        # print('hd:', hd_sph)
        # print('p2f:', p2f)
        # print(res_mean)


if __name__ == '__main__':
#     output_pcl_dir = '/media/huxin/MyDisk/upsample_history/upsample_edit8/Compare/Self-PU-GAN/sacpcu-PU-GAN/pcl'
#     gts_pcl_dir = '/media/huxin/MyDisk/upsample_history/upsample_edit8/data/Mixed/pointclouds/test/8192_poisson'
#     gts_mesh_dir = '/media/huxin/MyDisk/upsample_history/upsample_edit8/data/Mixed/meshes/test'

    output_pcl_dir = '/media/huxin/MyDisk/upsample_history/upsample_edit8/Compare/Self-PU1K/sapcu_pu1k_out/pcl'
    gts_pcl_dir = '/media/huxin/MyDisk/upsample_history/upsample_edit8/data/PU1K/test/pointclouds/8192_poisson'
    gts_mesh_dir = '/media/huxin/MyDisk/upsample_history/upsample_edit8/data/PU1K/test/original_meshes'

    # output_pcl_dir = '/media/huxin/MyDisk/upsample_history/upsample_edit8/Compare/Self-sketchfab/sapcu_sketchfab/pcl'
    # gts_pcl_dir = '/media/huxin/MyDisk/upsample_history/upsample_edit8/data/Sketchfab/pointclouds/test/8192_poisson'
    # gts_mesh_dir = '/media/huxin/MyDisk/upsample_history/upsample_edit8/data/Sketchfab/meshes/test'

    evaluator = Evaluator(output_pcl_dir, gts_pcl_dir, gts_mesh_dir)
    evaluator.run()
