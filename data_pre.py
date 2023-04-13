import os
import argparse
import multiprocessing as mp
import logging
import numpy as np
import open3d as o3d
logging.basicConfig(level=logging.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='/media/huxin/MyDisk/upsample_history/upsample_edit8/data/Mixed')
args = parser.parse_args()

DATASET_ROOT = args.dataset_root
MESH_ROOT = os.path.join(DATASET_ROOT, 'meshes/test')
POINTCLOUD_ROOT = os.path.join(DATASET_ROOT, 'pointclouds/test')
RESOLUTIONS = [256, 512]
SAMPLERS = ['poisson']#, 'random']
SUBSETS = ['test']#['train', 'test']
NUM_WORKERS = 32


def poisson_sample(mesh, num_points):
    pc = mesh.sample_points_poisson_disk(num_points)
    pc = np.asarray(pc.points)
    # if pc.shape[0] > num_points:
    #     pc = pc[:num_points, :]
    # else:
    #     compl = mesh.sample_points_uniformly(num_points - pc.shape[0])
    #     # Notice: if (num_points - pc.shape[0]) == 1, sample_mesh_random will
    #     #          return a tensor of size (3, ) but not (1, 3)
    #     compl = np.reshape(compl, [-1, 3])
    #     pc = np.concatenate([pc, compl], axis=0)
    return pc

def random_sample(mesh, num_points):
    pc = mesh.sample_points_poisson_disk(num_points*3)
    pc = np.asarray(pc.points)
    pc = np.random.permutation(pc.shape[0])
    pc = pc[:num_points,:]
    return pc


def enum_configs():
    # for subset in SUBSETS:
    #     for resolution in RESOLUTIONS:
    #         for sampler in SAMPLERS:
    #             yield (subset, resolution, sampler)
    for resolution in RESOLUTIONS:
        for sampler in SAMPLERS:
            yield (resolution, sampler)


def enum_meshes():
    # for subset, resolution, sampler in enum_configs():
    for resolution, sampler in enum_configs():
        in_dir = os.path.join(MESH_ROOT)
        out_dir = os.path.join(POINTCLOUD_ROOT, '%d_%s' % (resolution, sampler))
        if not os.path.exists(in_dir):
            os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        # else:
        #     continue
        for fn in os.listdir(in_dir):
            if fn[-3:] == 'off':
                basename = fn[:-4]
                yield (resolution, sampler,
                        os.path.join(in_dir, fn),
                        os.path.join(out_dir, basename+'.xyz'))


def process(args):
    resolution, sampler, in_file, out_file = args
    if os.path.exists(out_file):
        logging.info('Already exists: ' + in_file)
        return
    logging.info('Start processing: [%d,%s] %s' % (resolution, sampler, in_file))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # v, f, n = pcu.read_off(in_file)
    mesh = o3d.io.read_triangle_mesh(in_file)
    if sampler == 'poisson':
        pointcloud = poisson_sample(mesh, resolution)
    elif sampler == 'random':
        pointcloud = random_sample(mesh, resolution)
    else:
        raise ValueError('Unknown sampler: ' + sampler)
    np.savetxt(out_file, pointcloud, '%.6f')
    # o3d.io.write_point_cloud(out_file, pointcloud)


if __name__ == '__main__':
    # if NUM_WORKERS > 1:
    #     with mp.Pool(processes=NUM_WORKERS) as pool:
    #         pool.map(process, enum_meshes())
    # else:
    for args in enum_meshes():
        process(args)
