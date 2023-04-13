import os
import argparse
import subprocess
import multiprocessing
from utils.misc import *

# Arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, default='results/2022_10_07__22_08_01/PU1K_Ours__input_2048_4_10-08-10-17-24/pcl')
parser.add_argument('--recons_exe', type=str, default='./mesh_recons/poisson_reconstruction')
parser.add_argument('--sm_distance', type=float, default=0.1)
parser.add_argument('--num_workers', type=int, default=20)
args = parser.parse_args()

# Logging
logger = get_logger('test', args.save_dir)
input_pcl_dir = os.path.join(args.save_dir, 'pcl')
output_mesh_dir = os.path.join(args.save_dir, 'mesh_%f' % args.sm_distance)
os.makedirs(output_mesh_dir, exist_ok=True)

def process(exe_path, in_path, out_path, sm_distance):
    proc = subprocess.run([
        exe_path, 
        in_path, 
        out_path, 
        '-sm_distance', str(sm_distance),
    ])

def enum_tasks():
    for fn in os.listdir(input_pcl_dir):
        output_name = fn[:-4] + '.off'
        yield (
            args.recons_exe,
            os.path.join(input_pcl_dir, fn),
            os.path.join(output_mesh_dir, output_name),
            args.sm_distance,
        )

with multiprocessing.Pool(processes=args.num_workers) as pool:
    pool.starmap(process, enum_tasks())
    
