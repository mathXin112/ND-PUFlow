import os
import argparse
import subprocess
import multiprocessing
from utils.misc import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--recons_exe', type=str, default='./mesh_recons/mesh_reconstruction')
parser.add_argument('--knn', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=20)
args = parser.parse_args()

# Logging
logger = get_logger('test', args.save_dir)
input_pcl_dir = os.path.join(args.save_dir, 'pcl')
output_mesh_dir = os.path.join(args.save_dir, 'mesh_k%d' % args.knn)
os.makedirs(output_mesh_dir, exist_ok=True)

def process(exe_path, in_path, out_path, knn):
    proc = subprocess.run([
        exe_path, 
        in_path, 
        out_path, 
        '--knn', str(knn),
    ])

def enum_tasks():
    for fn in os.listdir(input_pcl_dir):
        output_name = fn[:-4] + '.off'
        yield (
            args.recons_exe,
            os.path.join(input_pcl_dir, fn),
            os.path.join(output_mesh_dir, output_name),
            args.knn,
        )

with multiprocessing.Pool(processes=args.num_workers) as pool:
    pool.starmap(process, enum_tasks())
    
