import os
import argparse
import torch
import pytorch3d.ops
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from models.utils import chamfer_distance_unit_sphere
from utils.misc import *
from utils.evaluate import *


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--dataset_root', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='Mixed')
parser.add_argument('--res_gts', type=str, default='8192_poisson')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

# Logging
logger = get_logger('test', args.save_dir)

# Load data
evaluator = Evaluator(
    output_pcl_dir=os.path.join(args.save_dir, 'pcl'),
    summary_dir=os.path.dirname(args.save_dir),
    experiment_name=os.path.basename(args.save_dir),
    dataset_root=args.dataset_root,
    dataset=args.dataset,
    device=args.device,
    res_gts=args.res_gts,
    logger=logger
)
evaluator.run()
