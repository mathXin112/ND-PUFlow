import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import *
from utils.misc import *
from utils.patch_upsample import *
from utils.evaluate import *
from models.upsample import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='logs/PU-GAN/ckpt_pugan_self.pt')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--upsample_rate', type=int, default=4)
parser.add_argument('--rate_mult', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--dataset_root', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='PU-GAN')
parser.add_argument('--res_input', type=str, default='2048_poisson')
parser.add_argument('--res_gts', type=str, default='8192_poisson')
parser.add_argument('--output_dir', type=str, default='./results')
parser.add_argument('--aug_scale_d', type=float, default=0.3)
args = parser.parse_args()

# Logging
save_title = '{dataset}_Ours_{tag}_{res}_{rate}_{time}'.format_map({
    'dataset': args.dataset,
    'tag': args.tag,
    'res': args.res_input,
    'rate': args.upsample_rate,
    'time': time.strftime('%m-%d-%H-%M-%S', time.localtime())
})
save_dir = os.path.join(args.output_dir, save_title)
os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
logger.info('Loading checkpoint...')
ckpt = torch.load(args.ckpt)
seed_all(ckpt['args'].seed)

# Datasets and loaders
res_gts = '%d_%s' % (
    int(args.res_input.split('_')[0]) * args.upsample_rate,
    'poisson',
)
dataset_dir = os.path.join(args.dataset_root, args.dataset, 'pointclouds')
test_dset = PairedPointCloudDataset(
    root=dataset_dir,
    subset='test',
    cat_low=args.res_input,
    cat_high=args.res_gts,
)

# Model
logger.info('Loading model...')
model = UpsampleNet(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

# Upsample
pcl_results = []

t0 =time.time()
for data in tqdm(test_dset):
    pcl_low = data['pcl_low'].to(args.device)
    pcl_up = patch_based_upsample(
        args=args,
        model=model,
        pcl=pcl_low,
        patch_size=64
    )
    pcl_results.append({
        'name': data['name'],
        'pcl': pcl_up.cpu().detach().numpy()
    })

# Save
logger.info('Saving to: %s' % save_dir)
pcl_dir = os.path.join(save_dir, 'pcl')
os.makedirs(pcl_dir)
for pcl in tqdm(pcl_results, desc='Save'):
    np.savetxt(os.path.join(pcl_dir, pcl['name'] + '.xyz'), pcl['pcl'], fmt='%.6f')
print("Total test time:", time.time() - t0)


# Evaluate
evaluator = Evaluator(
    output_pcl_dir=pcl_dir,
    summary_dir=args.output_dir,
    experiment_name=save_title,
    dataset_root=args.dataset_root,
    dataset=args.dataset,
    device=args.device,
    res_gts=args.res_gts,
    logger=logger
)
evaluator.run()
