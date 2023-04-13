import os
import sys
import argparse
import torch
import numpy as np
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from datasets import *
from datasets.paired import PairedPointCloudDataset
from datasets.resample_aug import *
from utils.misc import *
from utils.patch_upsample import *
from models.upsample import *
from models.data_loss import *
from args import get_args

args = get_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, args)
    code_dir = os.path.join(log_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)
    os.system('cp train_pugan.py %s/' % code_dir)
    os.system('cp args.py %s/' % code_dir)
    os.system('cp models/upsample.py %s/' % code_dir)
    os.system('cp models/feature.py %s/' % code_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)
logger.info(repr(sys.argv))

# Resume from ckpt: arguments
if args.resume is not None:
    logger.info('Resuming from %s' % args.resume)
    resume_ckpt = torch.load(args.resume)
    args_resume = resume_ckpt['args']

    args_resume.max_iters = args.max_iters
    args_resume.val_freq = args.val_freq
    args_resume.val_num_visualize = args.val_num_visualize
    args_resume.upsample_rate = args.upsample_rate
    args_resume.val_res_low = args.val_res_low
    args_resume.val_res_high = args.val_res_high
    args_resume.resume = args.resume
    args = args_resume

# Datasets and loaders
logger.info('Loading datasets')
dataset_dir = os.path.join(args.dataset_root, args.dataset, 'pointclouds')

if args.manner_train == 'supervised':
    train_dset = Make_Patch(
                args=args,
                root=dataset_dir,
                subset='train',
                cat_low=args.resolutions_low,
                transform=standard_train_transforms(noise_std=args.aug_noise, label_noise=args.aug_label_noise),)
elif args.manner_train == 'self-supervised':
    train_dset = Make_Patch_Supervised(
                args=args,
                root=dataset_dir,
                subset='train',
                cat_low=args.resolutions_low,
                cat_high=args.resolutions_high,
                transform=standard_train_transforms(noise_std=args.aug_noise, label_noise=args.aug_label_noise),)

val_dset = PairedPointCloudDataset(
    root=dataset_dir,
    subset='test',
    cat_low=args.val_res_low,
    cat_high=args.val_res_high,
    transform=None,
)

train_loader = DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True)

# Model
logger.info('Building model...')
model = UpsampleNet(args).to(args.device)
logger.info(repr(model))

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5*1e-4 )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100000,eta_min=1e-8)
# Resume from ckpt: model, optimizer and scheduler
if args.resume is not None:
    model.load_state_dict(resume_ckpt['state_dict'])
    optimizer.load_state_dict(resume_ckpt['others']['optimizer'])
    scheduler.load_state_dict(resume_ckpt['others']['scheduler'])



def train(it):
    # Load data
    n_batches = len(train_loader)
    for idx, data in enumerate(train_loader):

        pcl_low = data['pcl_low'].to(args.device)
        pcl_high = data['pcl_high'].to(args.device)

        # Reset grad and model state
        optimizer.zero_grad()
        model.train()

        # Forward
        pcl_up_flow = model.upsample(pcl_low, pcl_high, rate=args.upsample_rate, fps=True,
                                                     rate_mult=args.rate_mult, state='train', it=it)
        loss_cd_high = chamfer_distance_unit_sphere(pcl_up_flow, pcl_high, batch_reduction='mean')[0]
        loss_emd_high = torch.mean(emd_loss(pcl_up_flow, pcl_high)[0])
        loss_density = DensityLoss(pcl_high, pcl_up_flow, knn=3)
        loss_flow = loss_cd_high + loss_emd_high
        loss = loss_flow
        loss.backward()

        # Backward and optimize
        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if orig_grad_norm > 500:
            print("Grad Problem!")

        # Logging
        n_itr = (it - 1) * n_batches + idx
        logger.info('[Train] Iter %04d | Loss %.6f | Grad %.6f' % (
            n_itr, loss.item(), orig_grad_norm,
        ))
        writer.add_scalar('train/loss', loss, n_itr)
        # writer.add_scalar('train/loss_log', loss, n_itr)
        writer.add_scalar('train/cd_loss', loss_cd_high, n_itr)
        writer.add_scalar('train/emd_loss', loss_emd_high, n_itr)
        writer.add_scalar('train/density_loss', loss_density, n_itr)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], n_itr)
        writer.add_scalar('train/grad_norm', orig_grad_norm, n_itr)
        writer.flush()

        scheduler.step()


def validate(it):
    all_high = []
    all_up = []
    for i, data in enumerate(tqdm(val_dset, desc='Validate')):
        pcl_low = data['pcl_low'].to(args.device)   # (N, 3)
        pcl_high = data['pcl_high'].to(args.device) # (rN, 3)
        pcl_up = patch_based_upsample(args=args, model=model, pcl=pcl_low, patch_size=args.patch_size) # (rN, 3)
        all_high.append(pcl_high.unsqueeze(0))
        all_up.append(pcl_up.unsqueeze(0))
    all_high = torch.cat(all_high, dim=0)
    all_up = torch.cat(all_up, dim=0)

    avg_chamfer = chamfer_distance_unit_sphere(all_up, all_high, batch_reduction='mean')[0].item()

    logger.info('[Val] Iter %04d | CD %.6f  ' % (it, avg_chamfer))
    writer.add_scalar('val/chamfer', avg_chamfer, it)
    writer.add_mesh('val/pcl', all_up[:args.val_num_visualize], global_step=it)
    writer.flush()

    return avg_chamfer

# Main loop
logger.info('Start training...')
try:
    for it in range(1, args.max_iters+1):
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            cd_loss = validate(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)

except KeyboardInterrupt:
    logger.info('Terminating...')
