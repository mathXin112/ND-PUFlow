import argparse
from utils.misc import str_list


# Arguments

NONLINEARITIES = ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
LAYERS = ["ignore", "concat", "concat_v2", "squash", "concatsquash", "scale", "concatscale"]

def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))

def add_args(parser):
    ## Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='PU-GAN')
    parser.add_argument('--patch_ratio', type=int, default=4,
                        help='Recommended: greater than dataset resolution ratio.')
    parser.add_argument('--resolutions_low', type=str, default='2048_poisson')
    parser.add_argument('--resolutions_high', type=str, default='8192_poisson')
    parser.add_argument('--manner_train', type=str, default='supervised', choices=['supervised', 'self-supervised'])
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--num_patches', type=int, default=195)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--aug_noise', type=float, default=0.1)
    parser.add_argument('--aug_scale_d', type=float, default=0.3)
    parser.add_argument('--aug_degree', type=float, default=180.0)
    parser.add_argument('--aug_label_noise', type=eval, default=False, choices=[True, False])
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--cho_k', type=int, default=6)
    ## Model architecture
    parser.add_argument('--frame_knn', type=int, default=8)
    parser.add_argument('--frame_scale', type=float, default=0.15)
    parser.add_argument("--flow", type=str, default="cnf", choices=['cnf', 'acl'])
    parser.add_argument('--flow_hidden_dims', type=int_tuple, default=(256, 256,))
    parser.add_argument("--flow_num_blocks", type=int, default=1,
                        help='Number of stacked CNFs.')
    parser.add_argument("--layer_type", type=str, default="concatsquash", choices=LAYERS)
    parser.add_argument('--time_length', type=float, default=0.5)
    parser.add_argument('--train_T', type=eval, default=True, choices=[True, False])
    parser.add_argument('--use_adjoint', type=eval, default=True, choices=[True, False])
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--if_bn', type=eval, default=True, choices=[True, False])
    parser.add_argument('--sync_bn', type=eval, default=False, choices=[True, False])
    parser.add_argument('--bn_lag', type=float, default=0)
    parser.add_argument('--std', type=float, default=0.5)
    ## Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=10.0)
    parser.add_argument('--sched_factor', type=float, default=0.5)
    parser.add_argument('--sched_patience', type=int, default=25)
    parser.add_argument('--sched_min_lr', type=float, default=1e-5)
    ## Training
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--logging', type=eval, default=False, choices=[True, False])
    parser.add_argument('--log_root', type=str, default='./logs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--rate_mult', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--val_num_visualize', type=int, default=4)
    parser.add_argument('--upsample_rate', type=int, default=4)
    parser.add_argument('--val_res_low', type=str, default='2048_poisson')
    parser.add_argument('--val_res_high', type=str, default='8192_poisson')
    parser.add_argument('--tag', type=str, default=None)


    return parser

def get_parser():
    ## command line args
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args