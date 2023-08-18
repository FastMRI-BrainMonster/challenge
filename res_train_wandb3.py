import torch
import argparse
import shutil
import os, sys
from pathlib import Path
from utils.data.data_augment import DataAugmentor

if os.getcwd() + '/unet_utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from unet_utils.learning.res_train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix

import wandb

def wandb_init():
    sweep_config = {
        'method': 'random',
        'metric': {'goal': 'minimize', 'name': 'Valid_Loss'},
    }

    parameters_dict = {
        'lr': {
            'values': [1e-5]
        }
    }

    sweep_config['parameters'] = parameters_dict
    
    sweep_id = wandb.sweep(sweep_config, project = 'FastMRI')
    return sweep_id

def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-f', '--base_filters', type=int, default=128, help='Base filters')
    parser.add_argument('-c', '--chanels', type=int, default=1, help='gray:1, color:3')

    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_unet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/home/yxxshin/Desktop/FastMRI_DEV/Data_ResUNet/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/home/yxxshin/Desktop/FastMRI_DEV/Data_ResUNet/val/', help='Directory of validation data')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--input-key', type=str, default='reconstruction', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--is_grappa', type=str, default='y', help='image + grappa image')
    parser.add_argument('--grappa_path', type=str, default='/home/yxxshin/Desktop/FastMRI_DEV/challenge/grappa', help='grappa path')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')

    #[modified]
    return parser

def train_wandb(config = None):
    with wandb.init(config = config):
        config = wandb.config
        args.lr = config.lr

        torch.cuda.empty_cache()
        train(args)

if __name__ == '__main__':
    # [add] parser for augmentation
    parser = parse()
    parser = DataAugmentor.add_augmentation_specific_args(parser)
    args = parser.parse_args()

    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = '../result3' / args.net_name / 'checkpoints'
    args.val_dir = '../result3' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result3' / args.net_name / __file__
    args.val_loss_dir = '../result3' / args.net_name

    if args.is_grappa == 'y':
        args.chanels = 2

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    sweep_id = wandb_init()

    wandb.agent(sweep_id, train_wandb, count = 5)
