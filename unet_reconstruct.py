import argparse
from pathlib import Path
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from unet_utils.learning.test_part import forward

    
def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_unet', help='Name of network')
    parser.add_argument('-p', '--path_data', type=Path, default='/home/yxxshin/Desktop/FastMRI_DEV/Data_ResUNet/leaderboard/', help='Directory of test data')
    
    parser.add_argument('-f', '--base_filters', type=int, default=64, help='Base filters')
    parser.add_argument('-c', '--chanels', type=int, default=1, help='gray:1, color:3')
    parser.add_argument('--input-key', type=str, default='reconstruction', help='Name of input key')
    parser.add_argument('--is_grappa', type=str, default='y', help='image + grappa image')
    parser.add_argument('--grappa_path', type=str, default='/home/yxxshin/Desktop/FastMRI_DEV/challenge/grappa', help='grappa path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    if args.is_grappa == 'y':
        args.chanels = 2
    # acc4
    args.data_path = args.path_data / "acc4"
    args.grappa_path = os.path.join('/home/yxxshin/Desktop/FastMRI_DEV/challenge/grappa','leaderboard','acc4')
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc4"
    print(args.forward_dir)
    forward(args)
    
    # acc8
    args.data_path = args.path_data / "acc8"
    args.grappa_path = os.path.join('/home/yxxshin/Desktop/FastMRI_DEV/challenge/grappa','leaderboard','acc8')
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc8"
    print(args.forward_dir)
    forward(args)
    
