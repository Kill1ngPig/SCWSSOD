#!/usr/bin/python3
# coding=utf-8

import os.path as osp
import argparse
import subprocess
import sys


EXP_NAME = 'scwssod_fusion_larger_weighted'


def run_cmd(cmd):
    print('=' * 80)
    print('Running:', ' '.join(cmd))
    print('=' * 80)
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data/DUTS')
    parser.add_argument('--save_root', type=str, default='./runs')
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--intersection_weight', type=float, default=2.0)
    parser.add_argument('--only_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--save_pred', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = osp.join(args.save_root, EXP_NAME, 'model-last.pt')

    if not args.only_test:
        train_cmd = [
            sys.executable, 'train_weighted.py',
            '--datapath', args.datapath,
            '--save_root', args.save_root,
            '--epoch', str(args.epoch),
            '--batch', str(args.batch),
            '--num_workers', str(args.num_workers),
            '--intersection_weight', str(args.intersection_weight),
        ]
        run_cmd(train_cmd)

    if not args.only_train:
        test_cmd = [
            sys.executable, 'test.py',
            '--model_path', model_path,
            '--tag', EXP_NAME,
        ]
        if args.save_pred:
            test_cmd.append('--save_pred')
        run_cmd(test_cmd)


if __name__ == '__main__':
    main()
