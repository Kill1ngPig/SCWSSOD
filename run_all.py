#!/usr/bin/python3
# coding=utf-8

import os
import os.path as osp
import argparse
import subprocess

TRAIN_MASK_DIRS = ['fusion_intersection', 'fusion_larger', 'fusion_union', 'only_dot']


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
    parser.add_argument('--only_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--save_pred', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    for mask_dir in TRAIN_MASK_DIRS:
        exp_name = f"scwssod_{mask_dir}"
        model_path = osp.join(args.save_root, exp_name, 'model-last.pt')

        # 1) train on one training label set
        if not args.only_test:
            train_cmd = [
                'python', 'train.py',
                '--datapath', args.datapath,
                '--mask_dir', mask_dir,
                '--save_root', args.save_root,
                '--epoch', str(args.epoch),
                '--batch', str(args.batch),
            ]
            run_cmd(train_cmd)

        # 2) test this trained model on all fixed public test datasets
        if not args.only_train:
            test_cmd = [
                'python', 'test.py',
                '--model_path', model_path,
                '--tag', exp_name,
            ]
            if args.save_pred:
                test_cmd.append('--save_pred')

            run_cmd(test_cmd)


if __name__ == '__main__':
    main()