#!/usr/bin/python3
# coding=utf-8

import argparse
import os
import os.path as osp
import re
import statistics
import subprocess
import sys


MASK_DIRS = ['only_dot', 'fusion_larger']
DATASETS = ['ECSSD', 'PASCAL', 'THUR', 'DUT', 'DUTS_Test']
LOG_PATTERN = re.compile(
    r'\[(?P<tag>[^\]]+)\]\s+'
    r'(?P<dataset>\S+)\s+'
    r'MAE=(?P<mae>\d+\.\d+),\s+'
    r'F-score=(?P<fscore>\d+\.\d+)'
)


def run_cmd(cmd):
    print('=' * 100)
    print('Running:', ' '.join(cmd))
    print('=' * 100)
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data/DUTS')
    parser.add_argument('--save_root', type=str, default='./repeat_runs')
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--only_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--save_pred', action='store_true')
    return parser.parse_args()


def parse_test_log(log_path):
    metrics = {}
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = LOG_PATTERN.search(line)
            if not match:
                continue
            dataset = match.group('dataset')
            metrics[dataset] = {
                'mae': float(match.group('mae')),
                'fscore': float(match.group('fscore')),
            }
    return metrics


def summarize_results(all_results, summary_path):
    lines = []
    for mask_dir in MASK_DIRS:
        lines.append(f'## {mask_dir}')
        for dataset in DATASETS:
            maes = []
            fscores = []
            for repeat_idx in range(1, len(all_results[mask_dir]) + 1):
                metrics = all_results[mask_dir][repeat_idx]
                if dataset not in metrics:
                    continue
                maes.append(metrics[dataset]['mae'])
                fscores.append(metrics[dataset]['fscore'])

            if not maes:
                lines.append(f'- {dataset}: missing')
                continue

            mae_mean = statistics.mean(maes)
            mae_std = statistics.stdev(maes) if len(maes) > 1 else 0.0
            f_mean = statistics.mean(fscores)
            f_std = statistics.stdev(fscores) if len(fscores) > 1 else 0.0
            lines.append(
                f'- {dataset}: '
                f'MAE mean={mae_mean:.6f}, std={mae_std:.6f}; '
                f'F-score mean={f_mean:.6f}, std={f_std:.6f}'
            )
        lines.append('')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines).rstrip() + '\n')

    print(f'Summary written to {summary_path}')


def main():
    args = parse_args()
    all_results = {mask_dir: {} for mask_dir in MASK_DIRS}

    for mask_dir in MASK_DIRS:
        for repeat_idx in range(1, args.repeats + 1):
            run_root = osp.join(args.save_root, f'{mask_dir}_repeat{repeat_idx}')
            exp_name = f'scwssod_{mask_dir}'
            model_path = osp.join(run_root, exp_name, 'model-last.pt')
            test_tag = f'{exp_name}_repeat{repeat_idx}'
            log_path = f'test_{test_tag}.log'

            os.makedirs(run_root, exist_ok=True)

            if not args.only_test:
                train_cmd = [
                    sys.executable, 'train.py',
                    '--datapath', args.datapath,
                    '--mask_dir', mask_dir,
                    '--save_root', run_root,
                    '--epoch', str(args.epoch),
                    '--batch', str(args.batch),
                    '--num_workers', str(args.num_workers),
                ]
                run_cmd(train_cmd)

            if not args.only_train:
                test_cmd = [
                    sys.executable, 'test.py',
                    '--model_path', model_path,
                    '--tag', test_tag,
                ]
                if args.save_pred:
                    test_cmd.append('--save_pred')
                run_cmd(test_cmd)
                all_results[mask_dir][repeat_idx] = parse_test_log(log_path)

    if not args.only_train:
        summary_path = osp.join(args.save_root, 'repeat_summary.md')
        summarize_results(all_results, summary_path)


if __name__ == '__main__':
    main()
