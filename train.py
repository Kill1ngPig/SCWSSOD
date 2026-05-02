#!/usr/bin/python3
# coding=utf-8

import sys
import os
import os.path as osp
import argparse
import datetime
import logging as logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data import dataset
from net_agg import SCWSSOD
from lib.data_prefetcher import DataPrefetcher
from lscloss import *
from tools import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BASE_LR = 1e-5
MAX_LR = 1e-2
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
l = 0.3


def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps * ratio)
    last = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur / total_steps)
    x = np.abs(cur * 2.0 / total_steps - 2.0 * cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr) * cur + min_lr * first - base_lr * total_steps) / (first - total_steps)

    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1. - x)
        else:
            momentum = momentums[0]

    return lr, momentum


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data/DUTS')
    parser.add_argument('--mask_dir', type=str, required=True,
                        choices=['fusion_intersection', 'fusion_larger', 'fusion_union', 'only_dot'])
    parser.add_argument('--image_dir', type=str, default='image')
    parser.add_argument('--list_file', type=str, default='train.txt')
    parser.add_argument('--save_root', type=str, default='./runs')
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu', type=str, default='0')
    return parser.parse_args()


def train(Dataset, Network, args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    exp_name = f"scwssod_{args.mask_dir}"
    save_path = osp.join(args.save_root, exp_name)
    os.makedirs(save_path, exist_ok=True)

    logger.basicConfig(
        level=logger.INFO,
        format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=osp.join(save_path, 'train.log'),
        filemode='w'
    )

    cfg = Dataset.Config(
        datapath=args.datapath,
        savepath=save_path,
        mode='train',
        batch=args.batch,
        lr=1e-3,
        momen=0.9,
        decay=5e-4,
        epoch=args.epoch,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        list_file=args.list_file
    )

    data = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=args.num_workers)

    net = Network(cfg)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    loss_lsc = LocalSaliencyCoherence().cuda()

    net.train(True)
    net.cuda()
    criterion.cuda()

    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = torch.optim.SGD(
        [{'params': base}, {'params': head}],
        lr=cfg.lr,
        momentum=cfg.momen,
        weight_decay=cfg.decay,
        nesterov=True
    )

    sw = SummaryWriter(save_path)
    global_step = 0
    db_size = len(loader)

    for epoch in range(cfg.epoch):
        prefetcher = DataPrefetcher(loader)
        batch_idx = -1
        image, mask = prefetcher.next()

        while image is not None:
            batch_idx += 1
            global_step += 1

            niter = epoch * db_size + batch_idx
            lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch * db_size, niter, ratio=1.)
            optimizer.param_groups[0]['lr'] = 0.1 * lr
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum = momentum

            image_scale = F.interpolate(image, scale_factor=0.3, mode='bilinear', align_corners=True)
            out2, out3, out4, out5 = net(image, 'Train')
            out2_s, out3_s, out4_s, out5_s = net(image_scale, 'Train')
            out2_scale = F.interpolate(out2[:, 1:2], scale_factor=0.3, mode='bilinear', align_corners=True)
            loss_ssc = SaliencyStructureConsistency(out2_s[:, 1:2], out2_scale, 0.85)

            gt = mask.squeeze(1).long()
            bg_label = gt.clone()
            fg_label = gt.clone()
            bg_label[gt != 0] = 255
            fg_label[gt == 0] = 255

            image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
            sample = {'rgb': image_}

            out2_ = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss2_lsc = loss_lsc(out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius,
                                 sample, image_.shape[2], image_.shape[3])['loss']
            loss2 = loss_ssc + criterion(out2, fg_label) + criterion(out2, bg_label) + l * loss2_lsc

            out3_ = F.interpolate(out3[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss3_lsc = loss_lsc(out3_, loss_lsc_kernels_desc_defaults, loss_lsc_radius,
                                 sample, image_.shape[2], image_.shape[3])['loss']
            loss3 = criterion(out3, fg_label) + criterion(out3, bg_label) + l * loss3_lsc

            out4_ = F.interpolate(out4[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss4_lsc = loss_lsc(out4_, loss_lsc_kernels_desc_defaults, loss_lsc_radius,
                                 sample, image_.shape[2], image_.shape[3])['loss']
            loss4 = criterion(out4, fg_label) + criterion(out4, bg_label) + l * loss4_lsc

            out5_ = F.interpolate(out5[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss5_lsc = loss_lsc(out5_, loss_lsc_kernels_desc_defaults, loss_lsc_radius,
                                 sample, image_.shape[2], image_.shape[3])['loss']
            loss5 = criterion(out5, fg_label) + criterion(out5, bg_label) + l * loss5_lsc

            loss = loss2 * 1 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            if batch_idx % 10 == 0:
                msg = ('%s | %s | step:%d | epoch:%d/%d | lr=%.6f | '
                       'loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f'
                       ) % (
                    exp_name, datetime.datetime.now(), global_step, epoch + 1, cfg.epoch,
                    optimizer.param_groups[0]['lr'],
                    loss.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()
                )
                print(msg)
                logger.info(msg)

            image, mask = prefetcher.next()

        if epoch > 28:
            torch.save(net.state_dict(), osp.join(save_path, f'model-{epoch + 1}.pt'))

    torch.save(net.state_dict(), osp.join(save_path, 'model-last.pt'))
    print(f'Training finished. Model saved to {save_path}')


if __name__ == '__main__':
    args = parse_args()
    train(dataset, SCWSSOD, args)