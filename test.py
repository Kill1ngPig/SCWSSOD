#!/usr/bin/python3
# coding=utf-8

import os
import sys
import argparse
# sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from skimage import img_as_ubyte
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib import dataset
from net_agg import SCWSSOD
import time
import logging as logger


GPU_ID = 0
# os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

# 保留你原本的测试集写法
DATASETS = ['./data/ECSSD', './data/PASCAL', './data/THUR', './data/DUT', './data/DUTS_Test']
# DATASETS = ['./data/ECSSD']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to trained model, e.g. ./runs/scwssod_fusion_union/model-last.pt')
    parser.add_argument('--tag', type=str, default=None,
                        help='experiment tag; if not set, use model folder name')
    parser.add_argument('--save_pred', action='store_true',
                        help='whether to save prediction maps')
    return parser.parse_args()


class Test(object):
    def __init__(self, Dataset, datapath, Network, model_path, tag):
        ## dataset
        self.datapath = datapath.split("/")[-1]
        self.tag = tag
        self.model_path = model_path

        print("Testing on %s" % self.datapath)

        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=True, num_workers=8)

        ## network
        self.net = Network(self.cfg)
        state_dict = torch.load(model_path)
        print('complete loading: {}'.format(model_path))
        self.net.load_state_dict(state_dict)
        print('model has {} parameters in total'.format(sum(x.numel() for x in self.net.parameters())))
        self.net.train(False)
        self.net.cuda()
        self.net.eval()

    def accuracy(self):
        with torch.no_grad():
            mae, fscore, cnt, number = 0, 0, 0, 256
            mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)
            cost_time = 0

            for image, mask, (H, W), maskpath in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()

                torch.cuda.synchronize()
                start_time = time.time()
                out2, out3, out4, out5 = self.net(image, 'Test')
                pred = torch.sigmoid(out2)
                torch.cuda.synchronize()
                end_time = time.time()

                cost_time += end_time - start_time

                ## MAE
                cnt += 1
                mae += (pred - mask).abs().mean()

                ## F-score
                precision = torch.zeros(number, device=pred.device)
                recall = torch.zeros(number, device=pred.device)
                for i in range(number):
                    temp = (pred >= threshod[i]).float()
                    precision[i] = (temp * mask).sum() / (temp.sum() + 1e-12)
                    recall[i] = (temp * mask).sum() / (mask.sum() + 1e-12)

                mean_pr += precision
                mean_re += recall
                fscore = mean_pr * mean_re * (1 + 0.3) / (0.3 * mean_pr + mean_re + 1e-12)

                if cnt % 20 == 0:
                    fps = image.shape[0] / (end_time - start_time)
                    print('MAE=%.6f, F-score=%.6f, fps=%.4f' % (
                        (mae / cnt).item(),
                        (fscore.max() / cnt).item(),
                        fps
                    ))

            fps = len(self.loader.dataset) / cost_time
            msg = '[{}] {} MAE={:.6f}, F-score={:.6f}, len(imgs)={}, fps={:.4f}'.format(
                self.tag,
                self.datapath,
                (mae / cnt).item(),
                (fscore.max() / cnt).item(),
                len(self.loader.dataset),
                fps
            )
            print(msg)
            logger.info(msg)

    def save(self):
        with torch.no_grad():
            for image, mask, (H, W), name in self.loader:
                out2, out3, out4, out5 = self.net(image.cuda().float(), 'Test')
                out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)
                pred = (torch.sigmoid(out2[0, 0])).cpu().numpy()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

                # 每个模型单独一个文件夹，下面再按测试集分开
                head = './pred_maps/{}/{}/'.format(self.tag, self.cfg.datapath.split('/')[-1])
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0], img_as_ubyte(pred))


if __name__ == '__main__':
    args = parse_args()

    if args.tag is None:
        # 默认用模型所在文件夹名当 tag
        # 比如 ./runs/scwssod_fusion_union/model-last.pt -> scwssod_fusion_union
        tag = os.path.basename(os.path.dirname(args.model_path))
    else:
        tag = args.tag

    log_file = "test_{}.log".format(tag)
    logger.basicConfig(
        level=logger.INFO,
        format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode="w"
    )

    for e in DATASETS:
        t = Test(dataset, e, SCWSSOD, args.model_path, tag)
        t.accuracy()
        if args.save_pred:
            t.save()