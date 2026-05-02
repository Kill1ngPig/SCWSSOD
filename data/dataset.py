#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import torch
import numpy as np

try:
    from . import transform
except:
    import transform

from torch.utils.data import Dataset, DataLoader
from lib.data_prefetcher import DataPrefetcher


class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-12s: %s' % (k, v))

        datapath = self.kwargs['datapath']

        if 'ECSSD' in datapath:
            self.mean = np.array([[[117.15, 112.48, 92.86]]])
            self.std  = np.array([[[56.36, 53.82, 54.23]]])
        elif 'DUTS' in datapath:
            self.mean = np.array([[[124.55, 118.90, 102.94]]])
            self.std  = np.array([[[56.77, 55.97, 57.50]]])
        elif 'DUT-OMRON' in datapath:
            self.mean = np.array([[[120.61, 121.86, 114.92]]])
            self.std  = np.array([[[58.10, 57.16, 61.09]]])
        elif 'MSRA-10K' in datapath:
            self.mean = np.array([[[115.57, 110.48, 100.00]]])
            self.std  = np.array([[[57.55, 54.89, 55.30]]])
        elif 'MSRA-B' in datapath:
            self.mean = np.array([[[114.87, 110.47, 95.76]]])
            self.std  = np.array([[[58.12, 55.30, 55.82]]])
        elif 'SED2' in datapath:
            self.mean = np.array([[[126.34, 133.87, 133.72]]])
            self.std  = np.array([[[45.88, 45.59, 48.13]]])
        elif 'PASCAL-S' in datapath:
            self.mean = np.array([[[117.02, 112.75, 102.48]]])
            self.std  = np.array([[[59.81, 58.96, 60.44]]])
        elif 'HKU-IS' in datapath:
            self.mean = np.array([[[123.58, 121.69, 104.22]]])
            self.std  = np.array([[[55.40, 53.55, 55.19]]])
        elif 'SOD' in datapath:
            self.mean = np.array([[[109.91, 112.13, 93.90]]])
            self.std  = np.array([[[53.29, 50.45, 48.06]]])
        elif 'THUR15K' in datapath:
            self.mean = np.array([[[122.60, 120.28, 104.46]]])
            self.std  = np.array([[[55.99, 55.39, 56.97]]])
        elif 'SOC' in datapath:
            self.mean = np.array([[[120.48, 111.78, 101.27]]])
            self.std  = np.array([[[58.51, 56.73, 56.38]]])
        else:
            self.mean = np.array([[[0.485 * 256, 0.456 * 256, 0.406 * 256]]])
            self.std  = np.array([[[0.229 * 256, 0.224 * 256, 0.225 * 256]]])

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        return None


class Data(Dataset):
    def __init__(self, cfg):
        self.samples = []

        txt_path = os.path.join(cfg.datapath, cfg.list_file)
        image_dir = os.path.join(cfg.datapath, cfg.image_dir)
        mask_dir = os.path.join(cfg.datapath, cfg.mask_dir)

        with open(txt_path, 'r') as lines:
            for line in lines:
                name = line.strip()
                imagepath = os.path.join(image_dir, name + '.jpg')
                maskpath  = os.path.join(mask_dir,  name + '.png')
                self.samples.append([imagepath, maskpath])

        if cfg.mode == 'train':
            self.transform = transform.Compose(
                transform.Normalize(mean=cfg.mean, std=cfg.std),
                transform.Resize(320, 320),
                transform.RandomHorizontalFlip(),
                transform.RandomCrop(320, 320),
                transform.ToTensor()
            )
        elif cfg.mode == 'test':
            self.transform = transform.Compose(
                transform.Normalize(mean=cfg.mean, std=cfg.std),
                transform.Resize(320, 320),
                transform.ToTensor()
            )
        else:
            raise ValueError('cfg.mode must be train or test')

    def __getitem__(self, idx):
        imagepath, maskpath = self.samples[idx]

        image = cv2.imread(imagepath)
        mask  = cv2.imread(maskpath)

        if image is None:
            raise FileNotFoundError(f'Image not found: {imagepath}')
        if mask is None:
            raise FileNotFoundError(f'Mask not found: {maskpath}')

        image = image.astype(np.float32)[:, :, ::-1]
        mask  = mask.astype(np.float32)[:, :, ::-1]

        H, W, C = mask.shape
        image, mask = self.transform(image, mask)

        mask[mask == 0.] = 255.
        mask[mask == 2.] = 0.

        return image, mask, (H, W), os.path.basename(maskpath)

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg = Config(
        mode='train',
        datapath='./data/DUTS',
        list_file='train.txt',
        image_dir='image',
        mask_dir='fusion_union'
    )
    data = Data(cfg)
    loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=4)
    prefetcher = DataPrefetcher(loader)

    image, mask = prefetcher.next()
    image = image[0].permute(1, 2, 0).cpu().numpy() * cfg.std + cfg.mean
    mask  = mask[0].cpu().numpy()

    plt.subplot(121)
    plt.imshow(np.uint8(image))
    plt.subplot(122)
    plt.imshow(mask)
    input()