#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset

try:
    from . import dataset as base_dataset
except ImportError:
    import dataset as base_dataset


class Config(base_dataset.Config):
    pass


class Data(Dataset):
    def __init__(self, cfg):
        if cfg.mode != 'train':
            raise ValueError('fusion_weighted_dataset only supports train mode')
        if cfg.mask_dir != 'fusion_larger':
            raise ValueError('weighted training only supports mask_dir=fusion_larger')

        self.cfg = cfg
        self.size = getattr(cfg, 'trainsize', 320) or 320
        self.samples = []

        txt_path = os.path.join(cfg.datapath, cfg.list_file)
        image_dir = os.path.join(cfg.datapath, cfg.image_dir)
        mask_dir = os.path.join(cfg.datapath, cfg.mask_dir)
        intersection_dir = os.path.join(cfg.datapath, cfg.intersection_dir)

        with open(txt_path, 'r') as lines:
            for line in lines:
                name = line.strip()
                if not name:
                    continue

                imagepath = os.path.join(image_dir, name + '.jpg')
                maskpath = os.path.join(mask_dir, name + '.png')
                intersection_path = os.path.join(intersection_dir, name + '.png')
                self.samples.append((imagepath, maskpath, intersection_path))

    def _transform_train(self, image, mask, intersection_mask):
        image = (image - self.cfg.mean) / self.cfg.std
        image = cv2.resize(image, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
        intersection_mask = cv2.resize(
            intersection_mask,
            dsize=(self.size, self.size),
            interpolation=cv2.INTER_LINEAR
        )

        if np.random.randint(2) == 1:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()
            intersection_mask = intersection_mask[:, ::-1].copy()

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        intersection_mask = torch.from_numpy(intersection_mask).unsqueeze(0)
        return image, mask, intersection_mask

    def __getitem__(self, idx):
        imagepath, maskpath, intersection_path = self.samples[idx]

        image = cv2.imread(imagepath)
        mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
        intersection_mask = cv2.imread(intersection_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f'Image not found: {imagepath}')
        if mask is None:
            raise FileNotFoundError(f'Mask not found: {maskpath}')
        if intersection_mask is None:
            raise FileNotFoundError(f'Intersection mask not found: {intersection_path}')

        image = image.astype(np.float32)[:, :, ::-1]
        mask = mask.astype(np.float32)
        intersection_mask = intersection_mask.astype(np.float32)

        height, width = mask.shape
        image, mask, intersection_mask = self._transform_train(image, mask, intersection_mask)

        train_mask = mask.clone()
        train_mask[train_mask == 0.] = 255.
        train_mask[train_mask == 2.] = 0.

        weight_map = torch.zeros_like(train_mask, dtype=torch.float32)
        valid_mask = train_mask != 255.
        weight_map[valid_mask] = 1.0

        # Reuse foreground pixels from fusion_intersection as the high-confidence area.
        intersection_fg = (intersection_mask.long() == 1) & (train_mask.long() == 1)
        weight_map[intersection_fg] = float(self.cfg.intersection_weight)

        return image, train_mask, weight_map, (height, width), os.path.basename(maskpath)

    def __len__(self):
        return len(self.samples)
