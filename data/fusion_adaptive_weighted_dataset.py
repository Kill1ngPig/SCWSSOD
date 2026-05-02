#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import torch
import numpy as np
try:
    from skimage.color import rgb2lab
    from skimage.segmentation import slic
except ImportError as exc:
    raise ImportError(
        'fusion_adaptive_weighted_dataset requires scikit-image. '
        'Install it with: pip install scikit-image'
    ) from exc

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
            raise ValueError('fusion_adaptive_weighted_dataset only supports train mode')
        if cfg.mask_dir != 'fusion_larger':
            raise ValueError('adaptive weighted training only supports mask_dir=fusion_larger')

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

    def _load_sample(self, idx):
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
        mask = mask.astype(np.uint8)
        intersection_mask = intersection_mask.astype(np.uint8)
        height, width = mask.shape
        return image, mask, intersection_mask, (height, width), os.path.basename(maskpath)

    def _transform_train(self, image, mask, intersection_mask):
        image = cv2.resize(image, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.size, self.size), interpolation=cv2.INTER_NEAREST)
        intersection_mask = cv2.resize(
            intersection_mask,
            dsize=(self.size, self.size),
            interpolation=cv2.INTER_NEAREST
        )

        if np.random.randint(2) == 1:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()
            intersection_mask = intersection_mask[:, ::-1].copy()

        return image, mask, intersection_mask

    def _build_weight_map(self, image, mask, intersection_mask):
        pure_fg = (intersection_mask == 1) & (mask == 1)
        impure_fg = (mask == 1) & (~pure_fg)
        background = mask == 2
        unknown = mask == 0

        weight_map = np.zeros(mask.shape, dtype=np.float32)
        weight_map[background] = 1.0
        weight_map[pure_fg] = 1.0

        if not np.any(impure_fg):
            return weight_map

        min_weight = float(self.cfg.impure_min_weight)
        max_weight = float(self.cfg.impure_max_weight)

        if not np.any(pure_fg):
            weight_map[impure_fg] = min_weight
            weight_map[unknown] = 0.0
            return weight_map

        image_01 = np.clip(image / 255.0, 0.0, 1.0)
        image_lab = rgb2lab(image_01)
        segments = self._run_slic(image_01)

        pure_centers, pure_colors = self._collect_pure_superpixels(segments, pure_fg, image_lab)
        if pure_centers.size == 0:
            weight_map[impure_fg] = min_weight
            weight_map[unknown] = 0.0
            return weight_map

        alpha = float(self.cfg.alpha)
        distance_temperature = float(self.cfg.distance_temperature)
        color_temperature = float(self.cfg.color_temperature)

        for segment_id in np.unique(segments[impure_fg]):
            segment_mask = segments == segment_id
            segment_impure = segment_mask & impure_fg
            if not np.any(segment_impure):
                continue

            coords = np.argwhere(segment_impure)
            segment_center = coords.mean(axis=0)
            segment_color = image_lab[segment_impure].mean(axis=0)

            distances = np.linalg.norm(pure_centers - segment_center[None, :], axis=1)
            nearest_idx = int(np.argmin(distances))
            nearest_distance = distances[nearest_idx]
            color_distance = np.linalg.norm(pure_colors[nearest_idx] - segment_color)

            distance_score = np.exp(-nearest_distance / max(distance_temperature, 1e-6))
            color_score = np.exp(-color_distance / max(color_temperature, 1e-6))
            reliability = alpha * distance_score + (1.0 - alpha) * color_score
            reliability = float(np.clip(reliability, 0.0, 1.0))
            weight = min_weight + (max_weight - min_weight) * reliability
            weight_map[segment_impure] = weight

        weight_map[unknown] = 0.0
        return weight_map

    @staticmethod
    def _collect_pure_superpixels(segments, pure_fg, image_lab):
        centers = []
        colors = []
        for segment_id in np.unique(segments[pure_fg]):
            segment_pure = (segments == segment_id) & pure_fg
            if not np.any(segment_pure):
                continue
            centers.append(np.argwhere(segment_pure).mean(axis=0))
            colors.append(image_lab[segment_pure].mean(axis=0))

        if not centers:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        return np.asarray(centers, dtype=np.float32), np.asarray(colors, dtype=np.float32)

    def _run_slic(self, image_01):
        kwargs = {
            'n_segments': int(self.cfg.num_segments),
            'compactness': float(self.cfg.compactness),
            'enforce_connectivity': True
        }
        try:
            return slic(image_01, multichannel=True, **kwargs)
        except TypeError:
            return slic(image_01, channel_axis=-1, start_label=0, **kwargs)

    def __getitem__(self, idx):
        image, mask, intersection_mask, (height, width), filename = self._load_sample(idx)
        image, mask, intersection_mask = self._transform_train(image, mask, intersection_mask)

        weight_map = self._build_weight_map(image, mask, intersection_mask)

        train_mask = mask.astype(np.float32)
        train_mask[train_mask == 0.] = 255.
        train_mask[train_mask == 2.] = 0.

        image = (image - self.cfg.mean) / self.cfg.std
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        train_mask = torch.from_numpy(train_mask).unsqueeze(0)
        weight_map = torch.from_numpy(weight_map).unsqueeze(0)

        return image, train_mask, weight_map, (height, width), filename

    def __len__(self):
        return len(self.samples)
