import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset
from loguru import logger
from src.utils.dataset import read_megadepth_depth, read_megadepth_color


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = Path(npz_path).stem
        self.image_root = kwargs.get('image_root', None) or root_dir
        self.depth_root = kwargs.get('depth_root', None) or root_dir
        self.image_path_prefix_to_strip = kwargs.get('image_path_prefix_to_strip', '') or ''
        self.depth_path_prefix_to_strip = kwargs.get('depth_path_prefix_to_strip', '') or ''
        self._image_abs_cache = {}
        self._depth_abs_cache = {}

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]
        before_filter = len(self.pair_infos)
        self.pair_infos = [pair_info for pair_info in self.pair_infos if self._pair_has_valid_entries(pair_info)]
        dropped = before_filter - len(self.pair_infos)
        if dropped > 0:
            logger.warning(
                f"[{self.scene_id}] dropped {dropped}/{before_filter} invalid pairs due to missing image/depth files; "
                f"kept {len(self.pair_infos)} pairs."
            )

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)  #

    @staticmethod
    def _to_path_str(raw_path):
        if isinstance(raw_path, bytes):
            return raw_path.decode('utf-8')
        if raw_path is None:
            return ''
        return str(raw_path)

    def _normalize_rel_path(self, raw_path, strip_prefix=''):
        rel = self._to_path_str(raw_path).strip()
        if rel.lower() == 'none' or rel == '':
            return ''
        if strip_prefix and rel.startswith(strip_prefix):
            rel = rel[len(strip_prefix):]
        return rel

    def _resolve_image_abs_path(self, raw_path):
        raw_key = self._to_path_str(raw_path)
        if raw_key in self._image_abs_cache:
            return self._image_abs_cache[raw_key]

        rel = self._normalize_rel_path(raw_path, self.image_path_prefix_to_strip)
        if rel == '':
            self._image_abs_cache[raw_key] = ''
            return ''

        p = Path(self.image_root) / rel
        if p.exists():
            out = str(p)
            self._image_abs_cache[raw_key] = out
            return out

        stem = p.with_suffix('')
        for suffix in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = stem.with_suffix(suffix)
            if candidate.exists():
                out = str(candidate)
                self._image_abs_cache[raw_key] = out
                return out

        self._image_abs_cache[raw_key] = ''
        return ''

    def _resolve_depth_abs_path(self, raw_path):
        raw_key = self._to_path_str(raw_path)
        if raw_key in self._depth_abs_cache:
            return self._depth_abs_cache[raw_key]

        rel = self._normalize_rel_path(raw_path, self.depth_path_prefix_to_strip)
        if rel == '':
            self._depth_abs_cache[raw_key] = ''
            return ''

        p = Path(self.depth_root) / rel
        out = str(p) if p.exists() else ''
        self._depth_abs_cache[raw_key] = out
        return out

    def _pair_has_valid_entries(self, pair_info):
        (idx0, idx1), _, _ = pair_info
        img0 = self._resolve_image_abs_path(self.scene_info['image_paths'][idx0])
        img1 = self._resolve_image_abs_path(self.scene_info['image_paths'][idx1])
        if img0 == '' or img1 == '':
            return False
        if self.mode in ['train', 'val']:
            dep0 = self._resolve_depth_abs_path(self.scene_info['depth_paths'][idx0])
            dep1 = self._resolve_depth_abs_path(self.scene_info['depth_paths'][idx1])
            if dep0 == '' or dep1 == '':
                return False
        return True

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = self._resolve_image_abs_path(self.scene_info['image_paths'][idx0])
        img_name1 = self._resolve_image_abs_path(self.scene_info['image_paths'][idx1])
        if not img_name0 or not img_name1:
            raise RuntimeError(f"Invalid image path entry in scene {self.scene_id}, pair idx {idx}.")

        imagec_0, scale0, mask0, prepad_size0 = read_megadepth_color(
            img_name0, self.img_resize, self.df, padding=True)
        imagec_1, scale1, mask1, prepad_size1 = read_megadepth_color(
            img_name1, self.img_resize, self.df, padding=True)

        # read depth. shape: (h, w)
        if self.mode in ['train', 'val']:
            depth_name0 = self._resolve_depth_abs_path(self.scene_info['depth_paths'][idx0])
            depth_name1 = self._resolve_depth_abs_path(self.scene_info['depth_paths'][idx1])
            if not depth_name0 or not depth_name1:
                raise RuntimeError(f"Invalid depth path entry in scene {self.scene_id}, pair idx {idx}.")
            depth0 = read_megadepth_depth(
                depth_name0, pad_to=self.depth_max_size)
            depth1 = read_megadepth_depth(
                depth_name1, pad_to=self.depth_max_size)
        else:
            depth0 = depth1 = torch.tensor([])

        # read intrinsics of original size
        K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        data = {
            'imagec_0': imagec_0.squeeze(0),  # (1, h, w, 3)
            'imagec_1': imagec_1.squeeze(0),  # (1, h, w, 3)
            'depth0': depth0,  # (h, w)
            'depth1': depth1,
            'prepad_size0': prepad_size0,
            'prepad_size1': prepad_size1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            'mask0': mask0,
            'mask1': mask1,
        }

        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data
