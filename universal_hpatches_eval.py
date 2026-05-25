import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

def compute_pose_auc(errors, thresholds):
    """严格还原顶会论文中的梯形积分 AUC 算法"""
    errors = np.array(errors)
    sort_idx = np.argsort(errors)
    errors = errors[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        auc = np.trapz(r, x=e) / t
        aucs.append(auc)
    return aucs

class UniversalHpatchesBenchmark:
    """通用单应性矩阵评估器 (完全对齐 DKM/CASP 论文尺度)"""
    def __init__(self, dataset_path):
        self.seqs_path = os.path.join(dataset_path, "hpatches-sequences-release")
        self.seq_names = sorted(os.listdir(self.seqs_path))
        # 严格使用 DKM 的 8 个过滤场景
        self.ignore_seqs = set([
            "i_contruction", "i_crownnight", "i_dc", "i_pencils",
            "i_whitebuilding", "v_artisans", "v_astronautis", "v_talent",
        ])

    def evaluate(self, match_fn):
        """
        match_fn: 用户传入的回调函数，只需接收 (img1_path, img2_path)，
                  并返回原图尺寸下的特征点对 (mkpts1, mkpts2)
        """
        homog_dists = []
        for seq_name in tqdm(self.seq_names, desc="Evaluating"):
            if seq_name in self.ignore_seqs:
                continue
            
            im1_path = os.path.join(self.seqs_path, seq_name, "1.ppm")
            im1 = Image.open(im1_path)
            w1, h1 = im1.size

            for im_idx in range(2, 7):
                im2_path = os.path.join(self.seqs_path, seq_name, f"{im_idx}.ppm")
                im2 = Image.open(im2_path)
                w2, h2 = im2.size
                H_gt = np.loadtxt(os.path.join(self.seqs_path, seq_name, f"H_1_{im_idx}"))

                # ==========================================
                # 🚀 核心：调用外部传入的任何模型来获取匹配点
                # ==========================================
                mkpts1, mkpts2 = match_fn(im1_path, im2_path)

                try:
                    # 动态计算阈值
                    ransac_thr = 3.0 * min(w2, h2) / 480.0
                    H_pred, _ = cv2.findHomography(
                        mkpts1, mkpts2, 
                        method=cv2.RANSAC, 
                        confidence=0.99999, 
                        ransacReprojThreshold=ransac_thr
                    )
                except:
                    H_pred = None

                # 极限惩罚
                if H_pred is None:
                    H_pred = np.zeros((3, 3))
                    H_pred[2, 2] = 1.0

                corners = np.array([[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, 0, 1], [w1 - 1, h1 - 1, 1]])
                
                real_warped_corners = np.dot(corners, H_gt.T)
                real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                
                warped_corners = np.dot(corners, H_pred.T)
                warped_corners = warped_corners[:, :2] / (warped_corners[:, 2:] + 1e-8)
                
                mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
                
                # 🌟 绝杀：强行将误差折算回 480 像素的相对标尺
                mean_dist = mean_dist / (min(w2, h2) / 480.0)
                homog_dists.append(mean_dist)

        thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        aucs = compute_pose_auc(homog_dists, thresholds)
        return {
            "AUC @3px": aucs[2] * 100,
            "AUC @5px": aucs[4] * 100,
            "AUC @10px": aucs[9] * 100,
        }
    
