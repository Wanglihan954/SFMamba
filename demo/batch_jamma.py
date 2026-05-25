import os
import argparse
import traceback
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from loguru import logger

os.sys.path.append("../")

from utlis import JamMa, cfg
from src.utils.dataset import read_megadepth_color


def extract_string(data_item):
    if isinstance(data_item, np.ndarray):
        data_item = data_item.item() if data_item.size == 1 else data_item[0]
    if isinstance(data_item, bytes):
        data_item = data_item.decode("utf-8")
    return str(data_item)


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    if n < 1e-12:
        t_err = 0.0
    else:
        t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
        t_err = np.minimum(t_err, 180 - t_err)

    if np.linalg.norm(t_gt) < ignore_gt_t_thr:
        t_err = 0.0

    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)
    R_err = np.rad2deg(np.abs(np.arccos(cos)))
    return t_err, R_err


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    kpts0_norm = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1_norm = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    ransac_thr = thresh / np.mean([K0[0, 0], K0[1, 1], K1[0, 0], K1[1, 1]])

    E, mask = cv2.findEssentialMat(
        kpts0_norm,
        kpts1_norm,
        np.eye(3),
        threshold=ransac_thr,
        prob=conf,
        method=cv2.RANSAC,
    )
    if E is None:
        return None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, int(len(E) / 3)):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0_norm, kpts1_norm, np.eye(3), 1e9, mask=mask
        )
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n
    return ret


def symmetric_epipolar_distance_numpy(pts0, pts1, E, K0, K1):
    pts0_norm = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1_norm = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    pts0_h = np.hstack((pts0_norm, np.ones((pts0_norm.shape[0], 1))))
    pts1_h = np.hstack((pts1_norm, np.ones((pts1_norm.shape[0], 1))))

    Ep0 = pts0_h @ E.T
    p1Ep0 = np.sum(pts1_h * Ep0, -1)
    Etp1 = pts1_h @ E

    d = p1Ep0 ** 2 * (
        1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2 + 1e-12)
        + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2 + 1e-12)
    )
    return d


def draw_xoftr_style(img0_path, img1_path, mkpts0, mkpts1, K0, K1, T_0to1, save_path):
    pixel_thr = 1.5
    ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh=pixel_thr)

    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    if img0 is None or img1 is None:
        raise RuntimeError(f"读取图像失败: {img0_path} | {img1_path}")

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h_max = max(h0, h1)

    out_img = np.zeros((h_max, w0 + w1, 3), dtype=np.uint8)
    out_img[:h0, :w0] = img0
    out_img[:h1, w0:w0 + w1] = img1

    # 固定字号（对所有输出一致）。如需更改大小，请修改 FONT_SCALE 常量
    FONT_SCALE = 1.6  # 固定基准字号（可调整）
    font_scale = FONT_SCALE
    font_thickness = max(2, int(round(font_scale * 2.5)))
    shadow_thickness = font_thickness + 2
    line_height = int(max(30, round(font_scale * 30)))

    if ret is None:
        text_lines = [
            f"Matches: {len(mkpts0)}",
            "Pose estimation failed",
        ]
        y_offset = line_height
        for line in text_lines:
            cv2.putText(out_img, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), shadow_thickness, cv2.LINE_AA)
            cv2.putText(out_img, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            y_offset += line_height
        cv2.imwrite(save_path, out_img)
        return

    R_pred, t_pred, inliers_mask = ret
    err_t, err_R = relative_pose_error(T_0to1, R_pred, t_pred)

    t_gt = T_0to1[:3, 3]
    tx = np.array([
        [0, -t_gt[2], t_gt[1]],
        [t_gt[2], 0, -t_gt[0]],
        [-t_gt[1], t_gt[0], 0]
    ])
    E_gt = tx @ T_0to1[:3, :3]

    inlier_pts0 = mkpts0[inliers_mask]
    inlier_pts1 = mkpts1[inliers_mask]

    epi_errs = symmetric_epipolar_distance_numpy(inlier_pts0, inlier_pts1, E_gt, K0, K1)

    precision_threshold = 5e-4
    precision_ratio = np.mean(epi_errs < precision_threshold) * 100 if len(epi_errs) > 0 else 0

    sort_indices = np.argsort(epi_errs)[::-1]
    for idx in sort_indices:
        pt1 = (int(inlier_pts0[idx][0]), int(inlier_pts0[idx][1]))
        pt2 = (int(inlier_pts1[idx][0] + w0), int(inlier_pts1[idx][1]))
        err = epi_errs[idx]

        if err < precision_threshold:
            color = (0, 255, 0)
        elif err < 5e-3:
            color = (0, 215, 255)
        else:
            color = (0, 0, 255)

        cv2.line(out_img, pt1, pt2, color, max(1, int(round(font_scale))), lineType=cv2.LINE_AA)
        radius = max(2, int(round(font_scale)))
        cv2.circle(out_img, pt1, radius, color, -1)
        cv2.circle(out_img, pt2, radius, color, -1)

    text_lines = [
        f"err_t: {err_t:.2f} deg",
        f"err_R: {err_R:.2f} deg",
        f"Precision(5e-4): {precision_ratio:.1f}% | Inliers: {len(inlier_pts0)}",
    ]

    y_offset = line_height
    for line in text_lines:
        cv2.putText(out_img, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), shadow_thickness, cv2.LINE_AA)
        cv2.putText(out_img, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        y_offset += line_height

    cv2.imwrite(save_path, out_img)


def load_npz_data(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    if isinstance(data, dict):
        target_data, keys = data, list(data.keys())
    elif hasattr(data, "files"):
        target_data, keys = data, data.files
    elif isinstance(data, np.ndarray) and data.ndim == 0:
        target_data, keys = data.item(), list(data.item().keys())
    else:
        raise ValueError(f"{npz_file} 无法解析结构")

    required_keys = {"image_paths", "intrinsics", "poses", "pair_infos"}
    if not required_keys.issubset(set(keys)):
        raise KeyError(f"{npz_file} 缺少必要字段: {required_keys - set(keys)}")

    image_paths = target_data["image_paths"]
    intrinsics = target_data["intrinsics"]
    poses = target_data["poses"]
    pair_infos = target_data["pair_infos"]

    if isinstance(image_paths, np.ndarray) and image_paths.ndim == 0:
        image_paths = image_paths.item()
    if isinstance(pair_infos, np.ndarray) and pair_infos.ndim == 0:
        pair_infos = pair_infos.item()

    return data, image_paths, intrinsics, poses, pair_infos


def resolve_image_path(root_dir, rel_or_abs_path):
    rel_or_abs_path = extract_string(rel_or_abs_path)
    if os.path.isabs(rel_or_abs_path) and os.path.exists(rel_or_abs_path):
        return rel_or_abs_path

    candidate = os.path.join(root_dir, rel_or_abs_path)
    if os.path.exists(candidate):
        return candidate

    basename_candidate = os.path.join(root_dir, os.path.basename(rel_or_abs_path))
    if os.path.exists(basename_candidate):
        return basename_candidate

    return candidate


def jamma_match_pair(jamma, img0_path, img1_path, image_size):
    device = next(jamma.parameters()).device

    image0, scale0, mask0, _ = read_megadepth_color(img0_path, image_size, 16, True)
    image1, scale1, mask1, _ = read_megadepth_color(img1_path, image_size, 16, True)

    mask0 = F.interpolate(
        mask0[None, None].float(),
        scale_factor=0.125,
        mode="nearest",
        recompute_scale_factor=False,
    )[0].bool()
    mask1 = F.interpolate(
        mask1[None, None].float(),
        scale_factor=0.125,
        mode="nearest",
        recompute_scale_factor=False,
    )[0].bool()

    data = {
        "imagec_0": image0.to(device),
        "imagec_1": image1.to(device),
        "mask0": mask0.to(device),
        "mask1": mask1.to(device),
        "scale0": scale0[None].to(device),
        "scale1": scale1[None].to(device),
    }

    jamma(data)

    if "m_bids" in data:
        b_mask = data["m_bids"] == 0
    else:
        conf_key = "mconf_f" if "mconf_f" in data else "mconf"
        b_mask = torch.ones_like(data[conf_key], dtype=torch.bool)

    if "mkpts0_f" in data and "mkpts1_f" in data:
        mkpts0 = data["mkpts0_f"][b_mask].detach().cpu().numpy()
        mkpts1 = data["mkpts1_f"][b_mask].detach().cpu().numpy()
    elif "mkpts0_c" in data and "mkpts1_c" in data:
        mkpts0 = data["mkpts0_c"][b_mask].detach().cpu().numpy()
        mkpts1 = data["mkpts1_c"][b_mask].detach().cpu().numpy()
    else:
        mkpts0 = np.empty((0, 2), dtype=np.float32)
        mkpts1 = np.empty((0, 2), dtype=np.float32)

    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    if img0 is None or img1 is None:
        raise RuntimeError(f"读取原图失败: {img0_path} | {img1_path}")

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    mkpts0[:, 0] = np.clip(mkpts0[:, 0], 0, w0 - 1)
    mkpts0[:, 1] = np.clip(mkpts0[:, 1], 0, h0 - 1)
    mkpts1[:, 0] = np.clip(mkpts1[:, 0], 0, w1 - 1)
    mkpts1[:, 1] = np.clip(mkpts1[:, 1], 0, h1 - 1)

    return mkpts0, mkpts1


def pair_output_name(npz_file, path0, path1, pair_idx):
    scene = os.path.splitext(os.path.basename(npz_file))[0]
    name0 = os.path.splitext(os.path.basename(path0))[0]
    name1 = os.path.splitext(os.path.basename(path1))[0]
    return f"{scene}_{pair_idx:06d}_{name0}__{name1}.jpg"


def parse_args():
    parser = argparse.ArgumentParser(description="Batch visualize JamMa matches from npz pair_infos")
    parser.add_argument(
        "--npz_files",
        nargs="+",
        required=True,
        help="一个或多个包含 image_paths/intrinsics/poses/pair_infos 的 npz 文件",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="图像根目录。若 npz 中是相对路径，将基于该目录拼接",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/jamma_batch_vis",
        help="输出目录",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=832,
        help="JamMa 输入尺寸",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=-1,
        help="最多处理多少对，-1 表示全部",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    jamma = JamMa(config=cfg,pretrained='/home/ly/9100pro/Reg/JamMa/jamma_log/jamma/version_10/checkpoints/epoch=21-auc@5=0.637-auc@10=0.770-auc@20=0.862.ckpt').eval().to(device)

    total_pairs = 0
    success_pairs = 0
    failed_pairs = 0

    for npz_file in args.npz_files:
        print(f"\n{'=' * 80}")
        print(f"处理 npz: {npz_file}")
        print(f"{'=' * 80}")

        try:
            data, image_paths, intrinsics, poses, pair_infos = load_npz_data(npz_file)
        except Exception as e:
            print(f"❌ 读取 npz 失败: {npz_file}\n{e}")
            failed_pairs += 1
            continue

        for pair_idx, pair in enumerate(pair_infos):
            if args.max_pairs >= 0 and total_pairs >= args.max_pairs:
                print("\n达到 max_pairs，提前结束。")
                if hasattr(data, "close"):
                    data.close()
                print(f"\n总计: {total_pairs}, 成功: {success_pairs}, 失败: {failed_pairs}")
                return

            total_pairs += 1


            try:
                # pair format: (array([id0, id1]), overlap, bbox)
                idxs = np.ravel(pair[0])
                if idxs.size >= 2:
                    id0 = int(idxs[0])
                    id1 = int(idxs[1])
                else:
                    # fallback for legacy formats
                    id0 = int(idxs[0])
                    id1 = int(np.ravel(pair[1])[0])

                rel0 = extract_string(image_paths[id0])
                rel1 = extract_string(image_paths[id1])

                # skip pairs with missing image paths
                if rel0 in (None, 'None', '') or rel1 in (None, 'None', ''):
                    print(f"⚠️ 跳过 pair 索引不完整: pair_idx={pair_idx} id0={id0} id1={id1} rel0={rel0} rel1={rel1}")
                    continue

                img0_path = resolve_image_path(args.image_root, rel0)
                img1_path = resolve_image_path(args.image_root, rel1)

                if not os.path.exists(img0_path):
                    raise FileNotFoundError(f"path0 不存在: {img0_path}")
                if not os.path.exists(img1_path):
                    raise FileNotFoundError(f"path1 不存在: {img1_path}")

                K0 = np.asarray(intrinsics[id0], dtype=np.float64)
                K1 = np.asarray(intrinsics[id1], dtype=np.float64)
                T_0to1 = np.asarray(poses[id1], dtype=np.float64) @ np.linalg.inv(np.asarray(poses[id0], dtype=np.float64))

                print(f"\n[{total_pairs}] {os.path.basename(img0_path)}  <->  {os.path.basename(img1_path)}")

                mkpts0, mkpts1 = jamma_match_pair(jamma, img0_path, img1_path, args.image_size)

                save_name = pair_output_name(npz_file, img0_path, img1_path, pair_idx)
                save_path = os.path.join(args.output_dir, save_name)

                draw_xoftr_style(
                    img0_path=img0_path,
                    img1_path=img1_path,
                    mkpts0=mkpts0,
                    mkpts1=mkpts1,
                    K0=K0,
                    K1=K1,
                    T_0to1=T_0to1,
                    save_path=save_path,
                )

                print(f"✅ 已保存: {save_path}")
                success_pairs += 1

            except Exception as e:
                failed_pairs += 1
                print(f"❌ pair 处理失败: {e}")
                traceback.print_exc()

        if hasattr(data, "close"):
            data.close()

    print(f"\n{'=' * 80}")
    print(f"全部完成")
    print(f"总计: {total_pairs}")
    print(f"成功: {success_pairs}")
    print(f"失败: {failed_pairs}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()