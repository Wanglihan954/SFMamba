import os
import argparse
import numpy as np
import cv2
import torch
import traceback
from loguru import logger
import torch.nn.functional as F

os.sys.path.append("../")

from utlis import JamMa, cfg
from src.utils.dataset import read_megadepth_color


def extract_string(data_item):
    """安全提取字符串"""
    if isinstance(data_item, np.ndarray):
        data_item = data_item.item() if data_item.size == 1 else data_item[0]
    if isinstance(data_item, bytes):
        data_item = data_item.decode("utf-8")
    return str(data_item)


def find_gt_for_pair(npz_files_list, img0_path, img1_path):
    target_name0 = os.path.splitext(os.path.basename(img0_path))[0]
    target_name1 = os.path.splitext(os.path.basename(img1_path))[0]

    for npz_file in npz_files_list:
        print(f"\n[{npz_file}] 正在检索匹配对: '{target_name0}' 与 '{target_name1}' ...")

        if not os.path.exists(npz_file):
            print(f"⚠️ 找不到文件 {npz_file}，将自动跳过...")
            continue

        try:
            data = np.load(npz_file, allow_pickle=True)
            if isinstance(data, dict):
                target_data, keys = data, list(data.keys())
            elif hasattr(data, "files"):
                target_data, keys = data, data.files
            elif isinstance(data, np.ndarray) and data.ndim == 0:
                target_data, keys = data.item(), list(data.item().keys())
            else:
                print(f"⚠️ {npz_file} 无法解析结构，跳过...")
                continue

            required_keys = {"image_paths", "intrinsics", "poses", "pair_infos"}
            if not required_keys.issubset(set(keys)):
                print(f"❌ {npz_file} 缺少必要的键值，跳过。")
                if hasattr(data, "close"):
                    data.close()
                continue

            image_paths = target_data["image_paths"]
            intrinsics = target_data["intrinsics"]
            poses = target_data["poses"]
            pair_infos = target_data["pair_infos"]

            if isinstance(image_paths, np.ndarray) and image_paths.ndim == 0:
                image_paths = image_paths.item()
            if isinstance(pair_infos, np.ndarray) and pair_infos.ndim == 0:
                pair_infos = pair_infos.item()

            for pair in pair_infos:
                id0 = int(np.ravel(pair[0])[0])
                id1 = int(np.ravel(pair[1])[0])

                path_str0 = extract_string(image_paths[id0])
                path_str1 = extract_string(image_paths[id1])

                curr_name0 = os.path.splitext(os.path.basename(path_str0))[0]
                curr_name1 = os.path.splitext(os.path.basename(path_str1))[0]

                # 正向：pair 里就是 (img0, img1)
                if curr_name0 == target_name0 and curr_name1 == target_name1:
                    print(f"✅ 成功命中！(正向匹配，图像索引: {id0} 和 {id1})")
                    K0, K1 = intrinsics[id0], intrinsics[id1]
                    T_0to1 = poses[id1] @ np.linalg.inv(poses[id0])
                    if hasattr(data, "close"):
                        data.close()
                    return K0, K1, T_0to1

                # 反向：pair 里是 (img1, img0)，但用户要的是 img0 -> img1
                elif curr_name0 == target_name1 and curr_name1 == target_name0:
                    print("⚠️ 成功命中！(反向匹配，正在自动调整内参和位姿...)")
                    K0, K1 = intrinsics[id1], intrinsics[id0]
                    T_0to1 = poses[id0] @ np.linalg.inv(poses[id1])
                    if hasattr(data, "close"):
                        data.close()
                    return K0, K1, T_0to1

            print(f"❌ {npz_file} 中没找到。")
            if hasattr(data, "close"):
                data.close()

        except Exception as e:
            print(f"⚠️ 读取 {npz_file} 时发生异常: {e}")

    raise FileNotFoundError(
        "\n❌ 检索了所有提供的 npz 文件，均未找到这两张图的组合信息！\n"
        "请确认这两张图片是否属于这些 npz 对应的场景。"
    )


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
    # 将字号增大 2 个单位（原始基准 1.6 -> 3.6）
    FONT_SCALE = 2.6  # 固定基准字号（已增大）
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
    print(f"完成！图像已保存至: {save_path}")

def jamma_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jamma = JamMa(config=cfg,pretrained='/home/ly/9100pro/Reg/JamMa/jamma_log/jamma/version_10/checkpoints/epoch=21-auc@5=0.637-auc@10=0.770-auc@20=0.862.ckpt').eval().to(device)

    image0, scale0, mask0, prepad0 = read_megadepth_color(
        args["path0"], args.get("image_size", 832), 16, True
    )
    image1, scale1, mask1, prepad1 = read_megadepth_color(
        args["path1"], args.get("image_size", 832), 16, True
    )

    mask0 = F.interpolate(
        mask0[None, None].float(),
        scale_factor=0.125,
        mode="nearest",
        recompute_scale_factor=False
    )[0].bool()
    mask1 = F.interpolate(
        mask1[None, None].float(),
        scale_factor=0.125,
        mode="nearest",
        recompute_scale_factor=False
    )[0].bool()

    data = {
        "imagec_0": image0.to(device),
        "imagec_1": image1.to(device),
        "mask0": mask0.to(device),
        "mask1": mask1.to(device),
        "scale0": scale0[None].to(device),
        "scale1": scale1[None].to(device),
    }

    logger.info(f"Matching: {args['path0']} and {args['path1']}")
    jamma(data)
    logger.info("Finish Matching")

    if "m_bids" in data:
        b_mask = data["m_bids"] == 0
    else:
        conf_key = "mconf_f" if "mconf_f" in data else "mconf"
        b_mask = torch.ones_like(data[conf_key], dtype=torch.bool)

    if "mkpts0_f" in data and "mkpts1_f" in data:
        mkpts0 = data["mkpts0_f"][b_mask].detach().cpu().numpy()
        mkpts1 = data["mkpts1_f"][b_mask].detach().cpu().numpy()
        scores = (
            data["mconf_f"][b_mask].detach().cpu().numpy()
            if "mconf_f" in data else np.ones(len(mkpts0), dtype=float)
        )
        src_name = "mkpts*_f"
    elif "mkpts0_c" in data and "mkpts1_c" in data:
        mkpts0 = data["mkpts0_c"][b_mask].detach().cpu().numpy()
        mkpts1 = data["mkpts1_c"][b_mask].detach().cpu().numpy()
        scores = (
            data["mconf"][b_mask].detach().cpu().numpy()
            if "mconf" in data else np.ones(len(mkpts0), dtype=float)
        )
        src_name = "mkpts*_c"
    else:
        mkpts0 = np.empty((0, 2), dtype=np.float32)
        mkpts1 = np.empty((0, 2), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        src_name = "none"

    # 不做二次缩放，只做边界裁剪
    img0 = cv2.imread(args["path0"])
    img1 = cv2.imread(args["path1"])
    if img0 is None or img1 is None:
        raise RuntimeError("读取原图失败。")
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    if len(mkpts0) > 0:
        logger.info(f"JamMa output source: {src_name}")
        logger.info(f"mkpts0 x range: [{mkpts0[:, 0].min():.2f}, {mkpts0[:, 0].max():.2f}], image width={w0}")
        logger.info(f"mkpts0 y range: [{mkpts0[:, 1].min():.2f}, {mkpts0[:, 1].max():.2f}], image height={h0}")
        logger.info(f"mkpts1 x range: [{mkpts1[:, 0].min():.2f}, {mkpts1[:, 0].max():.2f}], image width={w1}")
        logger.info(f"mkpts1 y range: [{mkpts1[:, 1].min():.2f}, {mkpts1[:, 1].max():.2f}], image height={h1}")

    mkpts0[:, 0] = np.clip(mkpts0[:, 0], 0, w0 - 1)
    mkpts0[:, 1] = np.clip(mkpts0[:, 1], 0, h0 - 1)
    mkpts1[:, 0] = np.clip(mkpts1[:, 0], 0, w1 - 1)
    mkpts1[:, 1] = np.clip(mkpts1[:, 1], 0, h1 - 1)

    return mkpts0, mkpts1, scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path0",
        type=str,
        default="/home/ly/9100pro/Reg/JamMa0/data/megadepth/test/Undistorted_SfM/0015/images/3228510319_5901f7015b_o.jpg",
    )
    # 0015_0.1_0.3_000017_3228510319_5901f7015b_o__549917433_99b38abc41_o
    #0015_0.1_0.3_000203_270885805_cd14f37f72_o__549917433_99b38abc41_o
    #0015_0.1_0.3_000296_2189792388_50f1cddf2a_o__549917433_99b38abc41_o
    # 0015_0.3_0.5_000007_2150679996_fd2cd6339e_o__549917433_99b38abc41_o
    parser.add_argument(
        "--path1",
        type=str,
        default="/home/ly/9100pro/Reg/JamMa0/data/megadepth/test/Undistorted_SfM/0015/images/549917433_99b38abc41_o.jpg",
    )
    parser.add_argument("--save_path", type=str, default="output/1.jpg")
    parser.add_argument("--image_size", type=int, default=1152)
    parser.add_argument(
        "--npz_files",
        nargs="+",
        default=["/home/ly/9100pro/Reg/CasP/CasP/assets/megadepth_test_1500_scene_info/0015_0.1_0.3.npz", "/home/ly/9100pro/Reg/CasP/CasP/assets/megadepth_test_1500_scene_info/0015_0.3_0.5.npz"],
        help="用于搜索 GT 的 npz 文件列表",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args_namespace = parse_args()
    args = dict(vars(args_namespace))

    try:
        K0, K1, T_0to1 = find_gt_for_pair(args["npz_files"], args["path0"], args["path1"])
    except Exception:
        print("\n" + "=" * 60)
        print("❌ 发生错误，程序已停止！详细报错如下：")
        traceback.print_exc()
        print("=" * 60 + "\n")
        raise SystemExit(1)

    print("\n正在运行 JamMa 模型提取特征点...")
    points0, points1, scores = jamma_inference(args)

    print("正在生成 XoFTR 风格的彩色可视化...")
    draw_xoftr_style(
        args["path0"],
        args["path1"],
        points0,
        points1,
        K0,
        K1,
        T_0to1,
        args["save_path"],
    )