import argparse
import json
import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

os.sys.path.append("../")

from utlis import JamMa, cfg
from src.utils.dataset import read_megadepth_color


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_VISIBLE_DIR = "/home/ly/9100pro/Reg/RoadScene/crop_LR_visible"
DEFAULT_WARPED_IR_DIR = "/home/ly/9100pro/Reg/RoadScene/homography_ir"
DEFAULT_OUTPUT_DIR = "/home/ly/9100pro/Reg/RoadScene/jamma_registered_ir"
DEFAULT_CKPT = "/home/ly/9100pro/Reg/JamMa/jamma_log/jamma/version_10/checkpoints/epoch=21-auc@5=0.637-auc@10=0.770-auc@20=0.862.ckpt"


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def build_pairs(visible_dir, warped_ir_dir):
    visible_paths = {p.name: p for p in list_images(visible_dir)}
    ir_paths = {p.name: p for p in list_images(warped_ir_dir)}

    common_names = sorted(set(visible_paths) & set(ir_paths))
    pairs = [(visible_paths[name], ir_paths[name]) for name in common_names]

    missing_in_ir = sorted(set(visible_paths) - set(ir_paths))
    missing_in_visible = sorted(set(ir_paths) - set(visible_paths))
    return pairs, missing_in_ir, missing_in_visible


def load_matcher(ckpt_path):
    if ckpt_path is None or ckpt_path == "":
        pretrained = "official"
    else:
        pretrained = ckpt_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = JamMa(config=cfg, pretrained=pretrained).eval().to(device)
    return matcher


def jamma_match_pair(jamma, vis_path, ir_path, image_size, pad_to_square=True, use_amp=False):
    device = next(jamma.parameters()).device

    image0, scale0, mask0, _ = read_megadepth_color(str(vis_path), resize=image_size, df=16, padding=pad_to_square)
    image1, scale1, mask1, _ = read_megadepth_color(str(ir_path), resize=image_size, df=16, padding=pad_to_square)

    if mask0 is not None:
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
        "scale0": scale0[None].to(device),
        "scale1": scale1[None].to(device),
    }
    if mask0 is not None:
        data["mask0"] = mask0.to(device)
        data["mask1"] = mask1.to(device)

    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        with torch.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
            jamma(data)
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    points0 = data["mkpts0_f"].detach().cpu().numpy()
    points1 = data["mkpts1_f"].detach().cpu().numpy()
    scores = data["mconf_f"].detach().cpu().numpy()
    return points0, points1, scores


def register_one_pair(jamma, vis_path, ir_path, image_size, ransac_thresh, ransac_conf, use_amp=False):
    points0, points1, scores = jamma_match_pair(
        jamma=jamma,
        vis_path=vis_path,
        ir_path=ir_path,
        image_size=image_size,
        pad_to_square=True,
        use_amp=use_amp,
    )

    if len(points0) < 4 or len(points1) < 4:
        return {
            "ok": False,
            "reason": "not_enough_matches",
            "num_matches": int(min(len(points0), len(points1))),
        }

    H_ir_to_vis, inliers = cv2.findHomography(
        points1,
        points0,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=ransac_conf,
    )

    if H_ir_to_vis is None:
        return {
            "ok": False,
            "reason": "homography_failed",
            "num_matches": int(min(len(points0), len(points1))),
        }

    vis_img = cv2.imread(str(vis_path), cv2.IMREAD_UNCHANGED)
    ir_img = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)
    if vis_img is None or ir_img is None:
        return {
            "ok": False,
            "reason": "image_read_failed",
            "num_matches": int(min(len(points0), len(points1))),
        }

    h_vis, w_vis = vis_img.shape[:2]
    aligned_ir = cv2.warpPerspective(
        ir_img,
        H_ir_to_vis,
        (w_vis, h_vis),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    inlier_count = int(inliers.sum()) if inliers is not None else 0
    return {
        "ok": True,
        "aligned_ir": aligned_ir,
        "H_ir_to_vis": H_ir_to_vis,
        "num_matches": int(min(len(points0), len(points1))),
        "num_inliers": inlier_count,
        "inlier_ratio": float(inlier_count / max(1, len(points0))),
        "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Batch registration for JamMa on visible and warped IR image pairs")
    parser.add_argument("--visible_dir", type=str, default=DEFAULT_VISIBLE_DIR)
    parser.add_argument("--warped_ir_dir", type=str, default=DEFAULT_WARPED_IR_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Path to jamma.ckpt; empty string uses official weights")
    parser.add_argument("--image_size", type=int, default=832)
    parser.add_argument("--ransac_thresh", type=float, default=3.0)
    parser.add_argument("--ransac_conf", type=float, default=0.99999)
    parser.add_argument("--max_pairs", type=int, default=-1)
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    pairs, missing_in_ir, missing_in_visible = build_pairs(args.visible_dir, args.warped_ir_dir)
    if not pairs:
        raise RuntimeError("No matched filenames between visible_dir and warped_ir_dir.")

    if args.max_pairs >= 0:
        pairs = pairs[: args.max_pairs]

    jamma = load_matcher(args.ckpt)

    report = {
        "visible_dir": args.visible_dir,
        "warped_ir_dir": args.warped_ir_dir,
        "output_dir": str(output_dir),
        "ckpt": args.ckpt,
        "image_size": args.image_size,
        "total_pairs": len(pairs),
        "success": 0,
        "failed": 0,
        "missing_in_ir": missing_in_ir,
        "missing_in_visible": missing_in_visible,
        "items": {},
    }

    for vis_path, ir_path in tqdm(pairs, desc="Registering"):
        item_key = vis_path.name
        try:
            result = register_one_pair(
                jamma=jamma,
                vis_path=vis_path,
                ir_path=ir_path,
                image_size=args.image_size,
                ransac_thresh=args.ransac_thresh,
                ransac_conf=args.ransac_conf,
                use_amp=args.amp,
            )
        except Exception as exc:
            report["failed"] += 1
            report["items"][item_key] = {
                "ok": False,
                "reason": f"inference_failed: {exc}",
                "source_visible": str(vis_path),
                "source_warped_ir": str(ir_path),
            }
            continue

        if not result.get("ok", False):
            report["failed"] += 1
            report["items"][item_key] = {
                "ok": False,
                "reason": result.get("reason", "unknown"),
                "num_matches": int(result.get("num_matches", 0)),
                "source_visible": str(vis_path),
                "source_warped_ir": str(ir_path),
            }
            continue

        out_path = output_images_dir / vis_path.name
        cv2.imwrite(str(out_path), result["aligned_ir"])

        report["success"] += 1
        report["items"][item_key] = {
            "ok": True,
            "source_visible": str(vis_path),
            "source_warped_ir": str(ir_path),
            "output_registered_ir": str(out_path),
            "num_matches": result["num_matches"],
            "num_inliers": result["num_inliers"],
            "inlier_ratio": result["inlier_ratio"],
            "mean_score": result["mean_score"],
            "H_ir_to_vis": result["H_ir_to_vis"].tolist(),
        }

    report_path = output_dir / "registration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Done. Total: {report['total_pairs']}, Success: {report['success']}, Failed: {report['failed']}")
    print(f"Registered images: {output_images_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
