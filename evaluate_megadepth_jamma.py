import argparse
import json
import os
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.lib.format as fmt
import torch
import torch.nn.functional as F
from torch.nn import Module
from tqdm import tqdm

# Make JamMa project importable when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.utlis import JamMa as JamMaWrapper, cfg as jamma_cfg
from src.utils.dataset import read_megadepth_color


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def safe_load_npz(npz_path: Path) -> Dict:
    """Load scene npz robustly for both standard and legacy payloads.

    This mirrors the robust loader used in the CasP repository, but kept
    fully self-contained here for JamMa evaluation.
    """
    try:
        out: Dict[str, Any] = {}
        with zipfile.ZipFile(npz_path, "r") as zf:
            for name in zf.namelist():
                with zf.open(name) as f:
                    data = f.read()
                bio = BytesIO(data)
                try:
                    val = fmt.read_array(bio, allow_pickle=False)
                except Exception:
                    # Handle legacy pickled numpy arrays that may expect
                    # numpy._core to exist (NumPy 1.x layout).
                    shim_key = "numpy._core"
                    prev = sys.modules.get(shim_key)
                    injected = False
                    try:
                        if shim_key not in sys.modules:
                            sys.modules[shim_key] = np.core
                            injected = True
                        bio.seek(0)
                        try:
                            val = fmt.read_array(
                                bio,
                                allow_pickle=True,
                                pickle_kwargs={"encoding": "latin1"},
                            )
                        except TypeError:
                            bio.seek(0)
                            val = np.load(bio, allow_pickle=True)
                    finally:
                        if injected:
                            if prev is None:
                                del sys.modules[shim_key]
                            else:
                                sys.modules[shim_key] = prev
                key = name[:-4] if name.endswith(".npy") else name
                out[key] = val
        return out
    except Exception:
        obj = np.load(npz_path, allow_pickle=True)
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "files"):
            try:
                return {k: obj[k] for k in obj.files}
            finally:
                if hasattr(obj, "close"):
                    obj.close()
        if isinstance(obj, np.ndarray) and obj.shape == () and isinstance(obj.item(), dict):
            return obj.item()
        raise ValueError(f"Unsupported npz payload type: {type(obj)} from {npz_path}")


def load_megadepth_pairs(npz_root: Path, npz_list_path: Path, min_overlap_score: float) -> List[Dict]:
    """Load MegaDepth scene pairs and their ground-truth poses.

    Compatible with the LoFTR / CasP MegaDepth 1500 evaluation protocol.
    """
    pairs: List[Dict] = []
    with open(npz_list_path, "r", encoding="utf-8") as f:
        scene_names = [line.split()[0].strip() for line in f if line.strip()]

    for scene_name in scene_names:
        npz_name = scene_name if scene_name.endswith(".npz") else f"{scene_name}.npz"
        scene = safe_load_npz(npz_root / npz_name)

        for pair_info in scene["pair_infos"]:
            (idx0, idx1), overlap_score, _ = pair_info
            if overlap_score < min_overlap_score:
                continue

            T0 = scene["poses"][idx0]
            T1 = scene["poses"][idx1]
            T_0to1 = np.matmul(T1, np.linalg.inv(T0))

            pairs.append(
                {
                    "scene": npz_name,
                    "im0": scene["image_paths"][idx0],
                    "im1": scene["image_paths"][idx1],
                    "K0": scene["intrinsics"][idx0].astype(np.float32),
                    "K1": scene["intrinsics"][idx1].astype(np.float32),
                    "T_0to1": T_0to1.astype(np.float32),
                }
            )
    return pairs


def relative_pose_error(
    T_0to1: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    ignore_gt_t_thr: float = 0.0,
) -> Tuple[float, float]:
    """COPY of src.utils.metrics.relative_pose_error for standalone use.

    Computes angular errors between predicted and ground-truth pose.
    """
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0.0

    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    R_err = np.rad2deg(np.abs(np.arccos(cos)))
    return float(t_err), float(R_err)


def estimate_pose_ransac(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    thresh: float,
    conf: float = 0.99999,
):
    """COPY of src.utils.metrics.estimate_pose for standalone use.

    Standard 5-point RANSAC on normalized points.
    """
    if len(kpts0) < 5:
        return None

    # normalize keypoints
    kpts0_n = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1_n = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    E, mask = cv2.findEssentialMat(
        kpts0_n,
        kpts1_n,
        np.eye(3),
        threshold=ransac_thr,
        prob=conf,
        method=cv2.RANSAC,
    )
    if E is None:
        return None

    best_num_inliers = 0
    best = None
    for _E in np.split(E, int(len(E) / 3)):
        n, R, t, _ = cv2.recoverPose(_E, kpts0_n, kpts1_n, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n
    return best


def estimate_pose_lo_ransac(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    thresh: float,
    conf: float = 0.99999,
):
    """Standalone LO-RANSAC pose estimation mirroring estimate_lo_pose.

    Requires poselib and the same Camera/Pose wrappers as JamMa, but is
    implemented locally to keep this file independent.
    """
    from src.utils.warppers import Camera, Pose  # type: ignore
    import poselib

    camera0, camera1 = Camera.from_calibration_matrix(K0).float(), Camera.from_calibration_matrix(K1).float()
    pts0, pts1 = kpts0, kpts1

    M, info = poselib.estimate_relative_pose(
        pts0,
        pts1,
        camera0.to_cameradict(),
        camera1.to_cameradict(),
        {
            "max_epipolar_error": thresh,
        },
    )
    success = M is not None and (((M.t != [0.0, 0.0, 0.0]).all()) or ((M.q != [1.0, 0.0, 0.0, 0.0]).all()))
    if success:
        M = Pose.from_Rt(torch.tensor(M.R), torch.tensor(M.t))
    else:
        M = Pose.from_4x4mat(torch.eye(4).numpy())

    estimation = {
        "success": success,
        "M_0to1": M,
        "inliers": torch.tensor(info.pop("inliers")),
        **info,
    }
    return estimation


def error_auc(errors: List[float], thresholds: List[int]) -> Dict[str, float]:
    """COPY of src.utils.metrics.error_auc for standalone use.

    Note: the JamMa implementation ignores the thresholds argument values
    and always uses [5, 10, 20] internally; we keep the same behavior for
    numerical parity.
    """
    errors_sorted = [0.0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors_sorted)))

    aucs: List[float] = []
    # JamMa hard-codes [5, 10, 20]
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors_sorted, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors_sorted[:last_index] + [thr]
        aucs.append(float(np.trapz(y, x) / thr))
    return {f"auc@{t}": a for t, a in zip(thresholds, aucs)}


def load_jamma_matcher(
    ckpt_path: str,
    device: str = "cpu",
) -> Tuple[Module, Any]:
    """Load JamMa matcher and MegaDepth color reader.

    ckpt_path can be "official" (download from GitHub release) or a local
    path to jamma.ckpt.
    """
    pretrained = "official" if ckpt_path in ("", "official") else ckpt_path
    net = JamMaWrapper(config=jamma_cfg, pretrained=pretrained)
    net = net.eval().to(device)
    return net, read_megadepth_color


def run_matcher_jamma(
    matcher: Module,
    read_megadepth_color_fn,
    path0: Path,
    path1: Path,
    image_size: int,
    pad_to_square: bool,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run JamMa on an image pair and return keypoints & scores.

    This follows JamMa/demo/demo.py and its internal matching pipeline.
    """
    # JamMa expects MegaDepth-style color images with resize & padding.
    image0, scale0, mask0, _ = read_megadepth_color_fn(
        str(path0), resize=image_size, df=16, padding=pad_to_square
    )
    image1, scale1, mask1, _ = read_megadepth_color_fn(
        str(path1), resize=image_size, df=16, padding=pad_to_square
    )

    device = next(matcher.parameters()).device

    if mask0 is not None:
        mask0_down = F.interpolate(
            mask0[None, None].float(),
            scale_factor=0.125,
            mode="nearest",
            recompute_scale_factor=False,
        )[0].bool()
        mask1_down = F.interpolate(
            mask1[None, None].float(),
            scale_factor=0.125,
            mode="nearest",
            recompute_scale_factor=False,
        )[0].bool()
    else:
        mask0_down = mask1_down = None

    # JamMa expects keys named imagec_0 / imagec_1 etc.
    data: Dict[str, Any] = {
        "imagec_0": image0.to(device),
        "imagec_1": image1.to(device),
        # scale is [w/w_new, h/h_new]; add batch dimension.
        "scale0": scale0[None].to(device),
        "scale1": scale1[None].to(device),
    }
    if mask0_down is not None:
        data["mask0"] = mask0_down.to(device)
        data["mask1"] = mask1_down.to(device)

    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
            matcher(data)
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        match_time_sec = time.perf_counter() - t0

    # JamMa writes final matched keypoints and confidences into the batch dict.
    points0 = data["mkpts0_f"].detach().cpu().numpy()
    points1 = data["mkpts1_f"].detach().cpu().numpy()
    scores = data["mconf_f"].detach().cpu().numpy()
    return points0, points1, scores, float(match_time_sec)


def evaluate_pose_auc(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = resolve_path(args.data_root)
    npz_root = resolve_path(args.npz_root)
    npz_list = resolve_path(args.npz_list)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if not npz_root.exists():
        raise FileNotFoundError(f"npz_root not found: {npz_root}")
    if not npz_list.exists():
        raise FileNotFoundError(f"npz_list not found: {npz_list}")

    ckpt_desc = args.ckpt if args.ckpt else "official"
    matcher, jamma_read_color = load_jamma_matcher(
        ckpt_path=ckpt_desc,
        device=str(device),
    )

    pairs = load_megadepth_pairs(npz_root, npz_list, args.min_overlap_score)
    target_pairs = pairs if args.max_pairs <= 0 else pairs[: args.max_pairs]

    pose_errors: List[float] = []  # will store max(R_err, t_err) per pair (and per repeat)
    match_times_sec: List[float] = []
    failed = 0
    current_image_size = args.image_size
    current_pad_to_square = args.pad_to_square
    current_use_amp = args.use_amp

    def _next_image_size(size: int) -> int:
        # Keep sizes divisible by 32 because JamMa expects factor alignment.
        candidates = [1152, 1024, 960, 896, 832, 768, 704, 640, 576, 512]
        for c in candidates:
            if c < size and c >= args.min_image_size:
                return c
        return max(args.min_image_size, (size // 64) * 64)

    for idx, pair in enumerate(tqdm(target_pairs, desc="Evaluating JamMa on MegaDepth")):
        path0 = data_root / pair["im0"]
        path1 = data_root / pair["im1"]

        # Skip pairs with missing files to avoid crashing on cv2.imread or PIL
        if not path0.exists() or not path1.exists():
            print(f"[WARN] Missing image file(s), skipping pair: {path0} , {path1}")
            failed += 1
            continue

        # Retry with smaller image size if coarse matching runs OOM.
        while True:
            try:
                points0, points1, _, match_time_sec = run_matcher_jamma(
                    matcher=matcher,
                    read_megadepth_color_fn=jamma_read_color,
                    path0=path0,
                    path1=path1,
                    image_size=current_image_size,
                    pad_to_square=current_pad_to_square,
                    use_amp=current_use_amp,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if not args.auto_reduce_on_oom:
                    raise
                next_size = _next_image_size(current_image_size)
                if next_size < current_image_size:
                    print(
                        f"[WARN] CUDA OOM at image_size={current_image_size}. "
                        f"Retrying with image_size={next_size}."
                    )
                    current_image_size = next_size
                elif current_pad_to_square:
                    print("[WARN] CUDA OOM at minimum size. Retrying with --pad_to_square=False.")
                    current_pad_to_square = False
                elif not current_use_amp:
                    print("[WARN] CUDA OOM at minimum size. Retrying with AMP enabled.")
                    current_use_amp = True
                else:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        in_timing_window = idx >= args.warmup_pairs and (
            args.timing_pairs <= 0 or idx < args.warmup_pairs + args.timing_pairs
        )
        if in_timing_window:
            match_times_sec.append(match_time_sec)

        # Pose estimation logic mirroring compute_pose_errors (single repeat)
        K0 = pair["K0"]
        K1 = pair["K1"]
        T_0to1 = pair["T_0to1"]

        if args.ransac_method == "RANSAC":
            ret = estimate_pose_ransac(
                points0,
                points1,
                K0,
                K1,
                thresh=args.ransac_thr,
                conf=args.ransac_conf,
            )
            if ret is None:
                pose_errors.append(float("inf"))
                failed += 1
                continue
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1, R, t)
            pose_errors.append(float(max(t_err, R_err)))

        elif args.ransac_method == "LO_RANSAC":
            est = estimate_pose_lo_ransac(
                points0,
                points1,
                K0,
                K1,
                thresh=args.ransac_thr,
                conf=args.ransac_conf,
            )
            if not est["success"]:
                # JamMa uses fixed large errors (90) on failure; here we
                # approximate with +inf so they do not contribute within the
                # typical thresholds.
                pose_errors.append(float("inf"))
                failed += 1
                continue
            M = est["M_0to1"]
            t_err, R_err = relative_pose_error(T_0to1, M.R, M.t)
            pose_errors.append(float(max(t_err, R_err)))

        else:
            raise ValueError(f"Unknown RANSAC method: {args.ransac_method}")

    auc = error_auc(pose_errors, args.thresholds)
    auc_percent = {k: float(v * 100.0) for k, v in auc.items()}

    times_np = np.array(match_times_sec, dtype=np.float64)
    timing = {
        "warmup_pairs": int(args.warmup_pairs),
        "timing_pairs_requested": int(args.timing_pairs),
        "num_timed_pairs": int(len(times_np)),
        "match_time_mean_ms": float(times_np.mean() * 1e3) if len(times_np) > 0 else 0.0,
        "match_time_median_ms": float(np.median(times_np) * 1e3) if len(times_np) > 0 else 0.0,
        "match_time_p95_ms": float(np.percentile(times_np, 95) * 1e3) if len(times_np) > 0 else 0.0,
        "match_time_total_s": float(times_np.sum()) if len(times_np) > 0 else 0.0,
    }

    return {
        "method": "jamma_outdoor",
        "ckpt": str(ckpt_desc),
        "data_root": str(data_root),
        "npz_root": str(npz_root),
        "npz_list": str(npz_list),
        "image_size": args.image_size,
        "effective_image_size": current_image_size,
        "effective_pad_to_square": current_pad_to_square,
        "effective_use_amp": current_use_amp,
        "matching_threshold": None,
        "min_overlap_score": args.min_overlap_score,
        "ransac_thr": args.ransac_thr,
        "ransac_conf": args.ransac_conf,
        "thresholds": args.thresholds,
        "num_pairs": len(target_pairs),
        "num_failed": failed,
        "failure_rate": (failed / len(target_pairs)) if target_pairs else 0.0,
        "auc": auc,
        "auc_percent": auc_percent,
        "timing": timing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate JamMa on MegaDepth pose AUC@5/10/20",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=str, default="", help="JamMa checkpoint path; empty for 'official' release")
    parser.add_argument("--data_root", type=str, default="data/megadepth/test")
    parser.add_argument(
        "--npz_root",
        type=str,
        default="assets/megadepth_test_1500_scene_info",
        help="Directory containing MegaDepth scene .npz files",
    )
    parser.add_argument(
        "--npz_list",
        type=str,
        default="assets/megadepth_test_1500_scene_info/megadepth_test_1500.txt",
        help="Text file listing scenes to evaluate",
    )
    parser.add_argument("--image_size", type=int, default=832)
    parser.add_argument("--pad_to_square", action="store_true", default=True)
    parser.add_argument("--no_pad_to_square", dest="pad_to_square", action="store_false")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_use_amp", dest="use_amp", action="store_false")
    parser.add_argument("--auto_reduce_on_oom", action="store_true", default=False)
    parser.add_argument("--no_auto_reduce_on_oom", dest="auto_reduce_on_oom", action="store_false")
    parser.add_argument("--min_image_size", type=int, default=512)
    parser.add_argument("--min_overlap_score", type=float, default=0.0)
    parser.add_argument("--ransac_thr", type=float, default=0.5)
    parser.add_argument("--ransac_conf", type=float, default=0.99999)
    parser.add_argument(
        "--ransac_method",
        type=str,
        default="RANSAC",
        choices=["RANSAC", "LO_RANSAC"],
        help="RANSAC variant to use for pose estimation (LO_RANSAC will use OpenCV USAC LO if available)",
    )
    parser.add_argument("--thresholds", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--warmup_pairs", type=int, default=50)
    parser.add_argument("--timing_pairs", type=int, default=200)
    parser.add_argument("--save_json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_pose_auc(args)

    print(
        "AUC results (percent): "
        + ", ".join([f"{k}={v:.2f}" for k, v in summary["auc_percent"].items()])
    )
    print(
        f"Pairs={summary['num_pairs']} Failed={summary['num_failed']} "
        f"FailureRate={summary['failure_rate']:.4f}"
    )
    print(
        "Matching time: "
        f"mean={summary['timing']['match_time_mean_ms']:.2f} ms, "
        f"median={summary['timing']['match_time_median_ms']:.2f} ms, "
        f"p95={summary['timing']['match_time_p95_ms']:.2f} ms, "
        f"total={summary['timing']['match_time_total_s']:.2f} s"
    )

    if args.save_json:
        save_path = resolve_path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {save_path}")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
