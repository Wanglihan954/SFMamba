import os
os.sys.path.append("../")  # Add the project directory
from pathlib import Path
import torch
from utlis import JamMa, cfg
from src.utils.dataset import read_megadepth_color
import argparse
from loguru import logger
import torch.nn.functional as F
from src.utils.plotting import make_confidence_figure, make_evaluation_figure_wheel
from numpy import ndarray
from typing import Any, Dict, Optional, Tuple, Type,Union,List
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import cv2


def _to_numpy(array_like):
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)
def load_image(
    path: str,
    mode: str = "gray",
    size: Optional[Union[int, Tuple[int, int]]] = None,
    factor: int = 1,
    pad_to_square: bool = False,
) -> Tuple[ndarray, Optional[ndarray], ndarray]:
    if mode == "gray":
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif mode == "color":
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Supported modes are `gray` and `color`.")

    h, w = image.shape[:2]
    if size is None:
        new_w, new_h = w, h
    elif isinstance(size, int):
        k = size / max(w, h)
        new_w, new_h = round(w * k), round(h * k)
    else:
        new_w, new_h = size
    new_w, new_h = new_w // factor * factor, new_h // factor * factor
    image = cv2.resize(image, (new_w, new_h))
    scale = np.array([w / new_w, h / new_h])

    mask = None
    if pad_to_square:
        length = max(new_w, new_h)
        pad_size = (length, length) if mode == "gray" else (length, length, 3)
        pad_image = np.zeros(pad_size, dtype=image.dtype)
        pad_image[:new_h, :new_w] = image
        image = pad_image
        mask = np.zeros((length, length), dtype=bool)
        mask[:new_h, :new_w] = True
    return image, mask, scale
def _make_matching_figure(
    image0: ndarray,
    image1: ndarray,
    points0: ndarray,
    points1: ndarray,
    colors: ndarray,
    enable_line: bool = True,
    dpi: int = 75,
    pad: float = 1.0,
    text: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    for ax, image in zip(axes, [image0, image1]):
        ax.imshow(image)
        ax.axis("off")
    plt.tight_layout(pad=pad)

    if len(points0) != 0 and len(points0) == len(points1):
        fig.canvas.draw()
        if enable_line:
            fig_points0 = axes[0].transData.transform(points0)
            fig_points1 = axes[1].transData.transform(points1)
            fig_points0 = fig.transFigure.inverted().transform(fig_points0)
            fig_points1 = fig.transFigure.inverted().transform(fig_points1)
            for i in range(len(points0)):
                x = fig_points0[i, 0], fig_points1[i, 0]
                y = fig_points0[i, 1], fig_points1[i, 1]
                line = Line2D(
                    x, y, c=colors[i], lw=2, transform=fig.transFigure
                )
                fig.add_artist(line)

        axes[0].autoscale(enable=False)
        axes[1].autoscale(enable=False)
        axes[0].scatter(points0[:, 0], points0[:, 1], c=colors[:, :3], s=4)
        axes[1].scatter(points1[:, 0], points1[:, 1], c=colors[:, :3], s=4)

    if text is not None:
        text = "\n".join(text)
        color = "k" if image0[:100, :200].mean() > 200 else "w"
        fig.text(
            0.01,
            0.99,
            text,
            c=color,
            va="top",
            ha="left",
            fontsize=15,
            transform=axes[0].transAxes,
        )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        return None
    else:
        return fig


def _make_colormap(
    errors: ndarray, threshold: float, alpha: float = 1.0
) -> ndarray:
    x = 1.0 - (errors / (threshold * 2.0)).clip(min=0.0, max=1.0)
    colormap = np.stack(
        [2.0 - x * 2.0, x * 2.0, np.zeros_like(x), np.ones_like(x) * alpha],
        axis=-1,
    ).clip(min=0.0, max=1.0)
    return colormap


def make_matching_figure(
    path0: str,
    path1: str,
    points0: ndarray,
    points1: ndarray,
    errors: ndarray,
    threshold: float,
    text: Optional[List[str]] = None,
    **kwargs,
) -> Optional[Figure]:
    points0 = _to_numpy(points0)
    points1 = _to_numpy(points1)
    errors = _to_numpy(errors)

    image0, _, _ = load_image(path0, mode="color")
    image1, _, _ = load_image(path1, mode="color")
    colors = _make_colormap(errors, threshold, alpha=0.1)
    text = [f"#matches: {len(points0)}"] if text is None else text
    figure = _make_matching_figure(
        image0, image1, points0, points1, colors, text=text, **kwargs
    )
    return figure

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching with JamMa',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--image1', type=str, default='/home/ly/9100pro/Reg/CasP/CasP/data/megadepth/test/Undistorted_SfM/0015/images/549917433_99b38abc41_o.jpg',
        help='Path to the source image')
    parser.add_argument(
        '--image2', type=str, default='/home/ly/9100pro/Reg/CasP/CasP/data/megadepth/test/Undistorted_SfM/0015/images/2248320068_7553a48263_o.jpg',
        help='Path to the target image')
    parser.add_argument(
        '--output_dir', type=str, default='/home/ly/9100pro/Reg/JamMa/assets/figs/',
        help='Path of the outputs')
    parser.add_argument("--save_path", default="/home/ly/9100pro/Reg/JamMa/assets/figs/jamma_matches.png", type=str)

    opt = parser.parse_args()
    Path(opt.output_dir).mkdir(exist_ok=True, parents=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jamma = JamMa(config=cfg).eval().to(device)

    image0, scale0, mask0, prepad_size0 = read_megadepth_color(opt.image1, 832, 16, True)
    image1, scale1, mask1, prepad_size1 = read_megadepth_color(opt.image2, 832, 16, True)
    mask0 = F.interpolate(mask0[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
    mask1 = F.interpolate(mask1[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
    data = {
        'imagec_0': image0.to(device),
        'imagec_1': image1.to(device),
        'mask0': mask0.to(device),
        'mask1': mask1.to(device),
    }

    logger.info(f"Matching: {opt.image1} and {opt.image2}")
    jamma(data)
    logger.info(f"Finish Matching, Visualizing")
    topk = 10000
    num = len(data['mconf_f']) if len(data['mconf_f']) < topk else topk
    idx = torch.topk(data['mconf_f'], num, 0).indices
    kpts0 = data['mkpts0_f'][idx].detach().cpu().numpy()
    kpts1 = data['mkpts1_f'][idx].detach().cpu().numpy()
    certainty = data['mconf_f'][idx].detach().cpu().numpy()
    # read_megadepth_color returns scale as [w / w_new, h / h_new].
    # Back-project keypoints from resized space to original image space with (x, y) order.
    scale0_np = scale0.cpu().numpy()
    scale1_np = scale1.cpu().numpy()
    kpts0 = kpts0 * scale0_np[None, :]
    kpts1 = kpts1 * scale1_np[None, :]
    args, _ = parser.parse_known_args()

    # Keep points inside image bounds to avoid tiny plotting artifacts at borders.
    img0_h, img0_w = load_image(opt.image1, mode="color")[0].shape[:2]
    img1_h, img1_w = load_image(opt.image2, mode="color")[0].shape[:2]
    kpts0[:, 0] = np.clip(kpts0[:, 0], 0, img0_w - 1)
    kpts0[:, 1] = np.clip(kpts0[:, 1], 0, img0_h - 1)
    kpts1[:, 0] = np.clip(kpts1[:, 0], 0, img1_w - 1)
    kpts1[:, 1] = np.clip(kpts1[:, 1], 0, img1_h - 1)
    logger.info(f"Done")
    errors = 1-certainty
    text = ['Jamma', f"#matches: {len(kpts1)}"]
    
    make_matching_figure(
        args.image1,
        args.image2,
        kpts0,
        kpts1,
        errors,
        0.5,
        dpi=300,
        save_path=args.save_path,
    )
