"""
Run JamMa on HPatches using the UniversalHpatchesBenchmark interface.

Place this file in the JamMa root and run from that directory:
  python test_universal_jamma.py

It loads the JamMa official weights and evaluates on HPatches dataset.
Modify DATA_PATH to point to your hpatches root.
"""
import sys
from pathlib import Path
import numpy as np

# ensure repo root is on path so we can import shared helpers
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_megadepth_jamma import load_jamma_matcher, run_matcher_jamma
from universal_hpatches_eval import UniversalHpatchesBenchmark

import torch


# Initialize JamMa matcher once
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading JamMa matcher to {device} (official weights)...")
matcher, read_fn = load_jamma_matcher(ckpt_path="official", device=device)


def match_jamma(img1_path: str, img2_path: str):
    # reuse JamMa runner with typical outdoor image_size=832, pad_to_square=True
    pts0, pts1, scores, t = run_matcher_jamma(
        matcher=matcher,
        read_megadepth_color_fn=read_fn,
        path0=Path(img1_path),
        path1=Path(img2_path),
        image_size=832,
        pad_to_square=True,
        use_amp=False,
    )
    return pts0, pts1


if __name__ == "__main__":
    # Update this to point to your HPatches root directory
    DATA_PATH = r"/home/ly/9100pro/Reg/hpatches"
    print(f"📂 Initializing Universal Benchmark with data from: {DATA_PATH}")
    benchmark = UniversalHpatchesBenchmark(DATA_PATH)
    print("🔥 Starting evaluation using JamMa...")
    results = benchmark.evaluate(match_jamma)
    print("\n" + "=" * 40)
    print("🏆 Universal Framework Results for JamMa:")
    for k, v in results.items():
        print(f"** {k}: {v:.2f} **")
    print("=" * 40)
