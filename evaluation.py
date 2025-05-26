#!/usr/bin/env python3
"""texture_quality_eval.py – **lightweight** comparison for texture quilting

Compares a synthesized quilt against its exemplar without downloading heavy
CNN weights.  It samples *K* random, non‑overlapping patches from both
images (after optional tiling / resizing) and reports Structural Similarity
Index (SSIM) statistics plus mean‑squared error (MSE).

Usage
-----
```bash
python texture_quality_eval.py \
        --ref   texture.jpg \
        --synth big_quilt.png \
        --patch 40         # patch size (KxK)
        --num   250        # how many patches to sample
        --mode  tile       # alignment: tile | resize | crop
```
SSIM relies only on OpenCV (or scikit‑image fallback) – **no VGG download**.

Why no LPIPS / VGG download?
---------------------------
LPIPS extracts VGG‑16 features; first use triggers a ~519 MB weight download.
If your machine lacks internet or you just want quick feedback, SSIM gives a
reasonable correlate of perceptual quality at a fraction of the cost.

Feel free to extend the `metrics` list if you later reinstall lpips.
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# ----------------------- SSIM impl -----------------------------------------
try:
    # OpenCV contrib (fast C++): quality module present?
    _ = cv2.quality.QualitySSIM_compute
    def _ssim(a: np.ndarray, b: np.ndarray) -> float:  # type: ignore
        return float(cv2.quality.QualitySSIM_compute(a, b)[0])
except (AttributeError, cv2.error):
    # Fallback to skimage (pure python, slower but ubiquitous)
    from skimage.metrics import structural_similarity as _sk_ssim  # type: ignore
    def _ssim(a: np.ndarray, b: np.ndarray) -> float:
        return float(_sk_ssim(a, b, channel_axis=2))

# ---------------------- Alignment helpers ----------------------------------

def _tile_to(shape: Tuple[int, int], img: np.ndarray) -> np.ndarray:
    h, w = shape
    th, tw, _ = img.shape
    rep_y = h // th + 1
    rep_x = w // tw + 1
    tiled = np.tile(img, (rep_y, rep_x, 1))
    return tiled[:h, :w]

def _align(ref: np.ndarray, synth: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "tile":
        ref_aligned = _tile_to(synth.shape[:2], ref)
        return ref_aligned, synth
    if mode == "resize":
        ref_aligned = cv2.resize(ref, synth.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        return ref_aligned, synth
    if mode == "crop":
        h = min(ref.shape[0], synth.shape[0])
        w = min(ref.shape[1], synth.shape[1])
        return ref[:h, :w], synth[:h, :w]
    raise ValueError(f"Unknown mode {mode}")

# ----------------------- Main routine --------------------------------------

def sample_patches(ref: np.ndarray, synth: np.ndarray, k: int, n: int, rng: random.Random):
    h, w = ref.shape[:2]
    if h < k or w < k:
        raise ValueError("Patch size larger than aligned image")
    scores = []
    mses   = []
    for _ in range(n):
        y = rng.randint(0, h - k)
        x = rng.randint(0, w - k)
        r_patch = ref[y:y+k, x:x+k]
        s_patch = synth[y:y+k, x:x+k]
        scores.append(_ssim(r_patch, s_patch))
        mses.append(float(np.mean((r_patch.astype(np.float32) - s_patch.astype(np.float32)) ** 2)))
    return np.array(scores), np.array(mses)


def main():
    p = argparse.ArgumentParser(description="Random‑patch SSIM evaluation for texture quilting.")
    p.add_argument("--ref",   type=Path, required=True, help="Exemplar texture image")
    p.add_argument("--synth", type=Path, required=True, help="Synthesized/quilted image")
    p.add_argument("--patch", type=int, default=40, help="Patch (block) size")
    p.add_argument("--num",   type=int, default=250, help="Number of random patches")
    p.add_argument("--mode",  choices=["tile", "resize", "crop"], default="tile", help="How to align differently‑sized inputs")
    p.add_argument("--seed",  type=int, default=0, help="RNG seed for reproducibility")
    args = p.parse_args()

    ref   = cv2.imread(str(args.ref), cv2.IMREAD_COLOR)
    synth = cv2.imread(str(args.synth), cv2.IMREAD_COLOR)
    if ref is None or synth is None:
        raise SystemExit("Error: could not load one of the images.")

    ref, synth = _align(ref, synth, args.mode)

    rng = random.Random(args.seed)
    ssim_scores, mses = sample_patches(ref, synth, args.patch, args.num, rng)

    def _stats(arr: np.ndarray):
        return {
            "mean":   arr.mean(),
            "median": np.median(arr),
            "min":    arr.min(),
            "max":    arr.max(),
            "std":    arr.std(ddof=1),
        }

    print("\n=== Patchwise statistics (N =", args.num, ") ===")
    s = _stats(ssim_scores)
    print("SSIM : mean {mean:.3f}, median {median:.3f}, min {min:.3f}, max {max:.3f}, std {std:.3f}".format(**s))
    s = _stats(mses)
    print("MSE  : mean {mean:.1f}, median {median:.1f}, min {min:.1f}, max {max:.1f}, std {std:.1f}".format(**s))


if __name__ == "__main__":
    main()
