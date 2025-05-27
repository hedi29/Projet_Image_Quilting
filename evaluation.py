#!/usr/bin/env python3
"""Lightweight comparison for texture quilting.

Compares a synthesized quilt against its exemplar using SSIM and MSE
on random patches, avoiding heavy CNN downloads.

Usage:
  python texture_quality_eval.py --ref texture.jpg --synth big_quilt.png \
                                 --patch 40 --num 250 --mode tile

SSIM uses OpenCV or scikit-image as a fallback.
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# --- SSIM implementation ---
try:
    # Prefer OpenCV contrib's fast C++ QualitySSIM
    _ = cv2.quality.QualitySSIM_compute # Check if available
    def _ssim(a: np.ndarray, b: np.ndarray) -> float:  # type: ignore
        return float(cv2.quality.QualitySSIM_compute(a, b)[0])
except (AttributeError, cv2.error):
    # Fallback to scikit-image (slower, pure Python)
    from skimage.metrics import structural_similarity as _sk_ssim  # type: ignore
    def _ssim(a: np.ndarray, b: np.ndarray) -> float:
        return float(_sk_ssim(a, b, channel_axis=2))

# --- Alignment helpers ---

def _tile_to(shape: Tuple[int, int], img: np.ndarray) -> np.ndarray:
    "Tiles `img` to match `shape`."
    h, w = shape
    th, tw, _ = img.shape
    rep_y = h // th + 1
    rep_x = w // tw + 1
    tiled = np.tile(img, (rep_y, rep_x, 1))
    return tiled[:h, :w]

def _align(ref: np.ndarray, synth: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    "Aligns `ref` and `synth` images based on `mode`."
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
    raise ValueError(f"Unknown alignment mode: {mode}")

# --- Main routine ---

def sample_patches(ref: np.ndarray, synth: np.ndarray, k: int, n: int, rng: random.Random):
    "Samples `n` random `k`x`k` patches from aligned `ref` and `synth` images."
    h, w = ref.shape[:2]
    if h < k or w < k:
        raise ValueError("Patch size larger than aligned image.")
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
    p = argparse.ArgumentParser(description="Random-patch SSIM/MSE evaluation for texture quilting.")
    p.add_argument("--ref",   type=Path, required=True, help="Reference exemplar texture image.")
    p.add_argument("--synth", type=Path, required=True, help="Synthesized (quilted) image.")
    p.add_argument("--patch", type=int, default=40, help="Patch size (KxK pixels) for comparison.")
    p.add_argument("--num",   type=int, default=250, help="Number of random patches to sample.")
    p.add_argument("--mode",  choices=["tile", "resize", "crop"], default="tile", help="Alignment mode for images of different sizes.")
    p.add_argument("--seed",  type=int, default=0, help="RNG seed for reproducible patch sampling.")
    args = p.parse_args()

    ref_img = cv2.imread(str(args.ref), cv2.IMREAD_COLOR)
    synth_img = cv2.imread(str(args.synth), cv2.IMREAD_COLOR)
    if ref_img is None or synth_img is None:
        raise SystemExit("Error: Could not load one or both images.")

    aligned_ref, aligned_synth = _align(ref_img, synth_img, args.mode)

    rng = random.Random(args.seed)
    ssim_scores, mses = sample_patches(aligned_ref, aligned_synth, args.patch, args.num, rng)

    def _stats(arr: np.ndarray):
        return {
            "mean":   arr.mean(),
            "median": np.median(arr),
            "min":    arr.min(),
            "max":    arr.max(),
            "std":    arr.std(ddof=1), # Sample standard deviation
        }

    print(f"\n=== Patchwise statistics (N = {args.num}) ===")
    s_ssim = _stats(ssim_scores)
    print(f"SSIM : mean {s_ssim['mean']:.3f}, median {s_ssim['median']:.3f}, min {s_ssim['min']:.3f}, max {s_ssim['max']:.3f}, std {s_ssim['std']:.3f}")
    s_mse = _stats(mses)
    print(f"MSE  : mean {s_mse['mean']:.1f}, median {s_mse['median']:.1f}, min {s_mse['min']:.1f}, max {s_mse['max']:.1f}, std {s_mse['std']:.1f}")

if __name__ == "__main__":
    main()
