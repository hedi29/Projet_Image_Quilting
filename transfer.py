import cv2
import numpy as np
import argparse
from pathlib import Path

###############################################################
#                         Core helpers                        #
###############################################################

def extract_luminance(image: np.ndarray) -> np.ndarray:
    """Return the V (luminance) channel from an HSV conversion of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2].copy()


def extract_blocks_and_luminance(image: np.ndarray, block_size: int):
    """Slide a window over *image* and collect RGB blocks and their luminance blocks."""
    h, w = image.shape[:2]
    lum_img = extract_luminance(image)
    blocks, lum_blocks = [], []
    for y in range(0, h - block_size + 1):
        for x in range(0, w - block_size + 1):
            blocks.append(image[y : y + block_size, x : x + block_size].copy())
            lum_blocks.append(lum_img[y : y + block_size, x : x + block_size].copy())
    return blocks, lum_blocks


###############################################################
#                    Error and seam functions                 #
###############################################################

def overlap_error_horizontal(place: np.ndarray, cand: np.ndarray, ov: int) -> float:
    diff = place[:, -ov:, :].astype(np.float32) - cand[:, :ov, :].astype(np.float32)
    return float(np.sum(diff ** 2))


def overlap_error_vertical(place: np.ndarray, cand: np.ndarray, ov: int) -> float:
    diff = place[-ov:, :, :].astype(np.float32) - cand[:ov, :, :].astype(np.float32)
    return float(np.sum(diff ** 2))


def correspondence_error(cand_lum: np.ndarray, target_lum: np.ndarray) -> float:
    if cand_lum.shape != target_lum.shape:
        target_lum = cv2.resize(target_lum, (cand_lum.shape[1], cand_lum.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = cand_lum.astype(np.float32) - target_lum.astype(np.float32)
    return float(np.sum(diff ** 2))


def choose_block(blocks, lum_blocks, left_blk, top_blk, ov, alpha, target_lum):
    """Return the RGB block that minimises the weighted error criterion."""
    best_idx, best_err = 0, float("inf")
    for i, blk in enumerate(blocks):
        err = (1 - alpha) * correspondence_error(lum_blocks[i], target_lum)
        if left_blk is not None:
            err += alpha * overlap_error_vertical(left_blk, blk, ov)
        if top_blk is not None:
            err += alpha * overlap_error_horizontal(top_blk, blk, ov)
        if err < best_err:
            best_err, best_idx = err, i
    return blocks[best_idx].copy()


########################  Min‑cut seams  ########################

def _min_cut_vertical(e: np.ndarray) -> np.ndarray:
    rows, cols = e.shape
    E = np.zeros_like(e, dtype=np.float64)
    E[0] = e[0]
    for i in range(1, rows):
        for j in range(cols):
            E[i, j] = e[i, j] + min(
                E[i - 1, j],
                E[i - 1, j - 1] if j > 0 else np.inf,
                E[i - 1, j + 1] if j < cols - 1 else np.inf,
            )
    seam = np.zeros(rows, dtype=np.int32)
    seam[-1] = int(np.argmin(E[-1]))
    for i in range(rows - 2, -1, -1):
        prev = seam[i + 1]
        candidates = [prev]
        if prev > 0:
            candidates.append(prev - 1)
        if prev < cols - 1:
            candidates.append(prev + 1)
        seam[i] = min(candidates, key=lambda c: E[i, c])
    return seam


def _min_cut_horizontal(e: np.ndarray) -> np.ndarray:
    rows, cols = e.shape
    E = np.zeros_like(e, dtype=np.float64)
    E[:, 0] = e[:, 0]
    for j in range(1, cols):
        for i in range(rows):
            E[i, j] = e[i, j] + min(
                E[i, j - 1],
                E[i - 1, j - 1] if i > 0 else np.inf,
                E[i + 1, j - 1] if i < rows - 1 else np.inf,
            )
    seam = np.zeros(cols, dtype=np.int32)
    seam[-1] = int(np.argmin(E[:, -1]))
    for j in range(cols - 2, -1, -1):
        prev = seam[j + 1]
        candidates = [prev]
        if prev > 0:
            candidates.append(prev - 1)
        if prev < rows - 1:
            candidates.append(prev + 1)
        seam[j] = min(candidates, key=lambda r: E[r, j])
    return seam


###############################################################
#                       Fusion helpers                        #
###############################################################

def fusion_horizontal(left_blk, cand_blk, ov):
    size = left_blk.shape[0]
    overlap_left = left_blk[:, size - ov :, :]
    overlap_cand = cand_blk[:, :ov, :]
    e = np.sum((overlap_left.astype(np.float32) - overlap_cand.astype(np.float32)) ** 2, axis=2)
    seam = _min_cut_vertical(e)
    mask = np.zeros((size, ov), dtype=np.uint8)
    for y, cut_x in enumerate(seam):
        mask[y, :cut_x] = 1
    fused = cand_blk.copy()
    fused[:, :ov][mask == 1] = overlap_left[mask == 1]
    return fused


def fusion_vertical(top_blk, cand_blk, ov):
    size = top_blk.shape[1]
    overlap_top = top_blk[-ov:, :, :]
    overlap_cand = cand_blk[:ov, :, :]
    e = np.sum((overlap_top.astype(np.float32) - overlap_cand.astype(np.float32)) ** 2, axis=2)
    seam = _min_cut_horizontal(e)
    mask = np.zeros((ov, size), dtype=np.uint8)
    for x, cut_y in enumerate(seam):
        mask[:cut_y, x] = 1
    fused = cand_blk.copy()
    fused[:ov][mask == 1] = overlap_top[mask == 1]
    return fused


def fusion_mixte(left_blk, top_blk, cand_blk, ov):
    size = cand_blk.shape[0]
    # Vertical seam (left)
    overlap_left = left_blk[:, size - ov :, :]
    overlap_cand_v = cand_blk[:, :ov, :]
    e_v = np.sum((overlap_left.astype(np.float32) - overlap_cand_v.astype(np.float32)) ** 2, axis=2)
    seam_v = _min_cut_vertical(e_v)
    mask_v = np.zeros((size, ov), dtype=np.uint8)
    for y, cut_x in enumerate(seam_v):
        mask_v[y, :cut_x] = 1
    # Horizontal seam (top)
    overlap_top = top_blk[-ov:, :, :]
    overlap_cand_h = cand_blk[:ov, :, :]
    e_h = np.sum((overlap_top.astype(np.float32) - overlap_cand_h.astype(np.float32)) ** 2, axis=2)
    seam_h = _min_cut_horizontal(e_h)
    mask_h = np.zeros((ov, size), dtype=np.uint8)
    for x, cut_y in enumerate(seam_h):
        mask_h[:cut_y, x] = 1
    # Apply masks
    fused = cand_blk.copy()
    fused[:, :ov][mask_v == 1] = overlap_left[mask_v == 1]
    fused[:ov][mask_h == 1] = overlap_top[mask_h == 1]
    return fused


###############################################################
#                      Quilting transfer                      #
###############################################################

def quilting_transfer(texture: np.ndarray, target: np.ndarray, block_size: int, overlap_rate: float, alpha: float) -> np.ndarray:
    """Transfer *texture* onto *target* using Efros & Freeman image quilting (single‑pass)."""
    blocks, lum_blocks = extract_blocks_and_luminance(texture, block_size)
    target_lum_full = extract_luminance(target)

    ov = int(round(overlap_rate * block_size))
    step = block_size - ov
    out_h, out_w = target.shape[:2]

    n_x = int(np.ceil((out_w - ov) / float(step)))
    n_y = int(np.ceil((out_h - ov) / float(step)))
    temp_w = ov + n_x * step
    temp_h = ov + n_y * step
    result = np.zeros((temp_h, temp_w, 3), dtype=np.uint8)

    for by in range(n_y):
        for bx in range(n_x):
            y, x = by * step, bx * step
            # Target luminance patch (zero‑padded if needed)
            tl_patch = np.zeros((block_size, block_size), dtype=target_lum_full.dtype)
            y_end = min(y + block_size, target_lum_full.shape[0])
            x_end = min(x + block_size, target_lum_full.shape[1])
            tl_patch[: y_end - y, : x_end - x] = target_lum_full[y:y_end, x:x_end]

            if by == 0 and bx == 0:  # first block
                chosen = choose_block(blocks, lum_blocks, None, None, ov, alpha, tl_patch)
                result[y : y + block_size, x : x + block_size] = chosen
            elif by == 0:  # first row (only left overlap)
                left_blk = result[y : y + block_size, x - step : x - step + block_size]
                chosen = choose_block(blocks, lum_blocks, left_blk, None, ov, alpha, tl_patch)
                fused = fusion_horizontal(left_blk, chosen, ov)
                result[y : y + block_size, x : x + block_size] = fused
            elif bx == 0:  # first column (only top overlap)
                top_blk = result[y - step : y - step + block_size, x : x + block_size]
                chosen = choose_block(blocks, lum_blocks, None, top_blk, ov, alpha, tl_patch)
                fused = fusion_vertical(top_blk, chosen, ov)
                result[y : y + block_size, x : x + block_size] = fused
            else:  # general case
                left_blk = result[y : y + block_size, x - step : x - step + block_size]
                top_blk = result[y - step : y - step + block_size, x : x + block_size]
                chosen = choose_block(blocks, lum_blocks, left_blk, top_blk, ov, alpha, tl_patch)
                fused = fusion_mixte(left_blk, top_blk, chosen, ov)
                result[y : y + block_size, x : x + block_size] = fused

    return result[:out_h, :out_w]


###############################################################
#                           CLI                               #
###############################################################

def main():
    parser = argparse.ArgumentParser(
        description="Texture‑to‑image transfer using image quilting (single pass).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--texture", required=True, type=Path, help="Path to the texture source image")
    parser.add_argument("--target", required=True, type=Path, help="Path to the target image")
    parser.add_argument("--output", type=Path, default="transfer.png", help="Where to save the result")
    parser.add_argument("--block_size", type=int, default=40, help="Size of quilting blocks in pixels")
    parser.add_argument("--overlap_rate", type=float, default=0.15, help="Fraction of block used as overlap (0‒1)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight of overlap vs. luminance term (0‒1)")

    args = parser.parse_args()

    texture = cv2.imread(str(args.texture), cv2.IMREAD_COLOR)
    target = cv2.imread(str(args.target), cv2.IMREAD_COLOR)
    if texture is None or target is None:
        raise FileNotFoundError("Unable to load texture or target image. Check the provided paths.")

    result = quilting_transfer(texture, target, args.block_size, args.overlap_rate, args.alpha)
    cv2.imwrite(str(args.output), result)
    print(f"✅ Transfer complete → {args.output}")


if __name__ == "__main__":
    main()
