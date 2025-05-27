import cv2
import numpy as np
import argparse
from pathlib import Path

###############################################################
#                         Core helpers                        #
###############################################################

def extract_luminance(image: np.ndarray) -> np.ndarray:
    """Extracts the V (luminance) channel from an BGR image via HSV conversion."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2].copy()


def extract_blocks_and_luminance(image: np.ndarray, block_size: int):
    """Slides a window over `image`, collecting RGB blocks and their luminance versions."""
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
    """Calculates SSD error for horizontal overlap between `place` and `cand`."""
    diff = place[:, -ov:, :].astype(np.float32) - cand[:, :ov, :].astype(np.float32)
    return float(np.sum(diff ** 2))


def overlap_error_vertical(place: np.ndarray, cand: np.ndarray, ov: int) -> float:
    """Calculates SSD error for vertical overlap between `place` and `cand`."""
    diff = place[-ov:, :, :].astype(np.float32) - cand[:ov, :, :].astype(np.float32)
    return float(np.sum(diff ** 2))


def correspondence_error(cand_lum: np.ndarray, target_lum: np.ndarray) -> float:
    """Calculates SSD error between candidate luminance and target luminance.
    Resizes target_lum if shapes mismatch.
    """
    if cand_lum.shape != target_lum.shape:
        target_lum = cv2.resize(target_lum, (cand_lum.shape[1], cand_lum.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = cand_lum.astype(np.float32) - target_lum.astype(np.float32)
    return float(np.sum(diff ** 2))


def choose_block(blocks, lum_blocks, left_blk, top_blk, ov, alpha, target_lum):
    """Selects the best RGB block based on a weighted sum of overlap and correspondence error."""
    best_idx, best_err = 0, float("inf")
    for i, blk in enumerate(blocks):
        err = (1 - alpha) * correspondence_error(lum_blocks[i], target_lum)
        if left_blk is not None: # Add error from left overlap if applicable
            err += alpha * overlap_error_vertical(left_blk, blk, ov) # Note: `overlap_error_vertical` is used for the left block's right edge.
        if top_blk is not None: # Add error from top overlap if applicable
            err += alpha * overlap_error_horizontal(top_blk, blk, ov) # Note: `overlap_error_horizontal` is used for the top block's bottom edge.
        if err < best_err:
            best_err, best_idx = err, i
    return blocks[best_idx].copy()


########################  Min‑cut seams  ########################

def _min_cut_vertical(e: np.ndarray) -> np.ndarray:
    """Finds the minimum cost vertical seam path using dynamic programming."""
    rows, cols = e.shape
    E = np.zeros_like(e, dtype=np.float64) # Accumulated cost matrix
    E[0] = e[0] # First row costs are just the errors themselves
    # Fill DP table
    for i in range(1, rows):
        for j in range(cols):
            E[i, j] = e[i, j] + min(
                E[i - 1, j],
                E[i - 1, j - 1] if j > 0 else np.inf,
                E[i - 1, j + 1] if j < cols - 1 else np.inf,
            )
    # Backtrack to find the seam
    seam = np.zeros(rows, dtype=np.int32)
    seam[-1] = int(np.argmin(E[-1])) # Start from min cost in last row
    for i in range(rows - 2, -1, -1):
        prev_col = seam[i + 1]
        candidates = [prev_col]
        if prev_col > 0:
            candidates.append(prev_col - 1)
        if prev_col < cols - 1:
            candidates.append(prev_col + 1)
        seam[i] = min(candidates, key=lambda c: E[i, c]) # Choose predecessor with min accumulated cost
    return seam


def _min_cut_horizontal(e: np.ndarray) -> np.ndarray:
    """Finds the minimum cost horizontal seam path by transposing and using vertical cut."""
    return _min_cut_vertical(e.T).T # Transpose, find vertical cut, transpose back


###############################################################
#                       Fusion helpers                        #
###############################################################

def fusion_horizontal(left_blk, cand_blk, ov):
    """Fuses `cand_blk` to the right of `left_blk` using a vertical min-cut seam."""
    size = left_blk.shape[0] # Assuming square blocks or at least same height
    overlap_left = left_blk[:, size - ov:, :] # Right part of left_blk
    overlap_cand = cand_blk[:, :ov, :]      # Left part of cand_blk
    # SSD error in the BGR channels, summed to get a 2D error surface
    e = np.sum((overlap_left.astype(np.float32) - overlap_cand.astype(np.float32)) ** 2, axis=2)
    seam = _min_cut_vertical(e) # Find optimal vertical cut path

    # Create a mask from the seam: 1 for left_blk, 0 for cand_blk
    mask = np.zeros((size, ov), dtype=np.uint8)
    for y, cut_x in enumerate(seam):
        mask[y, :cut_x] = 1 # Pixels to the left of cut_x come from left_blk
    
    fused = cand_blk.copy()
    # Apply the mask: where mask is 1, take pixels from overlap_left
    fused[:, :ov][mask == 1] = overlap_left[mask == 1]
    return fused


def fusion_vertical(top_blk, cand_blk, ov):
    """Fuses `cand_blk` below `top_blk` using a horizontal min-cut seam."""
    size = top_blk.shape[1] # Assuming square blocks or at least same width
    overlap_top = top_blk[-ov:, :, :] # Bottom part of top_blk
    overlap_cand = cand_blk[:ov, :, :]   # Top part of cand_blk
    e = np.sum((overlap_top.astype(np.float32) - overlap_cand.astype(np.float32)) ** 2, axis=2)
    seam = _min_cut_horizontal(e) # Find optimal horizontal cut path

    mask = np.zeros((ov, size), dtype=np.uint8)
    for x, cut_y in enumerate(seam):
        mask[:cut_y, x] = 1 # Pixels above cut_y come from top_blk

    fused = cand_blk.copy()
    fused[:ov][mask == 1] = overlap_top[mask == 1]
    return fused


def fusion_mixte(left_blk, top_blk, cand_blk, ov):
    """Fuses `cand_blk` with both `left_blk` and `top_blk` using respective min-cut seams."""
    size_h = cand_blk.shape[0] # Height
    size_w = cand_blk.shape[1] # Width

    fused = cand_blk.copy()

    # Vertical seam (left overlap)
    overlap_left = left_blk[:, size_w - ov:, :] 
    overlap_cand_v = cand_blk[:, :ov, :]      
    e_v = np.sum((overlap_left.astype(np.float32) - overlap_cand_v.astype(np.float32)) ** 2, axis=2)
    seam_v = _min_cut_vertical(e_v)
    mask_v = np.zeros((size_h, ov), dtype=np.uint8)
    for y, cut_x in enumerate(seam_v):
        mask_v[y, :cut_x] = 1
    fused[:, :ov][mask_v == 1] = overlap_left[mask_v == 1]

    # Horizontal seam (top overlap)
    # Important: Use the *already partially fused* block for the horizontal cut in the corner
    # This ensures consistency if the vertical cut modified pixels in the top-left overlap region of cand_blk
    overlap_top = top_blk[-ov:, :, :]         
    overlap_cand_h_fused = fused[:ov, :, :] # Use current state of fused block for candidate top overlap
    e_h = np.sum((overlap_top.astype(np.float32) - overlap_cand_h_fused.astype(np.float32)) ** 2, axis=2)
    seam_h = _min_cut_horizontal(e_h)
    mask_h = np.zeros((ov, size_w), dtype=np.uint8)
    for x, cut_y in enumerate(seam_h):
        mask_h[:cut_y, x] = 1
    # Apply horizontal cut. This will correctly blend with the vertical cut in the corner.
    fused[:ov][mask_h == 1] = overlap_top[mask_h == 1]
    
    return fused


###############################################################
#                      Quilting transfer                      #
###############################################################

def quilting_transfer(texture: np.ndarray, target: np.ndarray, block_size: int, overlap_rate: float, alpha: float) -> np.ndarray:
    """Transfers `texture` style onto `target` image using quilting.
    
    Args:
        texture: Source texture image (BGR).
        target: Target image to transfer style to (BGR).
        block_size: Size of square blocks.
        overlap_rate: Fractional overlap between blocks (0 to 1).
        alpha: Weight for correspondence error vs. overlap error (0 to 1).
               Alpha=0 is like standard quilting, Alpha=1 ignores neighbors.

    Returns:
        Resulting image with texture transferred onto target.
    """
    blocks, lum_blocks = extract_blocks_and_luminance(texture, block_size)
    target_lum_full = extract_luminance(target)

    ov = int(round(overlap_rate * block_size)) # Overlap in pixels
    step = block_size - ov # Step size for placing blocks
    out_h, out_w = target.shape[:2]

    # Calculate dimensions for the temporary canvas to build the result
    n_x = int(np.ceil((out_w - ov) / float(step)))
    n_y = int(np.ceil((out_h - ov) / float(step)))
    temp_w = ov + n_x * step
    temp_h = ov + n_y * step
    result = np.zeros((temp_h, temp_w, 3), dtype=np.uint8)

    for by in range(n_y): # Block row index
        for bx in range(n_x): # Block column index
            y, x = by * step, bx * step # Top-left corner of current block in result
            
            # Extract corresponding target luminance patch for correspondence error
            tl_patch = np.zeros((block_size, block_size), dtype=target_lum_full.dtype)
            y_end_target = min(y + block_size, target_lum_full.shape[0])
            x_end_target = min(x + block_size, target_lum_full.shape[1])
            tl_patch[: y_end_target - y, : x_end_target - x] = target_lum_full[y:y_end_target, x:x_end_target]

            if by == 0 and bx == 0:  # First block (top-left corner)
                chosen_block = choose_block(blocks, lum_blocks, None, None, ov, alpha, tl_patch)
                block_to_place = chosen_block
            elif by == 0:  # First row (horizontal overlap only)
                left_neighbor = result[y : y + block_size, x - step : x - step + block_size]
                chosen_block = choose_block(blocks, lum_blocks, left_neighbor, None, ov, alpha, tl_patch)
                block_to_place = fusion_horizontal(left_neighbor, chosen_block, ov)
            elif bx == 0:  # First column (vertical overlap only)
                top_neighbor = result[y - step : y - step + block_size, x : x + block_size]
                chosen_block = choose_block(blocks, lum_blocks, None, top_neighbor, ov, alpha, tl_patch)
                block_to_place = fusion_vertical(top_neighbor, chosen_block, ov)
            else:  # General case (both horizontal and vertical overlaps)
                left_neighbor = result[y : y + block_size, x - step : x - step + block_size]
                top_neighbor = result[y - step : y - step + block_size, x : x + block_size]
                chosen_block = choose_block(blocks, lum_blocks, left_neighbor, top_neighbor, ov, alpha, tl_patch)
                block_to_place = fusion_mixte(left_neighbor, top_neighbor, chosen_block, ov)
            
            result[y : y + block_size, x : x + block_size] = block_to_place

    return result[:out_h, :out_w] # Crop to original target dimensions


###############################################################
#                           CLI                               #
###############################################################

def main():
    parser = argparse.ArgumentParser(
        description="Texture-to-image transfer using single-pass image quilting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--texture", "-t", required=True, type=Path, help="Path to the source texture image.")
    parser.add_argument("--target", "-g", required=True, type=Path, help="Path to the target guide image.") # Changed from "target image"
    parser.add_argument("--output", "-o", type=Path, default="transfer_result.png", help="Path to save the output image.")
    parser.add_argument("--block_size", "-b", type=int, default=36, help="Size of quilting blocks in pixels.")
    parser.add_argument("--overlap_rate", "-r", type=float, default=0.1, help="Fraction of block for overlap (0.0-1.0).")
    parser.add_argument("--alpha", "-a", type=float, default=0.1, help="Weight for correspondence vs. overlap error (0.0-1.0). 0.0=quilting, 1.0=no spatial coherence.")

    args = parser.parse_args()

    texture_img = cv2.imread(str(args.texture), cv2.IMREAD_COLOR)
    target_img = cv2.imread(str(args.target), cv2.IMREAD_COLOR)

    if texture_img is None:
        raise FileNotFoundError(f"Unable to load texture image: {args.texture}")
    if target_img is None:
        raise FileNotFoundError(f"Unable to load target image: {args.target}")

    print(f"Starting texture transfer... Texture: {args.texture}, Target: {args.target}")
    result_img = quilting_transfer(texture_img, target_img, args.block_size, args.overlap_rate, args.alpha)
    
    # Ensure output directory exists
    output_dir = args.output.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(args.output), result_img)
    print(f"Texture transfer complete. Result saved to: {args.output}")


if __name__ == "__main__":
    main()
