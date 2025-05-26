"""
Texture Transfer implementation for Image Quilting.

This module implements texture transfer as described in the Efros & Freeman paper,
allowing to render one image using the texture style of another image.
"""

import numpy as np
from tqdm import tqdm
import cv2
from typing import Tuple, Optional, Dict
import time 

# Attempt to import the local correspondence_map module
try:
    from .correspondence_map import get_correspondence_map
except ImportError:
    # Fallback for running script directly or if module not found in path
    try:
        from correspondence_map import get_correspondence_map
        print("Loaded correspondence_map directly.")
    except ImportError:
        print("Critical: correspondence_map module not found!")
        # Define a placeholder if the import fails, to allow basic script execution
        def get_correspondence_map(image, map_type, blur_sigma=None):
            print(f"Warning: Using placeholder get_correspondence_map for {map_type}")
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            gray = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-8) * 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=2)

class TextureTransfer:
    """
    Texture Transfer using Image Quilting with correspondence constraints.
    """
    
    def __init__(self, block_size: int = 32, overlap_ratio: float = 1/6, 
                 alpha: float = 0.8, iterations: int = 3, tolerance: float = 0.1):
        """
        Initialize texture transfer algorithm.
        
        Parameters:
        -----------
        block_size : int
            Size of quilting blocks.
        overlap_ratio : float
            Overlap size as ratio of block_size.
        alpha : float
            Weight between texture constraint (alpha) and correspondence constraint (1-alpha).
            alpha = 1.0: only texture matters, alpha = 0.0: only correspondence matters.
        iterations : int
            Number of refinement iterations with decreasing block sizes.
        tolerance : float
            Error tolerance for block selection (e.g., 0.1 means 10% above min_error).
        """
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        if not (0 < overlap_ratio < 1):
            raise ValueError("Overlap ratio must be between 0 and 1 (exclusive).")
        if block_size <= 0:
            raise ValueError("Block size must be positive.")
        if iterations <=0:
            raise ValueError("Number of iterations must be positive.")
            
        self.block_size = block_size
        self.overlap_ratio = overlap_ratio # Store ratio for dynamic overlap calculation
        self.alpha = alpha
        self.iterations = iterations
        self.tolerance = tolerance
        
        self.source_patches: Optional[Dict[str, np.ndarray]] = None
        self.source_patch_dims: Optional[Tuple[int, int, int, int]] = None # Store H,W,C for patches

    def _get_overlap(self, current_block_size: int) -> int:
        """Calculate overlap based on current block size."""
        return max(1, int(current_block_size * self.overlap_ratio))

    def transfer_texture(self, source_texture: np.ndarray, 
                        target_image: np.ndarray,
                        correspondence_map_type: str = 'luminance',
                        blur_sigma_source: float = 3.0,
                        blur_sigma_target: float = 3.0,
                        output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Transfer texture from source to target image.
        
        Parameters:
        -----------
        source_texture : ndarray
            Source texture to transfer (H, W, 3 or H, W).
        target_image : ndarray
            Target image that guides the transfer (H, W, 3 or H, W).
        correspondence_map_type : str
            Type of correspondence: 'luminance' (default) or 'blurred'.
        blur_sigma_source : float
            Sigma for blurring source correspondence map (if 'blurred').
        blur_sigma_target : float
            Sigma for blurring target correspondence map (if 'blurred').
        output_size : tuple, optional
            Output size (height, width), defaults to target image size.
            
        Returns:
        --------
        ndarray
            Transferred texture image (H, W, 3).
        """
        
        start_time = time.time()
        
        # Ensure images are 3-channel uint8 for processing
        source_texture = self._prepare_image(source_texture, "Source Texture")
        target_image = self._prepare_image(target_image, "Target Image")

        if output_size is None:
            output_size = target_image.shape[:2] # (height, width)
        else:
            if not (isinstance(output_size, tuple) and len(output_size) == 2 and 
                      all(isinstance(dim, int) and dim > 0 for dim in output_size)):
                raise ValueError("output_size must be a tuple of two positive integers (height, width).")

        print(f"🎨 Starting texture transfer...")
        print(f"   Source: {source_texture.shape[:2]}, Target: {target_image.shape[:2]}, Output: {output_size}")
        print(f"   Alpha: {self.alpha}, Iterations: {self.iterations}, Initial Block Size: {self.block_size}")
        
        # Generate correspondence maps (these will be H, W, 3, uint8)
        print(f"   Generating correspondence maps ('{correspondence_map_type}')...")
        source_corr = get_correspondence_map(source_texture, correspondence_map_type, blur_sigma=blur_sigma_source)
        target_corr = get_correspondence_map(target_image, correspondence_map_type, blur_sigma=blur_sigma_target)
        
        # Resize target correspondence to output size if needed
        if target_corr.shape[:2] != output_size:
            print(f"   Resizing target correspondence map to {output_size}...")
            target_corr = cv2.resize(target_corr, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Initialize output with random noise or a simple fill
        # Using a mean color fill can sometimes be better than random noise
        output_texture = np.full((*output_size, 3), np.mean(source_texture, axis=(0,1)), dtype=np.uint8)
        
        # Multi-resolution synthesis
        current_block_size = self.block_size
        
        for iteration in range(self.iterations):
            iter_start_time = time.time()
            print(f"\n🔄 Iteration {iteration+1}/{self.iterations} (Block Size: {current_block_size}) başladı...")
            
            # Adjust alpha for each iteration (more texture weight initially, more correspondence later)
            # This schedule can be tuned. Original paper uses fixed alpha.
            # Let's try a schedule that emphasizes structure more in later iterations.
            current_alpha = self.alpha * (0.9 ** iteration) # Decrease alpha slightly
            # current_alpha = self.alpha # Or keep it fixed as per original paper for simplicity
            print(f"   Current Alpha: {current_alpha:.3f}")

            # Pre-extract source patches for the current block size
            # This is done per iteration because block_size changes
            self._precompute_patches(source_texture, source_corr, current_block_size)
            
            # Perform transfer for this iteration
            output_texture = self._transfer_iteration(
                current_output=output_texture, 
                target_correspondence=target_corr.astype(np.float32), # For error calcs
                block_size=current_block_size, 
                alpha=current_alpha, 
                iteration_num=iteration
            )
            
            # Reduce block size for next iteration (e.g., by 1/2 or 2/3)
            # Ensure block size doesn't get too small (e.g., min 8 or 16)
            if iteration < self.iterations - 1:
                current_block_size = max(16, int(current_block_size * 0.67))
            
            iter_time = time.time() - iter_start_time
            print(f"   Iteration {iteration+1} completed in {iter_time:.2f}s.")

        total_time = time.time() - start_time
        print(f"\n✅ Texture transfer completed in {total_time:.2f}s!")
        return output_texture

    def _prepare_image(self, image: np.ndarray, name: str) -> np.ndarray:
        """Ensure image is HWC, uint8, 3-channel."""
        if image.ndim == 2: # Grayscale
            print(f"   Converting {name} from grayscale to RGB.")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1: # Grayscale with channel dim
            print(f"   Converting {name} from single-channel grayscale to RGB.")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4: # RGBA
            print(f"   Converting {name} from RGBA to RGB.")
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name} has unsupported shape {image.shape}. Expected (H,W,3) or (H,W).")
        
        if image.dtype != np.uint8:
            print(f"   Converting {name} dtype from {image.dtype} to uint8.")
            if np.issubdtype(image.dtype, np.floating):
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            else: # Assume integer type, clip and cast
                image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _precompute_patches(self, source_texture: np.ndarray, 
                           source_correspondence: np.ndarray, 
                           current_block_size: int):
        """Pre-extract all patches from source texture and correspondence map for current block size."""
        h, w, _ = source_texture.shape
        ch, cw, cc = source_correspondence.shape
        assert h == ch and w == cw, "Source texture and correspondence map must have same H, W."

        num_patches_y = h - current_block_size + 1
        num_patches_x = w - current_block_size + 1
        
        if num_patches_y <= 0 or num_patches_x <= 0:
            raise ValueError(
                f"Source texture (shape {h}x{w}) is too small for block_size {current_block_size}. "
                f"Need at least {current_block_size}x{current_block_size}."
            )
        
        total_patches = num_patches_y * num_patches_x
        
        # Pre-allocate arrays
        # Storing as float32 for correspondence for precision in error calculations
        self.source_patches = {
            'texture': np.zeros((total_patches, current_block_size, current_block_size, 3), dtype=np.uint8),
            'correspondence': np.zeros((total_patches, current_block_size, current_block_size, 3), dtype=np.float32)
        }
        
        idx = 0
        for y in range(num_patches_y):
            for x in range(num_patches_x):
                self.source_patches['texture'][idx] = source_texture[y:y+current_block_size, x:x+current_block_size]
                self.source_patches['correspondence'][idx] = source_correspondence[y:y+current_block_size, x:x+current_block_size].astype(np.float32)
                idx += 1
        
        print(f"   Pre-extracted {total_patches} source patches (size {current_block_size}x{current_block_size}).")

    def _transfer_iteration(self, current_output: np.ndarray, 
                           target_correspondence: np.ndarray, # Should be float32
                           block_size: int, alpha: float, iteration_num: int) -> np.ndarray:
        """Perform one iteration of texture transfer."""
        output_h, output_w, _ = current_output.shape
        overlap = self._get_overlap(block_size)
        step_size = block_size - overlap
        
        # Pad output array to handle boundaries easily during quilting
        # This avoids complex boundary checks for every block
        # Padded amount should be at least `overlap`
        pad_width = overlap # Simplified padding, can be block_size for very robust handling
        padded_output = cv2.copyMakeBorder(current_output, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_REFLECT)
        padded_target_corr = cv2.copyMakeBorder(target_correspondence, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_REFLECT) 

        # Calculate number of blocks based on original output size
        num_blocks_y = (output_h - overlap + step_size -1) // step_size # considers first block fully placed
        num_blocks_x = (output_w - overlap + step_size -1) // step_size

        # Fill order: typically scanline, but random can be good too.
        # For simplicity, using scanline.
        block_indices = [(r, c) for r in range(num_blocks_y) for c in range(num_blocks_x)]
        if iteration_num % 2 == 1 : # Shuffle on alternate iterations
            np.random.shuffle(block_indices)

        for r_idx, c_idx in tqdm(block_indices, desc=f" Iteration {iteration_num+1} Quilting", unit="block"):
            y_start_output = r_idx * step_size
            x_start_output = c_idx * step_size
            
            # Coordinates in the padded array
            y_start_padded = y_start_output + pad_width
            x_start_padded = x_start_output + pad_width

            # Define the region in the *original* target correspondence map for this block
            target_corr_region = target_correspondence[
                y_start_output : min(y_start_output + block_size, output_h),
                x_start_output : min(x_start_output + block_size, output_w)
            ]
            
            # Find best block
            chosen_patch_texture, chosen_patch_corr = self._find_best_transfer_patch(
                padded_output, 
                target_corr_region, 
                y_start_padded, x_start_padded, 
                block_size, overlap, alpha, 
                is_first_row=(r_idx == 0), is_first_col=(c_idx == 0)
            )
            
            # Apply boundary cut and place the block
            final_block = self._apply_min_cut(
                chosen_patch_texture, 
                padded_output, 
                y_start_padded, x_start_padded, 
                block_size, overlap, 
                is_first_row=(r_idx == 0), is_first_col=(c_idx == 0)
            )
            
            # Place the block into the padded output array
            # Ensure block is not placed outside output boundaries (important for last blocks)
            h_block, w_block, _ = final_block.shape
            
            # Clip to actual output region if block is near boundary
            # y_end_padded = min(y_start_padded + h_block, padded_output.shape[0])
            # x_end_padded = min(x_start_padded + w_block, padded_output.shape[1])
            # h_place = y_end_padded - y_start_padded
            # w_place = x_end_padded - x_start_padded
            
            # This needs to map to original current_output size, not padded
            y_end_output = min(y_start_output + block_size, output_h)
            x_end_output = min(x_start_output + block_size, output_w)

            actual_block_h = y_end_output - y_start_output
            actual_block_w = x_end_output - x_start_output

            padded_output[y_start_padded : y_start_padded + actual_block_h, 
                          x_start_padded : x_start_padded + actual_block_w] = final_block[:actual_block_h, :actual_block_w]

        # Extract the relevant part from the padded output
        result_output = padded_output[pad_width : pad_width + output_h, pad_width : pad_width + output_w]
        return result_output

    def _find_best_transfer_patch(self, current_padded_output: np.ndarray, 
                                 target_corr_block_region: np.ndarray, # Region from original target_corr
                                 y_pos_padded: int, x_pos_padded: int, # Top-left of block in padded_output
                                 block_size: int, overlap: int, alpha: float, 
                                 is_first_row: bool, is_first_col: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Finds best source patch based on texture and correspondence error."""
        min_err = float('inf')
        best_patches_indices = [] # Store indices of patches within tolerance

        num_source_patches = self.source_patches['texture'].shape[0]
        errors = np.zeros(num_source_patches)

        for i in range(num_source_patches):
            source_patch_tex = self.source_patches['texture'][i]
            source_patch_corr = self.source_patches['correspondence'][i] # Already float32
            
            # 1. Texture Error (SSD in overlap regions)
            tex_err = 0.0
            if not is_first_row or not is_first_col: # No overlap error for the very first block (0,0)
                tex_err = self._calculate_overlap_error(
                    source_patch_tex, current_padded_output, 
                    y_pos_padded, x_pos_padded, 
                    block_size, overlap, is_first_row, is_first_col
                )
            
            # 2. Correspondence Error (SSD between source patch correspondence and target region correspondence)
            #    The target_corr_block_region is already the correct H,W for the current block.
            #    Source patch correspondence map needs to be matched to this size.
            h_corr_match = min(source_patch_corr.shape[0], target_corr_block_region.shape[0])
            w_corr_match = min(source_patch_corr.shape[1], target_corr_block_region.shape[1])

            if h_corr_match == 0 or w_corr_match == 0: # Should not happen if block_size > 0
                corr_err = float('inf')
            else:
                patch_corr_subset = source_patch_corr[:h_corr_match, :w_corr_match]
                target_corr_subset = target_corr_block_region[:h_corr_match, :w_corr_match]
                # L2 error, ensure types are float for subtraction
                corr_err = np.sum((patch_corr_subset - target_corr_subset.astype(np.float32))**2)
                # Normalize by size of overlap and number of channels (3)
                corr_err /= (h_corr_match * w_corr_match * 3 + 1e-6) # Add epsilon to avoid div by zero

            total_err = alpha * tex_err + (1 - alpha) * corr_err
            errors[i] = total_err
        
        min_err_val = np.min(errors)
        threshold = min_err_val * (1 + self.tolerance)
        candidate_indices = np.where(errors <= threshold)[0]
        
        if len(candidate_indices) == 0: # Should not happen if errors are finite
            chosen_idx = np.argmin(errors) # Fallback: pick absolute minimum
        else:
            chosen_idx = np.random.choice(candidate_indices)
            
        return self.source_patches['texture'][chosen_idx], self.source_patches['correspondence'][chosen_idx]

    def _calculate_overlap_error(self, patch: np.ndarray, 
                                 canvas_padded: np.ndarray, 
                                 y_pos_padded: int, x_pos_padded: int, 
                                 block_size: int, overlap: int, 
                                 is_first_row: bool, is_first_col: bool) -> float:
        """Calculates SSD error in overlap regions."""
        err = 0.0
        patch_float = patch.astype(np.float32)
        
        # Top overlap (if not first row)
        if not is_first_row:
            overlap_top_patch = patch_float[:overlap, :]
            overlap_top_canvas = canvas_padded[y_pos_padded : y_pos_padded + overlap, 
                                             x_pos_padded : x_pos_padded + block_size].astype(np.float32)
            err += np.sum((overlap_top_patch - overlap_top_canvas)**2)

        # Left overlap (if not first col)
        if not is_first_col:
            overlap_left_patch = patch_float[:, :overlap]
            overlap_left_canvas = canvas_padded[y_pos_padded : y_pos_padded + block_size, 
                                              x_pos_padded : x_pos_padded + overlap].astype(np.float32)
            err += np.sum((overlap_left_patch - overlap_left_canvas)**2)
            
        # Subtract double-counted corner if overlapping both top and left
        if not is_first_row and not is_first_col:
            overlap_corner_patch = patch_float[:overlap, :overlap]
            overlap_corner_canvas = canvas_padded[y_pos_padded : y_pos_padded + overlap,
                                                 x_pos_padded : x_pos_padded + overlap].astype(np.float32)
            err -= np.sum((overlap_corner_patch - overlap_corner_canvas)**2)
            
        # Normalize by number of pixels in overlap and channels (3)
        num_pixels_overlap = 0
        if not is_first_row: num_pixels_overlap += overlap * block_size * 3
        if not is_first_col: num_pixels_overlap += overlap * block_size * 3
        if not is_first_row and not is_first_col: num_pixels_overlap -= overlap * overlap * 3 # Subtract corner
        
        return err / (num_pixels_overlap + 1e-6) # Add epsilon

    def _apply_min_cut(self, patch_to_place: np.ndarray, 
                       canvas_padded: np.ndarray, 
                       y_pos_padded: int, x_pos_padded: int, 
                       block_size: int, overlap: int, 
                       is_first_row: bool, is_first_col: bool) -> np.ndarray:
        """Applies minimum error boundary cut to the patch."""
        cut_patch = patch_to_place.copy()

        # Vertical cut for left overlap
        if not is_first_col:
            # Region of the patch that overlaps with canvas on the left
            patch_overlap_left = patch_to_place[:, :overlap].astype(np.float32)
            # Corresponding region on the canvas
            canvas_overlap_left = canvas_padded[y_pos_padded : y_pos_padded + block_size, 
                                              x_pos_padded : x_pos_padded + overlap].astype(np.float32)
            
            # SSD error surface for the vertical cut
            err_surface_vert = np.sum((patch_overlap_left - canvas_overlap_left)**2, axis=2)
            path_mask_vert = self._find_min_cost_path_vertical(err_surface_vert) # True for patch, False for canvas

            for r in range(block_size):
                for c in range(overlap):
                    if not path_mask_vert[r, c]: # If canvas part is chosen by cut
                        cut_patch[r, c] = canvas_padded[y_pos_padded + r, x_pos_padded + c]
        
        # Horizontal cut for top overlap
        if not is_first_row:
            # Region of the patch that overlaps with canvas on the top
            patch_overlap_top = patch_to_place[:overlap, :].astype(np.float32)
            # Corresponding region on the canvas
            canvas_overlap_top = canvas_padded[y_pos_padded : y_pos_padded + overlap, 
                                             x_pos_padded : x_pos_padded + block_size].astype(np.float32)
            
            # SSD error surface for the horizontal cut
            err_surface_horiz = np.sum((patch_overlap_top - canvas_overlap_top)**2, axis=2)
            path_mask_horiz = self._find_min_cost_path_horizontal(err_surface_horiz) # True for patch, False for canvas

            for r in range(overlap):
                for c in range(block_size):
                    # Critical: Combine cuts correctly. If already taken from canvas by vertical cut,
                    # or if horizontal cut says take from canvas, then take from canvas.
                    # This means we prioritize the canvas if *either* cut says so in the common corner.
                    if not path_mask_horiz[r, c]:
                         cut_patch[r, c] = canvas_padded[y_pos_padded + r, x_pos_padded + c]
        return cut_patch

    def _find_min_cost_path_vertical(self, cost_surface: np.ndarray) -> np.ndarray:
        """Finds the min-cost path vertically. Returns a boolean mask (True=patch, False=canvas)."""
        # Cost surface is (height, overlap_width)
        h, w = cost_surface.shape
        dp_costs = np.zeros_like(cost_surface, dtype=np.float64)
        dp_paths = np.zeros_like(cost_surface, dtype=int) # To store previous column index

        dp_costs[:, 0] = cost_surface[:, 0]

        for c in range(1, w):
            for r in range(h):
                costs = []
                if r > 0: costs.append(dp_costs[r-1, c-1])
                costs.append(dp_costs[r, c-1])
                if r < h - 1: costs.append(dp_costs[r+1, c-1])
                
                min_prev_cost = costs[np.argmin(costs)]
                dp_costs[r,c] = cost_surface[r,c] + min_prev_cost
                
                # Path (for backtracking - not directly used in this simplified mask version)
                # if r > 0 and min_prev_cost == dp_costs[r-1, c-1]: dp_paths[r,c] = r-1
                # elif min_prev_cost == dp_costs[r, c-1]: dp_paths[r,c] = r
                # elif r < h-1 and min_prev_cost == dp_costs[r+1, c-1]: dp_paths[r,c] = r+1
        
        # Simpler path construction for mask: if cost_surface[r,c] is part of path from left.
        # This means the cut passes between (r, c-1) and (r, c).
        # The mask should be True for pixels from the new patch, False for pixels from the old canvas.
        # This is typically done by finding the path along the minimum values in dp_costs, column by column.
        # Let's trace back from the minimum of the last column (simplified)
        mask = np.ones((h,w), dtype=bool) # True = patch, False = canvas
        # Correct DP for vertical cut: iterate rows, then columns
        # Cost(i, j) = E(i, j) + min(Cost(i-1, j-1), Cost(i, j-1), Cost(i+1, j-1))
        # This is for path from left to right over the width of the overlap.
        # For a vertical cut dividing left (canvas) and right (patch) parts of overlap:
        # Path moves from top to bottom over the height of the overlap.
        # Cost(i,j) = E(i,j) + min( Cost(i-1, j-1), Cost(i-1, j), Cost(i-1, j+1) )
        
        # Re-implement DP correctly for vertical cut (path moves downwards)
        dp = cost_surface.copy()
        parent = np.zeros_like(cost_surface, dtype=int)

        for i in range(1, h):
            for j in range(w):
                choices = [dp[i-1, j]] # Straight from above
                indices = [j]
                if j > 0:
                    choices.append(dp[i-1, j-1]) # From top-left
                    indices.append(j-1)
                if j < w - 1:
                    choices.append(dp[i-1, j+1]) # From top-right
                    indices.append(j+1)
                
                min_cost_idx = np.argmin(choices)
                dp[i,j] += choices[min_cost_idx]
                parent[i,j] = indices[min_cost_idx]
        
        # Backtrack to find the cut path
        path_col = np.argmin(dp[h-1, :])
        for i in range(h - 1, -1, -1):
            mask[i, :path_col] = False # Canvas pixels to the left of cut
            mask[i, path_col] = True # Pixel on the cut itself (can be debated, assign to patch)
                                     # Pixels to the right are patch by default (mask initialized to True)
            if i > 0:
                path_col = parent[i, path_col]
        return mask

    def _find_min_cost_path_horizontal(self, cost_surface: np.ndarray) -> np.ndarray:
        """Finds the min-cost path horizontally. Returns a boolean mask.
           Cost surface is (overlap_height, width)
        """
        # Transpose, find vertical cut, then transpose back.
        mask_transposed = self._find_min_cost_path_vertical(cost_surface.T)
        return mask_transposed.T


# Convenience function for simple usage
def transfer_texture_simplified(source_texture: np.ndarray,
                                target_image: np.ndarray, 
                                alpha: float = 0.8,
                                block_size: int = 36, # Common good starting size
                                correspondence_type: str = 'luminance',
                                iterations: int = 3,
                                output_size: Optional[Tuple[int, int]] = None,
                                blur_sigma: float = 3.0 # Generic blur sigma if blurred map is used
                               ) -> np.ndarray:
    """
    Simplified texture transfer function.
    
    Parameters:
    -----------
    source_texture : ndarray
        Source texture image (H, W, C) or (H, W).
    target_image : ndarray
        Target image to guide the transfer (H, W, C) or (H, W).
    alpha : float, optional
        Balance: alpha=1.0 (pure texture), alpha=0.0 (pure correspondence). Default is 0.8.
    block_size : int, optional
        Initial size of quilting blocks. Default is 36.
    correspondence_type : str, optional
        'luminance' (default) or 'blurred'.
    iterations : int, optional
        Number of refinement passes. Default is 3.
    output_size : tuple (height, width), optional
        Desired output size. Defaults to target_image size.
    blur_sigma : float, optional
        Sigma for Gaussian blur if 'blurred' correspondence is used. Default 3.0.
        
    Returns:
    --------
    ndarray
        The resulting image with transferred texture.
    """
    
    transfer_agent = TextureTransfer(
        block_size=block_size,
        alpha=alpha,
        iterations=iterations,
        overlap_ratio=1/6, # Standard overlap
        tolerance=0.1      # Standard tolerance
    )
    
    return transfer_agent.transfer_texture(
        source_texture,
        target_image,
        correspondence_map_type=correspondence_type,
        blur_sigma_source=blur_sigma, # Using same blur for both for simplicity here
        blur_sigma_target=blur_sigma,
        output_size=output_size
    )

# Example Usage (if run as main script)
if __name__ == '__main__':
    print("Running Texture Transfer example...")

    # Create dummy images for testing
    # Source Texture: A simple checkerboard pattern
    source_h, source_w = 128, 128
    source_tex = np.zeros((source_h, source_w, 3), dtype=np.uint8)
    c1, c2 = [200, 50, 50], [50, 50, 200] # Redish and Blueish colors
    sq_size = 32
    for r in range(0, source_h, sq_size):
        for c in range(0, source_w, sq_size):
            color = c1 if (r//sq_size + c//sq_size) % 2 == 0 else c2
            source_tex[r : r+sq_size, c : c+sq_size] = color

    # Target Image: A gradient image
    target_h, target_w = 180, 240
    target_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    for r in range(target_h):
        for c in range(target_w):
            val = int((r / target_h) * 200 + (c / target_w) * 55) # Mix of vertical and horizontal gradient
            target_img[r, c] = [val, val, val] # Grayscale gradient
    
    # Make one part of target image colored to see correspondence effect
    target_img[target_h//4 : 3*target_h//4, target_w//4 : 3*target_w//4] = [50, 180, 50] # Greenish square

    print(f"Source texture shape: {source_tex.shape}")
    print(f"Target image shape: {target_img.shape}")

    # --- Parameters for transfer ---
    output_h, output_w = 256, 256 # Desired output size
    selected_block_size = 40
    selected_alpha = 0.5 # 0.1 = more target structure, 0.9 = more source texture
    selected_iterations = 2 # 2-3 is usually good for testing
    selected_correspondence = 'luminance' # 'luminance' or 'blurred'
    selected_blur_sigma = 5.0

    print(f"\nTransfer settings: block_size={selected_block_size}, alpha={selected_alpha}, iters={selected_iterations}, map='{selected_correspondence}'")

    try:
        # Test with the simplified function
        result_image = transfer_texture_simplified(
            source_texture=source_tex,
            target_image=target_img,
            alpha=selected_alpha,
            block_size=selected_block_size,
            correspondence_type=selected_correspondence,
            iterations=selected_iterations,
            output_size=(output_h, output_w),
            blur_sigma=selected_blur_sigma
        )
        
        print(f"\nResult image shape: {result_image.shape}")
        assert result_image.shape == (output_h, output_w, 3), "Output shape mismatch!"
        assert result_image.dtype == np.uint8, "Output dtype mismatch!"

        # Displaying images (requires OpenCV windowing support)
        try:
            cv2.imshow("Source Texture", source_tex)
            cv2.imshow("Target Image", target_img)
            cv2.imshow(f"Texture Transfer Result (alpha={selected_alpha}, block={selected_block_size})", result_image)
            print("\nDisplaying images. Press any key in an OpenCV window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"Could not display images (OpenCV GUI might not be available): {e}")
            print("If running in a headless environment, this is expected.")
            # Optionally, save the image if display fails
            cv2.imwrite("source_texture_test.png", source_tex)
            cv2.imwrite("target_image_test.png", target_img)
            cv2.imwrite("texture_transfer_result.png", result_image)
            print("Saved test images to: source_texture_test.png, target_image_test.png, texture_transfer_result.png")

    except ValueError as ve:
        print(f"\nTexture transfer failed with ValueError: {ve}")
    except ImportError as ie:
        print(f"\nTexture transfer failed due to ImportError: {ie}")
        print("Please ensure `correspondence_map.py` is in the same directory or accessible in PYTHONPATH.")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred during texture transfer: {e}")
        traceback.print_exc()

    print("\nTexture Transfer example finished.")