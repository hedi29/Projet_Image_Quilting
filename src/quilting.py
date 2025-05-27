import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing
import functools

# Top-level worker for multiprocessing to evaluate a single candidate patch.
# This must be a top-level function for pickling by multiprocessing.
def _top_level_worker_evaluate_patch(coord_tuple, input_texture, block_size, 
                                   output_texture, start_y, start_x, block_i, block_j, 
                                   compute_overlap_error_func):
    """Evaluates a candidate patch from input_texture at coord_tuple.
    Used by the multiprocessing pool in _find_best_block.
    """
    y_c, x_c = coord_tuple
    candidate_patch = input_texture[y_c:y_c+block_size, x_c:x_c+block_size]
    error = compute_overlap_error_func(
        candidate_patch, output_texture, start_y, start_x, block_i, block_j
    )
    return error, (y_c, x_c) # Return error and original coordinates of the patch

class ImageQuilting:
    """Implements the Image Quilting algorithm for texture synthesis."""
    
    def __init__(self, block_size: int = 32, overlap_ratio: float = 1/6, tolerance: float = 0.1):
        """Initializes the Image Quilting synthesizer.

        Args:
            block_size: Size of the square blocks (patches).
            overlap_ratio: Overlap size as a ratio of block_size.
            tolerance: Error tolerance for selecting among best matching blocks.
                       A value of 0.1 means blocks with error up to 10% above
                       minimum error are considered.
        """
        self.block_size = block_size
        # Ensure overlap is at least 1 pixel if block_size and overlap_ratio are small
        self.overlap = max(1, int(block_size * overlap_ratio))
        self.tolerance = tolerance
    
    def synthesize_texture(self, input_texture: np.ndarray, output_size: tuple) -> np.ndarray:
        """Synthesizes a new texture from an input_texture to a specified output_size."""
        output_height, output_width = output_size
        output_texture = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Step size determines how much to shift for each new block placement
        step_size = self.block_size - self.overlap
        # Calculate number of blocks needed horizontally and vertically
        num_blocks_h = (output_height - self.overlap + step_size -1) // step_size # Adjusted for full coverage
        num_blocks_w = (output_width - self.overlap + step_size -1) // step_size  # Adjusted for full coverage
        
        print(f"Synthesizing {num_blocks_h}×{num_blocks_w} blocks (Block: {self.block_size}px, Overlap: {self.overlap}px)...")
        
        # Quilt blocks in raster scan order (left-to-right, top-to-bottom)
        for i in tqdm(range(num_blocks_h), desc="Quilting rows"):
            for j in range(num_blocks_w):
                start_y = i * step_size
                start_x = j * step_size
                
                # Find the best block from input_texture for the current position
                best_block_candidate = self._find_best_block(input_texture, output_texture, 
                                                            start_y, start_x, i, j)
                
                # Determine the actual region to place the block in output_texture, handling boundaries
                end_y = min(start_y + self.block_size, output_height)
                end_x = min(start_x + self.block_size, output_width)
                
                # Crop the chosen block if it extends beyond output boundaries and place it
                output_texture[start_y:end_y, start_x:end_x] = \
                    best_block_candidate[:end_y-start_y, :end_x-start_x]
        
        return output_texture
    
    def _find_best_block(self, input_texture: np.ndarray, output_texture: np.ndarray, 
                         start_y: int, start_x: int, block_i: int, block_j: int) -> np.ndarray:
        """Finds the best matching block from input_texture for the current position.
        Uses parallel processing for candidate evaluation if not the first block.
        """
        # For the very first block (top-left corner), pick a random patch
        if block_i == 0 and block_j == 0:
            return self._random_patch(input_texture)

        h_input, w_input, _ = input_texture.shape
        # Generate all possible top-left coordinates for candidate patches from input_texture
        candidate_coords = [
            (y, x) for y in range(h_input - self.block_size + 1)
            for x in range(w_input - self.block_size + 1)
        ]

        if not candidate_coords: # Should not happen if input_texture is larger than block_size
            print("Warning: No valid candidate patches found in input texture. Returning random patch.")
            return self._random_patch(input_texture)

        # Prepare a partial function for multiprocessing.Pool.map
        # This binds fixed arguments, so pool.map only iterates over candidate_coords.
        partial_worker = functools.partial(_top_level_worker_evaluate_patch, 
                                           input_texture=input_texture, 
                                           block_size=self.block_size, 
                                           output_texture=output_texture, 
                                           start_y=start_y, start_x=start_x, 
                                           block_i=block_i, block_j=block_j, 
                                           compute_overlap_error_func=self._compute_overlap_error)

        error_coords_list = []
        try:
            # Use a pool of worker processes to evaluate candidate patches in parallel
            # Default is os.cpu_count(). Can be limited for testing, e.g., processes=4.
            with multiprocessing.Pool() as pool:
                error_coords_list = pool.map(partial_worker, candidate_coords)
        except Exception as e:
            # Fallback to sequential evaluation if multiprocessing fails
            print(f"Multiprocessing pool failed: {e}. Falling back to sequential patch evaluation.")
            for coord in candidate_coords:
                error_val, (y_val, x_val) = partial_worker(coord)
                error_coords_list.append((error_val, (y_val, x_val)))
        
        if not error_coords_list: # Should have been caught by `if not candidate_coords`
            print("Warning: error_coords_list is empty after evaluation. Returning random patch.")
            return self._random_patch(input_texture)

        all_errors = np.array([ec[0] for ec in error_coords_list])
        original_patch_coords = [ec[1] for ec in error_coords_list]

        if len(all_errors) == 0: # Should be redundant due to earlier checks
             print("Warning: all_errors is empty. Returning random patch.")
             return self._random_patch(input_texture)

        min_error_val = np.min(all_errors)
        # Define a threshold: min_error * (1 + tolerance)
        # Patches with error within this threshold are considered valid candidates
        threshold = min_error_val * (1 + self.tolerance)
        valid_indices = np.where(all_errors <= threshold)[0]

        if len(valid_indices) == 0:
            # If no patches meet the tolerance, pick the one with the absolute minimum error
            chosen_idx_in_error_list = np.argmin(all_errors)
        else:
            # Randomly pick one from the set of valid candidates
            chosen_idx_in_error_list = np.random.choice(valid_indices)

        # Retrieve the coordinates of the chosen patch from the original list
        chosen_y, chosen_x = original_patch_coords[chosen_idx_in_error_list]
        # Extract the chosen block from input_texture
        chosen_block = input_texture[chosen_y:chosen_y+self.block_size, 
                                     chosen_x:chosen_x+self.block_size].copy()
        
        # Apply min-cut blending if this is not the first block
        if block_i > 0 or block_j > 0:
            chosen_block = self._apply_min_cut(
                chosen_block, output_texture, start_y, start_x, block_i, block_j
            )
        
        return chosen_block
    
    def _compute_overlap_error(self, candidate_patch: np.ndarray, output_texture: np.ndarray, 
                               start_y: int, start_x: int, block_i: int, block_j: int) -> float:
        """Computes Sum of Squared Differences (SSD) error in overlap regions."""
        total_error = 0.0
        
        # Left overlap error (if not in the first column of blocks)
        if block_j > 0:
            cand_left_overlap = candidate_patch[:, :self.overlap]
            
            # Define the corresponding region in the existing output_texture
            out_reg_start_y = start_y
            out_reg_end_y = min(start_y + self.block_size, output_texture.shape[0])
            out_reg_start_x = start_x 
            out_reg_end_x = min(start_x + self.overlap, output_texture.shape[1])
            
            output_left_overlap = output_texture[out_reg_start_y:out_reg_end_y, out_reg_start_x:out_reg_end_x]
            
            # Ensure dimensions match for comparison (handles boundary cases)
            h_match = min(cand_left_overlap.shape[0], output_left_overlap.shape[0])
            w_match = min(cand_left_overlap.shape[1], output_left_overlap.shape[1])
            
            if h_match > 0 and w_match > 0:
                cand_region = cand_left_overlap[:h_match, :w_match]
                out_region = output_left_overlap[:h_match, :w_match]
                error = np.sum((cand_region.astype(float) - out_region.astype(float))**2)
                total_error += error
        
        # Top overlap error (if not in the first row of blocks)
        if block_i > 0:
            cand_top_overlap = candidate_patch[:self.overlap, :]

            out_reg_start_y = start_y
            out_reg_end_y = min(start_y + self.overlap, output_texture.shape[0])
            out_reg_start_x = start_x
            out_reg_end_x = min(start_x + self.block_size, output_texture.shape[1])
            
            output_top_overlap = output_texture[out_reg_start_y:out_reg_end_y, out_reg_start_x:out_reg_end_x]
            
            h_match = min(cand_top_overlap.shape[0], output_top_overlap.shape[0])
            w_match = min(cand_top_overlap.shape[1], output_top_overlap.shape[1])
            
            if h_match > 0 and w_match > 0:
                cand_region = cand_top_overlap[:h_match, :w_match]
                out_region = output_top_overlap[:h_match, :w_match]
                error = np.sum((cand_region.astype(float) - out_region.astype(float))**2)
                total_error += error
        
        return total_error
    
    def _apply_min_cut(self, chosen_block: np.ndarray, output_texture: np.ndarray, 
                       start_y: int, start_x: int, block_i: int, block_j: int) -> np.ndarray:
        """Applies minimum error boundary cut to blend chosen_block with output_texture."""
        blended_block = chosen_block.copy()
        
        # Vertical cut for left overlap (if not first column)
        if block_j > 0:
            # Extract overlapping regions
            candidate_overlap_left = chosen_block[:, :self.overlap]
            
            out_s_y, out_e_y = start_y, min(start_y + self.block_size, output_texture.shape[0])
            out_s_x, out_e_x = start_x, min(start_x + self.overlap, output_texture.shape[1])
            output_overlap_left = output_texture[out_s_y:out_e_y, out_s_x:out_e_x]
            
            # Ensure dimensions match
            h = min(candidate_overlap_left.shape[0], output_overlap_left.shape[0])
            w = min(candidate_overlap_left.shape[1], output_overlap_left.shape[1])
            
            if h > 0 and w > 0:
                cand_region = candidate_overlap_left[:h, :w]
                out_region = output_overlap_left[:h, :w]
                
                # SSD error surface (sum over color channels)
                error_surface_vertical = np.sum((cand_region.astype(float) - out_region.astype(float))**2, axis=2)
                cut_mask_vertical = self._find_vertical_cut(error_surface_vertical) # True for candidate, False for output
                
                # Apply the cut: where mask is False, use pixels from existing output_texture
                for r_idx in range(h):
                    for c_idx in range(w):
                        if not cut_mask_vertical[r_idx, c_idx]:
                            blended_block[r_idx, c_idx] = output_texture[start_y + r_idx, start_x + c_idx]
        
        # Horizontal cut for top overlap (if not first row)
        if block_i > 0:
            candidate_overlap_top = blended_block[:self.overlap, :] # Use current state of blended_block
            
            out_s_y, out_e_y = start_y, min(start_y + self.overlap, output_texture.shape[0])
            out_s_x, out_e_x = start_x, min(start_x + self.block_size, output_texture.shape[1])
            output_overlap_top = output_texture[out_s_y:out_e_y, out_s_x:out_e_x]
            
            h = min(candidate_overlap_top.shape[0], output_overlap_top.shape[0])
            w = min(candidate_overlap_top.shape[1], output_overlap_top.shape[1])
            
            if h > 0 and w > 0:
                cand_region = candidate_overlap_top[:h, :w]
                out_region = output_overlap_top[:h, :w]
                
                error_surface_horizontal = np.sum((cand_region.astype(float) - out_region.astype(float))**2, axis=2)
                cut_mask_horizontal = self._find_horizontal_cut(error_surface_horizontal) # True for candidate, False for output
                
                # Apply the cut, being careful not to overwrite parts already decided by vertical cut
                # if a pixel is in both vertical and horizontal overlap zones.
                # The current logic applies horizontal cut pixels if `not cut_mask_horizontal`.
                # A more robust way might involve combining masks or a specific corner handling.
                # However, paper's Fig 4 suggests vertical cut takes precedence in the corner for horizontal blending.
                # For vertical blending, horizontal takes precedence. This code has vertical first then horizontal.
                for r_idx in range(h):
                    for c_idx in range(w):
                        if not cut_mask_horizontal[r_idx, c_idx]:
                            # For the top-left corner (i < self.overlap and j < self.overlap):
                            # If vertical cut already decided to use output_texture (mask_v was False),
                            # and horizontal cut also says use output_texture (mask_h is False),
                            # it's consistent. If horizontal says use candidate but vertical said output,
                            # we need to respect the first (vertical) cut.
                            # The current simplified loop will overwrite. Let's refine:
                            # If it's in the left overlap (c_idx < self.overlap) AND the vertical cut mask for this pixel was False (meaning use output)
                            # then we should NOT overwrite it with the candidate block's pixel, even if horizontal cut says so.
                            # The crucial part is how cut_mask_vertical was applied to blended_block before this step.
                            # blended_block[r_idx, c_idx] would ALREADY be from output_texture if vertical cut decided so.
                            # So, if cut_mask_horizontal says use output, we just do that.
                            blended_block[r_idx, c_idx] = output_texture[start_y + r_idx, start_x + c_idx]
        
        return blended_block
    
    def _find_vertical_cut(self, error_surface: np.ndarray) -> np.ndarray:
        """Finds a min-cost vertical seam on the error_surface using dynamic programming.

        Args:
            error_surface: A 2D array where each element is the cost (e.g., SSD).

        Returns:
            A 2D boolean mask of the same shape as error_surface.
            `True` indicates pixels to the left of or on the seam (from new block),
            `False` indicates pixels to the right of the seam (from existing block).
        """
        h, w = error_surface.shape
        dp_costs = error_surface.copy() # DP table for accumulated costs
        
        # Fill DP table: cost to reach (r,c) is error_surface[r,c] + min_cost_from_previous_row
        for r in range(1, h):
            for c in range(w):
                cost_above_left = dp_costs[r-1, c-1] if c > 0 else np.inf
                cost_above = dp_costs[r-1, c]
                cost_above_right = dp_costs[r-1, c+1] if c < w-1 else np.inf
                dp_costs[r, c] += min(cost_above_left, cost_above, cost_above_right)
        
        # Backtrack to find the optimal path (seam)
        # True means this pixel is part of the new block (left of or on the cut)
        # False means it's part of the existing block (right of the cut)
        cut_mask = np.ones((h, w), dtype=bool) 
        
        # Start backtracking from the column with the minimum accumulated cost in the last row
        current_c = np.argmin(dp_costs[h-1, :])
        
        for r in range(h - 1, -1, -1):
            cut_mask[r, current_c:] = False # Pixels from current_c to the right are from existing block
            if r > 0:
                # Determine predecessor column for the seam path
                prev_costs = []
                prev_indices = []
                if current_c > 0:
                    prev_costs.append(dp_costs[r-1, current_c-1])
                    prev_indices.append(current_c-1)
                prev_costs.append(dp_costs[r-1, current_c])
                prev_indices.append(current_c)
                if current_c < w-1:
                    prev_costs.append(dp_costs[r-1, current_c+1])
                    prev_indices.append(current_c+1)
                
                current_c = prev_indices[np.argmin(prev_costs)]
        
        return cut_mask
    
    def _find_horizontal_cut(self, error_surface: np.ndarray) -> np.ndarray:
        """Finds a min-cost horizontal seam by transposing and using _find_vertical_cut."""
        # Transpose error surface, find vertical cut, then transpose mask back
        vertical_cut_on_transposed = self._find_vertical_cut(error_surface.T)
        return vertical_cut_on_transposed.T
    
    def _random_patch(self, texture: np.ndarray) -> np.ndarray:
        """Extracts a random block_size x block_size patch from the input texture."""
        h, w, _ = texture.shape
        # Ensure random coordinates are within valid bounds for extracting a full block
        rand_y = np.random.randint(0, h - self.block_size + 1)
        rand_x = np.random.randint(0, w - self.block_size + 1)
        return texture[rand_y : rand_y + self.block_size, rand_x : rand_x + self.block_size].copy()