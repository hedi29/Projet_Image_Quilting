"""
Image Quilting Algorithm - Simplified and Corrected Implementation
Based on 'Image Quilting for Texture Synthesis and Transfer' by Efros and Freeman
"""

import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing
import functools # Added for functools.partial

# Worker function defined at the top level for pickling
def _top_level_worker_evaluate_patch(coord_tuple, input_texture, block_size, 
                                   output_texture, start_y, start_x, block_i, block_j, 
                                   compute_overlap_error_func):
    y_c, x_c = coord_tuple
    candidate_patch = input_texture[y_c:y_c+block_size, x_c:x_c+block_size]
    error = compute_overlap_error_func(
        candidate_patch, output_texture, start_y, start_x, block_i, block_j
    )
    return error, (y_c, x_c) # Return error and original coordinates

class ImageQuilting:
    """Simplified and corrected implementation of Image Quilting algorithm."""
    
    def __init__(self, block_size=32, overlap_ratio=1/6, tolerance=0.1):
        """
        Initialize the Image Quilting algorithm.
        
        Parameters:
        -----------
        block_size : int
            Size of the square blocks
        overlap_ratio : float
            Overlap size as a ratio of block_size (default: 1/6 as in paper)
        tolerance : float
            Error tolerance for block selection (default: 0.1 as in paper)
        """
        self.block_size = block_size
        self.overlap = max(1, int(block_size * overlap_ratio))
        self.tolerance = tolerance
    
    def synthesize_texture(self, input_texture, output_size):
        """
        Synthesize texture using Image Quilting algorithm.
        
        Parameters:
        -----------
        input_texture : ndarray
            Input texture image (H, W, 3)
        output_size : tuple
            Desired output size (height, width)
            
        Returns:
        --------
        ndarray
            Synthesized texture
        """
        output_height, output_width = output_size
        
        # Initialize output texture
        output_texture = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Calculate step size and number of blocks
        step_size = self.block_size - self.overlap
        num_blocks_h = (output_height + step_size - 1) // step_size
        num_blocks_w = (output_width + step_size - 1) // step_size
        
        print(f"Synthesizing {num_blocks_h}×{num_blocks_w} blocks...")
        
        # Process blocks in raster scan order
        for i in tqdm(range(num_blocks_h), desc="Quilting"):
            for j in range(num_blocks_w):
                # Calculate placement position
                start_y = i * step_size
                start_x = j * step_size
                
                # Find and place the best block
                block = self._find_best_block(input_texture, output_texture, 
                                            start_y, start_x, i, j)
                
                # Calculate actual placement area (handle boundaries)
                end_y = min(start_y + self.block_size, output_height)
                end_x = min(start_x + self.block_size, output_width)
                
                # Place the block
                output_texture[start_y:end_y, start_x:end_x] = \
                    block[:end_y-start_y, :end_x-start_x]
        
        return output_texture
    
    def _find_best_block(self, input_texture, output_texture, start_y, start_x, block_i, block_j):
        """
        Find the best matching block for the current position.
        Parallelized candidate search using a top-level worker.
        """
        if block_i == 0 and block_j == 0:
            return self._random_patch(input_texture)

        h_input, w_input = input_texture.shape[:2]
        coords = []
        for y_coord in range(h_input - self.block_size + 1):
            for x_coord in range(w_input - self.block_size + 1):
                coords.append((y_coord, x_coord))

        if not coords:
            return self._random_patch(input_texture)

        # Use functools.partial to create a new function with some arguments pre-filled.
        # This makes it easier to use with pool.map, as pool.map expects a function with a single argument.
        # We pass input_texture directly here. For very large textures, shared memory might be better.
        partial_worker = functools.partial(_top_level_worker_evaluate_patch, 
                                           input_texture=input_texture, 
                                           block_size=self.block_size, 
                                           output_texture=output_texture, 
                                           start_y=start_y, start_x=start_x, 
                                           block_i=block_i, block_j=block_j, 
                                           compute_overlap_error_func=self._compute_overlap_error)

        error_coords_list = []
        try:
            # num_processes = min(multiprocessing.cpu_count(), 4) # For testing specific core counts
            # with multiprocessing.Pool(processes=num_processes) as pool:
            with multiprocessing.Pool() as pool: # Defaults to os.cpu_count()
                error_coords_list = pool.map(partial_worker, coords)
        except Exception as e:
            print(f"Multiprocessing pool failed: {e}. Falling back to sequential patch evaluation.")
            # Fallback sequential loop
            for coord in coords:
                error_val, (y_val, x_val) = partial_worker(coord) # Call the partial function directly
                error_coords_list.append((error_val, (y_val, x_val)))
        
        if not error_coords_list:
            return self._random_patch(input_texture)

        all_errors = np.array([ec[0] for ec in error_coords_list])
        original_patch_coords = [ec[1] for ec in error_coords_list]

        if len(all_errors) == 0:
             return self._random_patch(input_texture)

        min_error_val = np.min(all_errors)
        threshold = min_error_val * (1 + self.tolerance)
        valid_indices_in_error_list = np.where(all_errors <= threshold)[0]

        if len(valid_indices_in_error_list) == 0:
            if len(all_errors) > 0:
                 chosen_original_idx = np.argmin(all_errors)
            else:
                 return self._random_patch(input_texture)
        else:
            chosen_original_idx = np.random.choice(valid_indices_in_error_list)

        chosen_y, chosen_x = original_patch_coords[chosen_original_idx]
        chosen_block = input_texture[chosen_y:chosen_y+self.block_size, 
                                     chosen_x:chosen_x+self.block_size].copy()
        
        if block_i > 0 or block_j > 0:
            chosen_block = self._apply_min_cut(
                chosen_block, output_texture, start_y, start_x, block_i, block_j
            )
        
        return chosen_block
    
    def _compute_overlap_error(self, candidate, output_texture, start_y, start_x, block_i, block_j):
        """
        Compute overlap error between candidate block and existing output.
        
        Parameters:
        -----------
        candidate : ndarray
            Candidate block
        output_texture : ndarray
            Current output texture
        start_y, start_x : int
            Position in output texture
        block_i, block_j : int
            Block indices
            
        Returns:
        --------
        float
            Total overlap error
        """
        total_error = 0.0
        
        # Left overlap (if not first column)
        if block_j > 0:
            # Get overlap regions
            cand_left = candidate[:, :self.overlap]
            
            # Get corresponding region from output (handle boundaries)
            out_start_y = start_y
            out_end_y = min(start_y + self.block_size, output_texture.shape[0])
            out_start_x = start_x
            out_end_x = min(start_x + self.overlap, output_texture.shape[1])
            
            output_left = output_texture[out_start_y:out_end_y, out_start_x:out_end_x]
            
            # Adjust candidate region to match output region size
            h_match = min(cand_left.shape[0], output_left.shape[0])
            w_match = min(cand_left.shape[1], output_left.shape[1])
            
            if h_match > 0 and w_match > 0:
                cand_region = cand_left[:h_match, :w_match]
                out_region = output_left[:h_match, :w_match]
                
                # Compute L2 error
                error = np.sum((cand_region.astype(float) - out_region.astype(float))**2)
                total_error += error
        
        # Top overlap (if not first row)
        if block_i > 0:
            # Get overlap regions
            cand_top = candidate[:self.overlap, :]
            
            # Get corresponding region from output (handle boundaries)
            out_start_y = start_y
            out_end_y = min(start_y + self.overlap, output_texture.shape[0])
            out_start_x = start_x
            out_end_x = min(start_x + self.block_size, output_texture.shape[1])
            
            output_top = output_texture[out_start_y:out_end_y, out_start_x:out_end_x]
            
            # Adjust candidate region to match output region size
            h_match = min(cand_top.shape[0], output_top.shape[0])
            w_match = min(cand_top.shape[1], output_top.shape[1])
            
            if h_match > 0 and w_match > 0:
                cand_region = cand_top[:h_match, :w_match]
                out_region = output_top[:h_match, :w_match]
                
                # Compute L2 error
                error = np.sum((cand_region.astype(float) - out_region.astype(float))**2)
                total_error += error
        
        return total_error
    
    def _apply_min_cut(self, block, output_texture, start_y, start_x, block_i, block_j):
        """
        Apply minimum error boundary cut to seamlessly blend the block.
        
        Parameters:
        -----------
        block : ndarray
            Block to be placed
        output_texture : ndarray
            Current output texture
        start_y, start_x : int
            Position in output texture
        block_i, block_j : int
            Block indices
            
        Returns:
        --------
        ndarray
            Block with minimum cut applied
        """
        result_block = block.copy()
        
        # Apply vertical cut (left overlap)
        if block_j > 0:
            # Get overlap regions
            block_left = block[:, :self.overlap]
            
            out_start_y = start_y
            out_end_y = min(start_y + self.block_size, output_texture.shape[0])
            out_start_x = start_x
            out_end_x = min(start_x + self.overlap, output_texture.shape[1])
            
            output_left = output_texture[out_start_y:out_end_y, out_start_x:out_end_x]
            
            # Match dimensions
            h_match = min(block_left.shape[0], output_left.shape[0])
            w_match = min(block_left.shape[1], output_left.shape[1])
            
            if h_match > 0 and w_match > 0:
                block_region = block_left[:h_match, :w_match]
                output_region = output_left[:h_match, :w_match]
                
                # Compute error surface and find minimum cut
                error_surface = np.sum((block_region.astype(float) - 
                                     output_region.astype(float))**2, axis=2)
                
                cut_mask = self._find_vertical_cut(error_surface)
                
                # Apply cut
                for i in range(h_match):
                    for j in range(w_match):
                        if not cut_mask[i, j]:  # Use output texture
                            result_block[i, j] = output_texture[start_y + i, start_x + j]
        
        # Apply horizontal cut (top overlap)
        if block_i > 0:
            # Get overlap regions
            block_top = block[:self.overlap, :]
            
            out_start_y = start_y
            out_end_y = min(start_y + self.overlap, output_texture.shape[0])
            out_start_x = start_x
            out_end_x = min(start_x + self.block_size, output_texture.shape[1])
            
            output_top = output_texture[out_start_y:out_end_y, out_start_x:out_end_x]
            
            # Match dimensions
            h_match = min(block_top.shape[0], output_top.shape[0])
            w_match = min(block_top.shape[1], output_top.shape[1])
            
            if h_match > 0 and w_match > 0:
                block_region = block_top[:h_match, :w_match]
                output_region = output_top[:h_match, :w_match]
                
                # Compute error surface and find minimum cut
                error_surface = np.sum((block_region.astype(float) - 
                                     output_region.astype(float))**2, axis=2)
                
                cut_mask = self._find_horizontal_cut(error_surface)
                
                # Apply cut (but avoid overriding the vertical cut result)
                for i in range(h_match):
                    for j in range(w_match):
                        if not cut_mask[i, j]:  # Use output texture
                            # Only override if we're not in the left overlap region
                            if block_j == 0 or j >= self.overlap:
                                result_block[i, j] = output_texture[start_y + i, start_x + j]
        
        return result_block
    
    def _find_vertical_cut(self, error_surface):
        """
        Find minimum cost vertical cut using dynamic programming.
        
        Parameters:
        -----------
        error_surface : ndarray
            2D error surface
            
        Returns:
        --------
        ndarray
            Boolean mask (True = use new block, False = use existing)
        """
        h, w = error_surface.shape
        
        # Initialize DP table
        dp = error_surface.copy()
        
        # Fill DP table
        for i in range(1, h):
            for j in range(w):
                # Get possible previous positions
                prev_positions = []
                if j > 0:
                    prev_positions.append(dp[i-1, j-1])
                prev_positions.append(dp[i-1, j])
                if j < w-1:
                    prev_positions.append(dp[i-1, j+1])
                
                dp[i, j] += min(prev_positions)
        
        # Backtrack to find optimal path
        mask = np.ones((h, w), dtype=bool)
        
        # Start from minimum in last row
        j = np.argmin(dp[h-1, :])
        
        for i in range(h-1, -1, -1):
            # Mark cut boundary
            mask[i, j:] = False
            
            if i > 0:
                # Find best previous position
                prev_positions = []
                prev_indices = []
                
                if j > 0:
                    prev_positions.append(dp[i-1, j-1])
                    prev_indices.append(j-1)
                prev_positions.append(dp[i-1, j])
                prev_indices.append(j)
                if j < w-1:
                    prev_positions.append(dp[i-1, j+1])
                    prev_indices.append(j+1)
                
                # Move to position with minimum cost
                min_idx = np.argmin(prev_positions)
                j = prev_indices[min_idx]
        
        return mask
    
    def _find_horizontal_cut(self, error_surface):
        """
        Find minimum cost horizontal cut using dynamic programming.
        """
        # Transpose and use vertical cut, then transpose back
        return self._find_vertical_cut(error_surface.T).T
    
    def _random_patch(self, texture):
        """Extract a random patch from texture."""
        h, w = texture.shape[:2]
        i = np.random.randint(0, h - self.block_size + 1)
        j = np.random.randint(0, w - self.block_size + 1)
        return texture[i:i+self.block_size, j:j+self.block_size].copy()