"""
Core implementation of the Image Quilting algorithm for texture synthesis.

This module implements the texture synthesis algorithm described in:
'Image Quilting for Texture Synthesis and Transfer' by Efros and Freeman.
"""

import numpy as np
from tqdm import tqdm


def random_patch(texture, block_size):
    """Extract a random patch from the texture.
    
    Parameters:
    -----------
    texture : ndarray
        Input texture image as a numpy array
    block_size : int
        Size of the square block to extract
        
    Returns:
    --------
    ndarray
        A randomly selected block_size x block_size patch from the texture
    """
    h, w = texture.shape[:2]
    i = np.random.randint(0, h - block_size)
    j = np.random.randint(0, w - block_size)
    return texture[i:i+block_size, j:j+block_size].copy()


def get_overlapping_blocks(texture, block_size, overlap):
    """Get all possible overlapping blocks from the texture.
    
    Parameters:
    -----------
    texture : ndarray
        Input texture image as a numpy array
    block_size : int
        Size of the square block to extract
    overlap : int
        Size of the overlap region
        
    Returns:
    --------
    list
        A list of (block, position) tuples where position is (i, j) coordinates
    """
    h, w = texture.shape[:2]
    blocks = []
    
    for i in range(0, h - block_size + 1):
        for j in range(0, w - block_size + 1):
            block = texture[i:i+block_size, j:j+block_size].copy()
            blocks.append((block, (i, j)))
            
    return blocks


def compute_overlap_error(block, output_texture, current_i, current_j, block_size, overlap, direction):
    """Compute error in the overlap region between the block and existing output.
    
    Parameters:
    -----------
    block : ndarray
        The candidate block to place
    output_texture : ndarray
        The current state of the output texture
    current_i, current_j : int
        The position to place the block in the output texture
    block_size : int
        Size of the blocks
    overlap : int
        Size of the overlap region
    direction : str
        Direction of overlap: 'left', 'top', or 'both'
        
    Returns:
    --------
    ndarray
        Error surface in the overlap region
    """
    if direction == 'left':
        # Left overlap (compare with blocks to the left)
        error = np.sum((block[:, :overlap] - 
                      output_texture[current_i:current_i+block_size, 
                                   current_j:current_j+overlap])**2, axis=2)
        return error
    
    elif direction == 'top':
        # Top overlap (compare with blocks above)
        block_region = block[:overlap, :]
        output_region = output_texture[current_i:current_i+overlap, 
                                    current_j:current_j+block_size]
        
        # Ensure the shapes match before computing the error
        if output_region.shape[1] < block_region.shape[1]:
            # If output region is smaller (at the boundaries), trim the block region
            block_region = block_region[:, :output_region.shape[1]]
        elif output_region.shape[1] > block_region.shape[1]:
            # If output region is larger, use only what matches the block size
            output_region = output_region[:, :block_region.shape[1]]
            
        error = np.sum((block_region - output_region)**2, axis=2)
        return error
    
    elif direction == 'both':
        # Both left and top overlap
        # Left overlap
        left_block = block[:, :overlap]
        left_output = output_texture[current_i:current_i+block_size, 
                                   current_j:current_j+overlap]
        
        # Ensure left shapes match
        if left_output.shape[0] < left_block.shape[0]:
            left_block = left_block[:left_output.shape[0], :]
        elif left_output.shape[0] > left_block.shape[0]:
            left_output = left_output[:left_block.shape[0], :]
            
        left_error = np.sum((left_block - left_output)**2, axis=2)
        
        # Top overlap
        top_block = block[:overlap, :]
        top_output = output_texture[current_i:current_i+overlap, 
                                  current_j:current_j+block_size]
        
        # Ensure top shapes match
        if top_output.shape[1] < top_block.shape[1]:
            top_block = top_block[:, :top_output.shape[1]]
        elif top_output.shape[1] > top_block.shape[1]:
            top_output = top_output[:, :top_block.shape[1]]
            
        top_error = np.sum((top_block - top_output)**2, axis=2)
        
        # Corner region is counted twice, so we need to handle it separately
        corner_block = block[:overlap, :overlap]
        corner_output = output_texture[current_i:current_i+overlap, 
                                    current_j:current_j+overlap]
        
        # Ensure corner shapes match
        min_h = min(corner_block.shape[0], corner_output.shape[0])
        min_w = min(corner_block.shape[1], corner_output.shape[1])
        corner_block = corner_block[:min_h, :min_w]
        corner_output = corner_output[:min_h, :min_w]
        
        corner = np.sum((corner_block - corner_output)**2, axis=2)
        
        # Combine errors (we need to reshape to make dimensions match)
        error_left = np.zeros((block_size, block_size))
        error_left[:left_error.shape[0], :overlap] = left_error
        
        error_top = np.zeros((block_size, block_size))
        error_top[:overlap, :top_error.shape[1]] = top_error
        
        # Combine errors, but avoid double-counting the corner
        error = error_left + error_top
        error[:min_h, :min_w] = corner
        
        return error
    
    else:
        raise ValueError(f"Unknown direction: {direction}")


def find_best_block(texture, output_texture, current_i, current_j, 
                   block_size, overlap, tolerance=0.1, num_candidates=30):
    """Find the best matching block for the current position.
    
    Parameters:
    -----------
    texture : ndarray
        Input texture image
    output_texture : ndarray
        Current output texture being synthesized
    current_i, current_j : int
        Current position in the output texture
    block_size : int
        Size of the blocks
    overlap : int
        Size of the overlap region
    tolerance : float
        Error tolerance for selecting candidate blocks
    num_candidates : int
        Number of candidate blocks to consider
        
    Returns:
    --------
    ndarray
        The best matching block
    """
    h, w = texture.shape[:2]
    candidates = []
    errors = []
    
    # Determine overlap direction
    if current_i == 0 and current_j == 0:
        # First block, no overlap
        return random_patch(texture, block_size)
    elif current_i == 0:
        # Top row, only left overlap
        direction = 'left'
    elif current_j == 0:
        # Left column, only top overlap
        direction = 'top'
    else:
        # Both top and left overlap
        direction = 'both'
    
    # Try a number of random blocks and keep track of their errors
    for _ in range(100):  # Try 100 random blocks
        candidate = random_patch(texture, block_size)
        error = compute_overlap_error(candidate, output_texture, current_i, current_j, 
                                     block_size, overlap, direction)
        total_error = np.sum(error)
        candidates.append(candidate)
        errors.append(total_error)
    
    # Convert to numpy array for easier manipulation
    errors = np.array(errors)
    
    # Find the minimum error
    min_error = np.min(errors)
    
    # Find all blocks within the tolerance of the minimum error
    good_blocks = np.where(errors <= min_error * (1 + tolerance))[0]
    
    # Choose a random block from the good candidates
    chosen_idx = np.random.choice(good_blocks)
    return candidates[chosen_idx]


def minimum_error_boundary_cut(error_surface, direction):
    """Find minimum cost path through the error surface using dynamic programming.
    
    Parameters:
    -----------
    error_surface : ndarray
        2D array of errors in overlap region
    direction : str
        'vertical' or 'horizontal' cutting direction
        
    Returns:
    --------
    ndarray
        Binary mask defining the boundary cut
    """
    if direction == 'vertical':
        # For vertical overlap (cutting along vertical line)
        height, width = error_surface.shape
        cumulative_error = error_surface.copy()
        
        # Compute cumulative minimum error for all possible paths
        for i in range(1, height):
            for j in range(width):
                if j == 0:
                    cumulative_error[i, j] += min(cumulative_error[i-1, j], 
                                                cumulative_error[i-1, j+1])
                elif j == width - 1:
                    cumulative_error[i, j] += min(cumulative_error[i-1, j-1], 
                                                 cumulative_error[i-1, j])
                else:
                    cumulative_error[i, j] += min(cumulative_error[i-1, j-1], 
                                                 cumulative_error[i-1, j], 
                                                 cumulative_error[i-1, j+1])
        
        # Backtrack to find the optimal path
        mask = np.ones_like(error_surface, dtype=bool)
        j = np.argmin(cumulative_error[-1, :])
        
        for i in range(height-1, -1, -1):
            mask[i, j:] = False
            if i > 0:
                if j == 0:
                    j = j + np.argmin([cumulative_error[i-1, j], 
                                      cumulative_error[i-1, j+1]])
                elif j == width - 1:
                    j = j + np.argmin([cumulative_error[i-1, j-1], 
                                      cumulative_error[i-1, j]]) - 1
                else:
                    j = j + np.argmin([cumulative_error[i-1, j-1], 
                                      cumulative_error[i-1, j], 
                                      cumulative_error[i-1, j+1]]) - 1
        
        return mask
    
    elif direction == 'horizontal':
        # For horizontal overlap (cutting along horizontal line)
        # Transpose the error surface and treat it as a vertical cut
        return minimum_error_boundary_cut(error_surface.T, 'vertical').T
    
    else:
        raise ValueError(f"Unknown direction: {direction}")


def merge_blocks_with_cut(current_block, output_texture, current_i, current_j, 
                         block_size, overlap):
    """Merge the current block with the output texture using minimum error boundary cut.
    
    Parameters:
    -----------
    current_block : ndarray
        The block to be placed
    output_texture : ndarray
        Current output texture
    current_i, current_j : int
        Position to place the block
    block_size : int
        Size of the blocks
    overlap : int
        Size of the overlap region
        
    Returns:
    --------
    ndarray
        Merged block with minimum error boundary
    """
    # Create a copy of the block to modify
    merged_block = current_block.copy()
    
    # Handle left overlap
    if current_j > 0:
        # Get the regions to compare
        left_block = current_block[:, :overlap]
        left_output = output_texture[current_i:current_i+block_size, 
                                   current_j:current_j+overlap]
        
        # Ensure shapes match
        min_h = min(left_block.shape[0], left_output.shape[0])
        left_block = left_block[:min_h, :]
        left_output = left_output[:min_h, :]
        
        # Compute error surface in the overlap region
        left_error = np.sum((left_block - left_output)**2, axis=2)
        
        # Find minimum error boundary cut
        left_mask = minimum_error_boundary_cut(left_error, 'vertical')
        
        # Apply the mask to merge the blocks
        for i in range(min_h):
            for j in range(overlap):
                if left_mask[i, j]:
                    merged_block[i, j] = output_texture[current_i+i, current_j+j]
    
    # Handle top overlap
    if current_i > 0:
        # Get the regions to compare
        top_block = current_block[:overlap, :]
        top_output = output_texture[current_i:current_i+overlap, 
                                  current_j:current_j+block_size]
        
        # Ensure shapes match
        min_w = min(top_block.shape[1], top_output.shape[1])
        top_block = top_block[:, :min_w]
        top_output = top_output[:, :min_w]
        
        # Compute error surface in the overlap region
        top_error = np.sum((top_block - top_output)**2, axis=2)
        
        # Find minimum error boundary cut
        top_mask = minimum_error_boundary_cut(top_error, 'horizontal')
        
        # Apply the mask to merge the blocks
        for i in range(overlap):
            for j in range(min_w):
                if top_mask[i, j]:
                    merged_block[i, j] = output_texture[current_i+i, current_j+j]
    
    return merged_block


def synthesize_texture(input_texture, output_height, output_width, block_size, overlap_size, tolerance=0.1):
    """Synthesize a larger texture from an input texture using image quilting.
    
    Parameters:
    -----------
    input_texture : ndarray
        Input texture image (numpy array)
    output_height, output_width : int
        Size of the output texture
    block_size : int
        Size of the blocks to be quilted
    overlap_size : int
        Size of the overlap between blocks
    tolerance : float
        Error tolerance for choosing matching blocks
        
    Returns:
    --------
    ndarray
        The synthesized texture
    """
    # Calculate number of blocks needed
    num_blocks_h = (output_height - overlap_size) // (block_size - overlap_size) + 1
    num_blocks_w = (output_width - overlap_size) // (block_size - overlap_size) + 1
    
    # Initialize output texture with zeros
    output_texture = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Iterate over blocks in raster scan order
    for i in tqdm(range(num_blocks_h), desc="Synthesizing texture"):
        for j in range(num_blocks_w):
            # Calculate current position
            current_i = i * (block_size - overlap_size)
            current_j = j * (block_size - overlap_size)
            
            # Find best matching block
            current_block = find_best_block(input_texture, output_texture, current_i, current_j, 
                                           block_size, overlap_size, tolerance)
            
            # Merge the block with minimum error boundary cut
            if i > 0 or j > 0:
                current_block = merge_blocks_with_cut(current_block, output_texture, current_i, current_j, 
                                                    block_size, overlap_size)
            
            # Calculate the actual block size to paste (handle edge cases)
            actual_h = min(block_size, output_height - current_i)
            actual_w = min(block_size, output_width - current_j)
            
            # Paste the block onto the output texture
            output_texture[current_i:current_i+actual_h, current_j:current_j+actual_w] = \
                current_block[:actual_h, :actual_w]
    
    return output_texture