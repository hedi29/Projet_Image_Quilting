"""
Texture transfer extension for the Image Quilting algorithm.

This module implements the texture transfer algorithm described in:
'Image Quilting for Texture Synthesis and Transfer' by Efros and Freeman.
"""

import numpy as np
import cv2
from tqdm import tqdm
from .quilting import compute_overlap_error, minimum_error_boundary_cut, merge_blocks_with_cut


def find_best_block_for_transfer(source_texture, output_texture, target_image, target_region,
                                current_i, current_j, block_size, overlap_size, alpha, tolerance=0.1):
    """Find the best matching block for texture transfer.
    
    Parameters:
    -----------
    source_texture : ndarray
        Source texture image to transfer from
    output_texture : ndarray
        Current state of the output texture
    target_image : ndarray
        Target image that guides the transfer
    target_region : ndarray
        Region of the target image at current position
    current_i, current_j : int
        Current position in the output texture
    block_size : int
        Size of the blocks
    overlap_size : int
        Size of the overlap region
    alpha : float
        Weight between texture matching (alpha) and correspondence matching (1-alpha)
    tolerance : float
        Error tolerance for choosing matching blocks
        
    Returns:
    --------
    ndarray
        The best matching block
    """
    h, w = source_texture.shape[:2]
    candidates = []
    errors = []
    
    # Determine overlap direction
    if current_i == 0 and current_j == 0:
        # First block, no overlap
        direction = None
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
        # Extract a random block from source texture
        i = np.random.randint(0, h - block_size)
        j = np.random.randint(0, w - block_size)
        candidate = source_texture[i:i+block_size, j:j+block_size].copy()
        
        # Compute texture error (overlap constraint)
        texture_error = 0
        if direction:
            texture_error_surface = compute_overlap_error(
                candidate, output_texture, current_i, current_j, 
                block_size, overlap_size, direction
            )
            texture_error = np.sum(texture_error_surface)
        
        # Compute correspondence error with target region
        # Use a simple L2 norm between the luminance of the candidate and target
        if candidate.shape[:2] != target_region.shape[:2]:
            # Resize target region if necessary
            target_region_resized = cv2.resize(target_region, (block_size, block_size))
        else:
            target_region_resized = target_region
            
        candidate_lum = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(float)
        target_lum = cv2.cvtColor(target_region_resized, cv2.COLOR_RGB2GRAY).astype(float)
        correspondence_error = np.sum((candidate_lum - target_lum)**2)
        
        # Combine errors with the weighting parameter alpha
        total_error = alpha * texture_error + (1 - alpha) * correspondence_error
        
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


def texture_transfer(source_texture, target_image, output_size=None, block_size=16, 
                    overlap_size=4, alpha=0.8, iterations=3, tolerance=0.1):
    """Transfer texture from source to target while preserving structure.
    
    Parameters:
    -----------
    source_texture : ndarray
        Texture to be transferred (numpy array)
    target_image : ndarray
        Target image that guides the transfer (numpy array)
    output_size : tuple
        Size of the output texture (height, width). If None, use target_image size
    block_size : int
        Initial block size (will be reduced in subsequent iterations)
    overlap_size : int
        Size of the overlap between blocks
    alpha : float
        Weight between texture matching (alpha) and correspondence matching (1-alpha)
    iterations : int
        Number of refinement iterations
    tolerance : float
        Error tolerance for choosing matching blocks
        
    Returns:
    --------
    ndarray
        Output image with transferred texture
    """
    # Set output size if not specified
    if output_size is None:
        output_size = (target_image.shape[0], target_image.shape[1])
    
    # Resize target image to output size if needed
    if target_image.shape[:2] != output_size:
        target_image = cv2.resize(target_image, (output_size[1], output_size[0]))
    
    # Initialize output texture
    result = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)
    
    # For each iteration (with decreasing block size)
    current_block_size = block_size
    current_overlap_size = overlap_size
    
    for iteration in range(iterations):
        # Calculate number of blocks needed
        num_blocks_h = (output_size[0] - current_overlap_size) // (current_block_size - current_overlap_size) + 1
        num_blocks_w = (output_size[1] - current_overlap_size) // (current_block_size - current_overlap_size) + 1
        
        # Calculate current alpha value (increases with each iteration)
        current_alpha = alpha * (iteration / (iterations - 1) * 0.8 + 0.1) if iterations > 1 else alpha
        
        # Iterate over blocks in raster scan order
        for i in tqdm(range(num_blocks_h), desc=f"Iteration {iteration+1}/{iterations}"):
            for j in range(num_blocks_w):
                # Calculate current position
                current_i = i * (current_block_size - current_overlap_size)
                current_j = j * (current_block_size - current_overlap_size)
                
                # Extract corresponding target region
                target_region = target_image[
                    current_i:min(current_i+current_block_size, output_size[0]), 
                    current_j:min(current_j+current_block_size, output_size[1])
                ]
                
                # Find best matching block considering both texture and correspondence
                best_block = find_best_block_for_transfer(
                    source_texture, result, target_image, target_region,
                    current_i, current_j, current_block_size, current_overlap_size, 
                    current_alpha, tolerance
                )
                
                # Merge the block with minimum error boundary cut
                if i > 0 or j > 0:
                    best_block = merge_blocks_with_cut(
                        best_block, result, current_i, current_j, 
                        current_block_size, current_overlap_size
                    )
                
                # Calculate the actual block size to paste (handle edge cases)
                actual_h = min(current_block_size, output_size[0] - current_i)
                actual_w = min(current_block_size, output_size[1] - current_j)
                
                # Paste the block onto the output texture
                result[current_i:current_i+actual_h, current_j:current_j+actual_w] = \
                    best_block[:actual_h, :actual_w]
        
        # Reduce block size for next iteration
        if iterations > 1:
            current_block_size = max(4, current_block_size // 3 * 2)
            current_overlap_size = max(2, current_overlap_size // 3 * 2)
    
    return result