"""
Utility functions for the Image Quilting project.

This module provides helper functions for loading, saving, and visualizing
textures and synthesis results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


def load_texture(path: str) -> np.ndarray:
    """Loads a texture image from a file path into a NumPy array (RGB).

    Args:
        path: Path to the texture image file.

    Returns:
        Texture image as a NumPy array (H, W, 3), RGB format.

    Raises:
        ValueError: If the image cannot be loaded.
    """
    try:
        image = Image.open(path)
        if image.mode != 'RGB': # Ensure image is in RGB format
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Could not load texture from {path}: {e}")


def save_image(image: np.ndarray, path: str):
    """Saves a NumPy image array to a file.

    Args:
        image: NumPy array representing the image (H, W, 3 or H, W).
        path: Output file path.

    Raises:
        ValueError: If the image cannot be saved.
    """
    try:
        # Ensure image data is uint8 and clipped to 0-255 range
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Create output directory if it doesn't exist
        dir_path = os.path.dirname(path)
        if dir_path: # Only if path includes a directory
            os.makedirs(dir_path, exist_ok=True)
        
        Image.fromarray(image).save(path)
    except Exception as e:
        raise ValueError(f"Could not save image to {path}: {e}")


def visualize_results(original_texture: np.ndarray, synthesized_texture: np.ndarray, 
                      title: str = None, save_path: str = None) -> np.ndarray:
    """Visualizes original and synthesized textures side-by-side using Matplotlib.

    Args:
        original_texture: The original input texture.
        synthesized_texture: The generated (quilted) texture.
        title: Optional title for the entire visualization.
        save_path: Optional path to save the visualization image.

    Returns:
        A NumPy array representing the visualization image (RGB).
    """
    fig = plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_texture)
    plt.title('Original Texture')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(synthesized_texture)
    plt.title('Synthesized Texture')
    plt.axis('off')
    
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout() # Adjust subplot params for a tight layout.
    
    if save_path:
        # Ensure output directory for save_path exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
             os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Convert Matplotlib figure to a NumPy array
    fig.canvas.draw()
    try:
        # Standard way for RGB
        img_buffer = fig.canvas.tostring_rgb()
        img = np.frombuffer(img_buffer, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except AttributeError:
        # Fallback for some backends (e.g., FigureCanvasMac) that might use ARGB
        img_buffer = fig.canvas.tostring_argb()
        img = np.frombuffer(img_buffer, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, 1:4] # Convert ARGB to RGB, taking [R,G,B] channels
    
    plt.close(fig) # Close the figure to free memory
    return img


def visualize_min_cut(block1: np.ndarray, block2: np.ndarray, overlap: int, 
                        cut_mask: np.ndarray, direction: str = 'vertical') -> np.ndarray:
    """Visualizes the minimum cut between two overlapping blocks.

    Args:
        block1: The first block (e.g., existing block in output).
        block2: The second block (e.g., candidate block).
        overlap: Size of the overlap region in pixels.
        cut_mask: Boolean mask defining the cut (True=use block1, False=use block2, or vice-versa depending on convention).
                  This function assumes the mask aligns with error_surface: if mask[r,c] is True, 
                  it implies block1's pixel is chosen at that point in the overlap.
        direction: 'vertical' or 'horizontal' cut.

    Returns:
        A NumPy array representing the visualization image (RGB).
    """
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 4, figure=fig) # Grid for 4 subplots
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(block1)
    ax1.set_title('Block 1 (Existing/Output)')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(block2)
    ax2.set_title('Block 2 (Candidate)')
    ax2.axis('off')
    
    # Display absolute difference in the overlap region
    ax3 = fig.add_subplot(gs[0, 2])
    if direction == 'vertical':
        # Vertical cut implies overlap is on the left of block2 and right of block1
        overlap_b1 = block1[:, -overlap:] 
        overlap_b2 = block2[:, :overlap]
        # Ensure shapes match for diff calculation, common for boundary conditions
        h = min(overlap_b1.shape[0], overlap_b2.shape[0])
        w = min(overlap_b1.shape[1], overlap_b2.shape[1])
        diff_region = np.abs(overlap_b1[:h, :w].astype(float) - overlap_b2[:h, :w].astype(float)).astype(np.uint8)
        ax3.imshow(diff_region)
        ax3.set_title(f'Abs Diff (Vertical Overlap: {w}px)')
    else:  # horizontal cut
        overlap_b1 = block1[-overlap:, :] 
        overlap_b2 = block2[:overlap, :]
        h = min(overlap_b1.shape[0], overlap_b2.shape[0])
        w = min(overlap_b1.shape[1], overlap_b2.shape[1])
        diff_region = np.abs(overlap_b1[:h, :w].astype(float) - overlap_b2[:h, :w].astype(float)).astype(np.uint8)
        ax3.imshow(diff_region)
        ax3.set_title(f'Abs Diff (Horizontal Overlap: {h}px)')
    ax3.axis('off')
    
    # Display merged result based on the cut_mask
    ax4 = fig.add_subplot(gs[0, 3])
    # Create a result block, typically starting with block2 and filling in from block1 based on mask
    # The mask convention from _find_vertical_cut is True for new block (block2 here if it's the candidate)
    # and False for existing (block1).
    # So, if mask[r,c] is True, result[r,c] = block2[r,c] in overlap
    # If mask[r,c] is False, result[r,c] = block1[r,c] in overlap
    
    # Let's assume `cut_mask` means True for block2 (candidate), False for block1 (output_texture region)
    # This aligns with how _find_vertical_cut mask is interpreted for _apply_min_cut.
    # Result starts as candidate, and we overwrite with block1 where cut_mask is False.
    result_viz = block2.copy() # Start with candidate block
    h_ov, w_ov = cut_mask.shape[:2]

    if direction == 'vertical':
        # Overlap region is block2[:, :overlap]
        # block1 provides pixels for block2[:, :overlap] where cut_mask is False
        for r in range(min(block1.shape[0], block2.shape[0], h_ov)):
            for c in range(min(overlap, block1.shape[1], block2.shape[1], w_ov)):
                if not cut_mask[r, c]: # If cut says use existing (block1)
                    result_viz[r, c] = block1[r, c] # Use corresponding pixel from block1
    else:  # horizontal
        # Overlap region is block2[:overlap, :]
        for r in range(min(overlap, block1.shape[0], block2.shape[0], h_ov)):
            for c in range(min(block1.shape[1], block2.shape[1], w_ov)):
                if not cut_mask[r, c]: # If cut says use existing (block1)
                    result_viz[r, c] = block1[r, c]
    
    ax4.imshow(result_viz)
    ax4.set_title('Result with Min Cut')
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Convert figure to NumPy array
    fig.canvas.draw()
    try:
        img_buffer = fig.canvas.tostring_rgb()
        img = np.frombuffer(img_buffer, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except AttributeError:
        img_buffer = fig.canvas.tostring_argb()
        img = np.frombuffer(img_buffer, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, 1:4] # ARGB to RGB
    
    plt.close(fig)
    return img