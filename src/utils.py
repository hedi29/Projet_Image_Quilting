"""
Utility functions for the Image Quilting project.

This module contains helper functions for loading/saving images,
visualizing results, and other utility functions.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_texture(path):
    """Load an input texture image.
    
    Parameters:
    -----------
    path : str
        Path to the image file
        
    Returns:
    --------
    ndarray
        RGB image as a numpy array
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image, path):
    """Save an image to disk.
    
    Parameters:
    -----------
    image : ndarray
        RGB image as a numpy array
    path : str
        Path to save the image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert from RGB to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)


def visualize_results(original_texture, synthesized_texture, title=None, save_path=None):
    """Visualize original and synthesized textures side by side.
    
    Parameters:
    -----------
    original_texture : ndarray
        Original input texture
    synthesized_texture : ndarray
        Synthesized output texture
    title : str, optional
        Title for the visualization
    save_path : str, optional
        Path to save the visualization image
        
    Returns:
    --------
    ndarray
        Visualization image
    """
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Display original texture
    plt.subplot(1, 2, 1)
    plt.imshow(original_texture)
    plt.title('Original Texture')
    plt.axis('off')
    
    # Display synthesized texture
    plt.subplot(1, 2, 2)
    plt.imshow(synthesized_texture)
    plt.title('Synthesized Texture')
    plt.axis('off')
    
    # Set main title
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Convert figure to image - update to use tostring_argb instead of tostring_rgb
    fig = plt.gcf()
    fig.canvas.draw()
    # Fix for FigureCanvasMac 
    try:
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    except AttributeError:
        # Use tostring_argb and then rearrange the colors
        buf = fig.canvas.tostring_argb()
        img = np.frombuffer(buf, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert ARGB to RGB
        img = img[:, :, 1:] 
    else:
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    return img


def visualize_transfer_results(source_texture, target_image, transferred_image, title=None, save_path=None):
    """Visualize texture transfer results.
    
    Parameters:
    -----------
    source_texture : ndarray
        Source texture image
    target_image : ndarray
        Target image that guided the transfer
    transferred_image : ndarray
        Result of texture transfer
    title : str, optional
        Title for the visualization
    save_path : str, optional
        Path to save the visualization image
        
    Returns:
    --------
    ndarray
        Visualization image
    """
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Display source texture
    plt.subplot(1, 3, 1)
    plt.imshow(source_texture)
    plt.title('Source Texture')
    plt.axis('off')
    
    # Display target image
    plt.subplot(1, 3, 2)
    plt.imshow(target_image)
    plt.title('Target Image')
    plt.axis('off')
    
    # Display transferred result
    plt.subplot(1, 3, 3)
    plt.imshow(transferred_image)
    plt.title('Texture Transfer Result')
    plt.axis('off')
    
    # Set main title
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Convert figure to image - update to use tostring_argb instead of tostring_rgb
    fig = plt.gcf()
    fig.canvas.draw()
    # Fix for FigureCanvasMac 
    try:
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    except AttributeError:
        # Use tostring_argb and then rearrange the colors
        buf = fig.canvas.tostring_argb()
        img = np.frombuffer(buf, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert ARGB to RGB
        img = img[:, :, 1:] 
    else:
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    return img


def visualize_min_cut(block1, block2, overlap, mask, direction='vertical'):
    """Visualize the minimum cut between two overlapping blocks.
    
    Parameters:
    -----------
    block1 : ndarray
        First block
    block2 : ndarray
        Second block
    overlap : int
        Size of overlap region
    mask : ndarray
        Binary mask defining the cut
    direction : str
        'vertical' or 'horizontal' cut direction
    
    Returns:
    --------
    ndarray
        Visualization image
    """
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 4, figure=fig)
    
    # Display first block
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(block1)
    ax1.set_title('Block 1')
    ax1.axis('off')
    
    # Display second block
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(block2)
    ax2.set_title('Block 2')
    ax2.axis('off')
    
    # Display overlap region
    ax3 = fig.add_subplot(gs[0, 2])
    if direction == 'vertical':
        error_region = np.abs(block1[:, :overlap] - block2[:, :overlap])
        ax3.imshow(error_region)
        ax3.set_title('Overlap Error')
    else:  # horizontal
        error_region = np.abs(block1[:overlap, :] - block2[:overlap, :])
        ax3.imshow(error_region)
        ax3.set_title('Overlap Error')
    ax3.axis('off')
    
    # Display merged result with min cut
    ax4 = fig.add_subplot(gs[0, 3])
    result = block2.copy()
    
    if direction == 'vertical':
        for i in range(block1.shape[0]):
            for j in range(overlap):
                if mask[i, j]:
                    result[i, j] = block1[i, j]
    else:  # horizontal
        for i in range(overlap):
            for j in range(block1.shape[1]):
                if mask[i, j]:
                    result[i, j] = block1[i, j]
    
    ax4.imshow(result)
    ax4.set_title('Result with Min Cut')
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Convert figure to image - update to use tostring_argb instead of tostring_rgb
    fig.canvas.draw()
    # Fix for FigureCanvasMac 
    try:
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    except AttributeError:
        # Use tostring_argb and then rearrange the colors
        buf = fig.canvas.tostring_argb()
        img = np.frombuffer(buf, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert ARGB to RGB
        img = img[:, :, 1:] 
    else:
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img


def parameter_study(input_texture, block_sizes, overlap_ratios, output_size=(200, 200)):
    """Perform a parameter study to analyze the effect of block size and overlap.
    
    Parameters:
    -----------
    input_texture : ndarray
        Input texture image
    block_sizes : list
        List of block sizes to test
    overlap_ratios : list
        List of overlap ratios to test (as a fraction of block size)
    output_size : tuple
        Size of the output texture for each test
        
    Returns:
    --------
    dict
        Dictionary containing results of the parameter study
    """
    from .quilting import synthesize_texture
    from .evaluation import evaluate_synthesis_quality
    
    results = {}
    
    for block_size in block_sizes:
        block_results = {}
        for overlap_ratio in overlap_ratios:
            overlap_size = max(1, int(block_size * overlap_ratio))
            
            # Synthesize texture with current parameters
            synthesized = synthesize_texture(
                input_texture, output_size[0], output_size[1], 
                block_size, overlap_size
            )
            
            # Evaluate quality
            quality = evaluate_synthesis_quality(input_texture, synthesized)
            
            # Record results
            block_results[overlap_ratio] = {
                'synthesized': synthesized,
                'metrics': quality,
                'parameters': {
                    'block_size': block_size,
                    'overlap_size': overlap_size,
                    'overlap_ratio': overlap_ratio
                }
            }
        
        results[block_size] = block_results
    
    return results


def visualize_parameter_study(study_results, save_path=None):
    """Visualize the results of a parameter study.
    
    Parameters:
    -----------
    study_results : dict
        Dictionary containing results of the parameter study
    save_path : str, optional
        Path to save the visualization image
        
    Returns:
    --------
    ndarray
        Visualization image
    """
    block_sizes = list(study_results.keys())
    overlap_ratios = list(study_results[block_sizes[0]].keys())
    
    # Create figure
    n_rows = len(block_sizes)
    n_cols = len(overlap_ratios)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    
    # Plot results
    for i, block_size in enumerate(block_sizes):
        for j, overlap_ratio in enumerate(overlap_ratios):
            result = study_results[block_size][overlap_ratio]
            synthesized = result['synthesized']
            ssim = result['metrics']['ssim']
            
            if n_rows == 1 and n_cols == 1:
                ax = axes
            elif n_rows == 1:
                ax = axes[j]
            elif n_cols == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]
            
            ax.imshow(synthesized)
            ax.set_title(f'Block: {block_size}, Ratio: {overlap_ratio:.2f}\nSSIM: {ssim:.3f}')
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Convert figure to image - update to use tostring_argb instead of tostring_rgb
    fig.canvas.draw()
    # Fix for FigureCanvasMac 
    try:
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    except AttributeError:
        # Use tostring_argb and then rearrange the colors
        buf = fig.canvas.tostring_argb()
        img = np.frombuffer(buf, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert ARGB to RGB
        img = img[:, :, 1:] 
    else:
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img