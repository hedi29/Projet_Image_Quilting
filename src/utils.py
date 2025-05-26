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


def load_texture(path):
    """Load a texture image from file.
    
    Parameters:
    -----------
    path : str
        Path to the texture image file
        
    Returns:
    --------
    ndarray
        Texture image as numpy array with shape (height, width, channels)
    """
    try:
        image = Image.open(path)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Could not load texture from {path}: {e}")


def save_image(image, path):
    """Save an image array to file.
    
    Parameters:
    -----------
    image : ndarray
        Image array with shape (height, width, channels)
    path : str
        Output file path
    """
    try:
        # Ensure the image is in the correct format
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Create directory if it doesn't exist and path has a directory component
        dir_path = os.path.dirname(path)
        if dir_path:  # Only create directory if there is one
            os.makedirs(dir_path, exist_ok=True)
        
        # Save image
        Image.fromarray(image).save(path)
    except Exception as e:
        raise ValueError(f"Could not save image to {path}: {e}")


def visualize_results(original_texture, synthesized_texture, title=None, save_path=None):
    """Visualize original and synthesized textures side by side.
    
    Parameters:
    -----------
    original_texture : ndarray
        Original texture image
    synthesized_texture : ndarray
        Synthesized texture image
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