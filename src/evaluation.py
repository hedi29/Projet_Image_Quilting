"""
Evaluation metrics for texture synthesis quality assessment.

This module provides functions to evaluate the quality of synthesized textures
using various metrics such as SSIM, MSE, and patch-based comparison.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def compute_ssim(original_texture, synthesized_texture, patch_size=64, num_patches=20):
    """Compute SSIM between patches of original and synthesized textures.
    
    Parameters:
    -----------
    original_texture : ndarray
        Original input texture
    synthesized_texture : ndarray
        Synthesized output texture
    patch_size : int
        Size of patches to compare
    num_patches : int
        Number of random patches to sample
        
    Returns:
    --------
    float
        Average SSIM value across all patches and color channels
    """
    h_orig, w_orig = original_texture.shape[:2]
    h_synth, w_synth = synthesized_texture.shape[:2]
    
    # Ensure patch size is not larger than the textures
    patch_size = min(patch_size, h_orig, w_orig, h_synth, w_synth)
    
    ssim_values = []
    
    for _ in range(num_patches):
        # Extract random patch from original texture
        i_orig = np.random.randint(0, h_orig - patch_size)
        j_orig = np.random.randint(0, w_orig - patch_size)
        patch_orig = original_texture[i_orig:i_orig+patch_size, j_orig:j_orig+patch_size]
        
        # Extract random patch from synthesized texture
        i_synth = np.random.randint(0, h_synth - patch_size)
        j_synth = np.random.randint(0, w_synth - patch_size)
        patch_synth = synthesized_texture[i_synth:i_synth+patch_size, j_synth:j_synth+patch_size]
        
        # Compute SSIM for each channel
        ssim_rgb = []
        for c in range(3):
            ssim_value = ssim(patch_orig[:,:,c], patch_synth[:,:,c], data_range=255)
            ssim_rgb.append(ssim_value)
        
        # Average SSIM across channels
        ssim_values.append(np.mean(ssim_rgb))
    
    return np.mean(ssim_values)


def compute_mse(original_texture, synthesized_texture, patch_size=64, num_patches=20):
    """Compute Mean Squared Error between patches of original and synthesized textures.
    
    Parameters:
    -----------
    original_texture : ndarray
        Original input texture
    synthesized_texture : ndarray
        Synthesized output texture
    patch_size : int
        Size of patches to compare
    num_patches : int
        Number of random patches to sample
        
    Returns:
    --------
    float
        Average MSE value across all patches
    """
    h_orig, w_orig = original_texture.shape[:2]
    h_synth, w_synth = synthesized_texture.shape[:2]
    
    # Ensure patch size is not larger than the textures
    patch_size = min(patch_size, h_orig, w_orig, h_synth, w_synth)
    
    mse_values = []
    
    for _ in range(num_patches):
        # Extract random patch from original texture
        i_orig = np.random.randint(0, h_orig - patch_size)
        j_orig = np.random.randint(0, w_orig - patch_size)
        patch_orig = original_texture[i_orig:i_orig+patch_size, j_orig:j_orig+patch_size].astype(float)
        
        # Find best matching patch in synthesized texture
        best_mse = float('inf')
        
        # Sample random locations in synthesized texture
        for _ in range(10):
            i_synth = np.random.randint(0, h_synth - patch_size)
            j_synth = np.random.randint(0, w_synth - patch_size)
            patch_synth = synthesized_texture[i_synth:i_synth+patch_size, j_synth:j_synth+patch_size].astype(float)
            
            # Compute MSE
            mse = np.mean((patch_orig - patch_synth)**2)
            best_mse = min(best_mse, mse)
        
        mse_values.append(best_mse)
    
    return np.mean(mse_values)


def compute_histogram_distance(original_texture, synthesized_texture, num_bins=50):
    """Compute color histogram distance between original and synthesized textures.
    
    Parameters:
    -----------
    original_texture : ndarray
        Original input texture
    synthesized_texture : ndarray
        Synthesized output texture
    num_bins : int
        Number of bins for histogram computation
        
    Returns:
    --------
    float
        Histogram distance (chi-square distance)
    """
    distance = 0
    
    # Compute histogram for each channel
    for c in range(3):
        hist_orig = cv2.calcHist([original_texture], [c], None, [num_bins], [0, 256])
        hist_synth = cv2.calcHist([synthesized_texture], [c], None, [num_bins], [0, 256])
        
        # Normalize histograms
        hist_orig = cv2.normalize(hist_orig, hist_orig).flatten()
        hist_synth = cv2.normalize(hist_synth, hist_synth).flatten()
        
        # Compute chi-square distance
        chi_square = 0.5 * np.sum(((hist_orig - hist_synth)**2) / (hist_orig + hist_synth + 1e-10))
        distance += chi_square
    
    return distance / 3.0  # Average across channels


def visualize_patch_comparison(original_texture, synthesized_texture, patch_size=64, num_patches=5):
    """Visualize random patches from original and synthesized textures for visual comparison.
    
    Parameters:
    -----------
    original_texture : ndarray
        Original input texture
    synthesized_texture : ndarray
        Synthesized output texture
    patch_size : int
        Size of patches to compare
    num_patches : int
        Number of random patches to visualize
        
    Returns:
    --------
    ndarray
        Visualization image showing patch comparisons
    """
    h_orig, w_orig = original_texture.shape[:2]
    h_synth, w_synth = synthesized_texture.shape[:2]
    
    # Ensure patch size is not larger than the textures
    patch_size = min(patch_size, h_orig, w_orig, h_synth, w_synth)
    
    # Create figure
    fig, axes = plt.subplots(num_patches, 2, figsize=(10, num_patches*5))
    
    for i in range(num_patches):
        # Extract random patch from original texture
        i_orig = np.random.randint(0, h_orig - patch_size)
        j_orig = np.random.randint(0, w_orig - patch_size)
        patch_orig = original_texture[i_orig:i_orig+patch_size, j_orig:j_orig+patch_size]
        
        # Extract random patch from synthesized texture
        i_synth = np.random.randint(0, h_synth - patch_size)
        j_synth = np.random.randint(0, w_synth - patch_size)
        patch_synth = synthesized_texture[i_synth:i_synth+patch_size, j_synth:j_synth+patch_size]
        
        # Display patches
        axes[i, 0].imshow(cv2.cvtColor(patch_orig, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Original Patch {i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(cv2.cvtColor(patch_synth, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f'Synthesized Patch {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    # Convert figure to image
    fig.canvas.draw()
    comparison_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    comparison_img = comparison_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return comparison_img


def evaluate_synthesis_quality(original_texture, synthesized_texture, patch_size=64, 
                               num_patches=20, visualize=False):
    """Evaluate the quality of texture synthesis using multiple metrics.
    
    Parameters:
    -----------
    original_texture : ndarray
        Original input texture
    synthesized_texture : ndarray
        Synthesized output texture
    patch_size : int
        Size of patches to compare
    num_patches : int
        Number of random patches to sample
    visualize : bool
        Whether to generate visualization of patch comparisons
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Compute SSIM
    ssim_value = compute_ssim(original_texture, synthesized_texture, 
                              patch_size, num_patches)
    
    # Compute MSE
    mse_value = compute_mse(original_texture, synthesized_texture, 
                           patch_size, num_patches)
    
    # Compute histogram distance
    hist_distance = compute_histogram_distance(original_texture, synthesized_texture)
    
    # Prepare results
    results = {
        'ssim': ssim_value,
        'mse': mse_value,
        'histogram_distance': hist_distance
    }
    
    # Generate visualization if requested
    if visualize:
        comparison_img = visualize_patch_comparison(original_texture, synthesized_texture,
                                                   patch_size, min(5, num_patches))
        results['visualization'] = comparison_img
    
    return results