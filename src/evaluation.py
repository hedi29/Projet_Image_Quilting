"""
Simplified evaluation module for Image Quilting results.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def compute_ssim_patches(original, synthesized, patch_size=64, num_patches=20):
    """
    Compute SSIM between random patches from original and synthesized textures.
    
    Parameters:
    -----------
    original : ndarray
        Original texture
    synthesized : ndarray
        Synthesized texture
    patch_size : int
        Size of patches to compare
    num_patches : int
        Number of random patches to sample
        
    Returns:
    --------
    float
        Average SSIM value
    """
    h_orig, w_orig = original.shape[:2]
    h_synth, w_synth = synthesized.shape[:2]
    
    # Ensure patch size is valid
    patch_size = min(patch_size, h_orig, w_orig, h_synth, w_synth)
    
    if patch_size < 8:  # Minimum reasonable patch size
        return 0.0
    
    ssim_values = []
    
    for _ in range(num_patches):
        # Random patch from original
        i_orig = np.random.randint(0, h_orig - patch_size + 1)
        j_orig = np.random.randint(0, w_orig - patch_size + 1)
        patch_orig = original[i_orig:i_orig+patch_size, j_orig:j_orig+patch_size]
        
        # Random patch from synthesized
        i_synth = np.random.randint(0, h_synth - patch_size + 1)
        j_synth = np.random.randint(0, w_synth - patch_size + 1)
        patch_synth = synthesized[i_synth:i_synth+patch_size, j_synth:j_synth+patch_size]
        
        # Compute SSIM for each channel and average
        channel_ssims = []
        for c in range(3):
            try:
                ssim_val = ssim(patch_orig[:,:,c], patch_synth[:,:,c], 
                               data_range=255, win_size=min(7, patch_size//2*2-1))
                channel_ssims.append(ssim_val)
            except:
                channel_ssims.append(0.0)
        
        ssim_values.append(np.mean(channel_ssims))
    
    return np.mean(ssim_values)


def compute_histogram_distance(original, synthesized):
    """
    Compute histogram distance between textures.
    
    Parameters:
    -----------
    original : ndarray
        Original texture
    synthesized : ndarray
        Synthesized texture
        
    Returns:
    --------
    float
        Average histogram distance across channels
    """
    distances = []
    
    for c in range(3):
        hist_orig = cv2.calcHist([original], [c], None, [64], [0, 256])
        hist_synth = cv2.calcHist([synthesized], [c], None, [64], [0, 256])
        
        # Normalize
        hist_orig = hist_orig.flatten() / hist_orig.sum()
        hist_synth = hist_synth.flatten() / hist_synth.sum()
        
        # Chi-square distance
        distance = 0.5 * np.sum((hist_orig - hist_synth)**2 / (hist_orig + hist_synth + 1e-10))
        distances.append(distance)
    
    return np.mean(distances)


def evaluate_texture_quality(original, synthesized, verbose=True):
    """
    Evaluate synthesized texture quality.
    
    Parameters:
    -----------
    original : ndarray
        Original texture
    synthesized : ndarray
        Synthesized texture
    verbose : bool
        Whether to print results
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Compute metrics
    ssim_score = compute_ssim_patches(original, synthesized)
    hist_distance = compute_histogram_distance(original, synthesized)
    
    # Simple visual consistency check (edge preservation)
    orig_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), 50, 150)
    synth_edges = cv2.Canny(cv2.cvtColor(synthesized, cv2.COLOR_RGB2GRAY), 50, 150)
    
    orig_edge_density = np.sum(orig_edges > 0) / orig_edges.size
    synth_edge_density = np.sum(synth_edges > 0) / synth_edges.size
    edge_consistency = 1.0 - abs(orig_edge_density - synth_edge_density)
    
    results = {
        'ssim': ssim_score,
        'histogram_distance': hist_distance,
        'edge_consistency': edge_consistency,
        'overall_score': (ssim_score + edge_consistency) / 2 - hist_distance / 10
    }
    
    if verbose:
        print("\n" + "="*50)
        print("TEXTURE QUALITY EVALUATION")
        print("="*50)
        print(f"SSIM Score:           {ssim_score:.4f} (higher is better)")
        print(f"Histogram Distance:   {hist_distance:.4f} (lower is better)")
        print(f"Edge Consistency:     {edge_consistency:.4f} (higher is better)")
        print(f"Overall Score:        {results['overall_score']:.4f}")
        print("="*50)
    
    return results


def compare_parameters(input_texture, parameter_sets, output_size=(200, 200)):
    """
    Compare different parameter combinations.
    
    Parameters:
    -----------
    input_texture : ndarray
        Input texture
    parameter_sets : list
        List of dictionaries with parameters to test
    output_size : tuple
        Output size for synthesis
        
    Returns:
    --------
    dict
        Results for each parameter set
    """
    from .quilting import ImageQuilting
    
    results = {}
    
    print("Comparing parameter sets...")
    for i, params in enumerate(parameter_sets):
        print(f"\nTesting parameter set {i+1}: {params}")
        
        # Create quilter with these parameters
        quilter = ImageQuilting(**params)
        
        # Synthesize texture
        synthesized = quilter.synthesize_texture(input_texture, output_size)
        
        # Evaluate quality
        quality = evaluate_texture_quality(input_texture, synthesized, verbose=False)
        
        results[f"params_{i+1}"] = {
            'parameters': params,
            'synthesized': synthesized,
            'quality': quality
        }
        
        print(f"Overall Score: {quality['overall_score']:.4f}")
    
    # Find best parameters
    best_key = max(results.keys(), key=lambda k: results[k]['quality']['overall_score'])
    best_params = results[best_key]['parameters']
    best_score = results[best_key]['quality']['overall_score']
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    
    return results


# Example parameter comparison
if __name__ == "__main__":
    # Example usage with test texture
    print("Creating test texture for parameter comparison...")
    
    # Create a more structured test texture
    test_texture = np.zeros((80, 80, 3), dtype=np.uint8)
    
    # Create a brick-like pattern
    for i in range(80):
        for j in range(80):
            # Brick pattern
            brick_h = 10
            brick_w = 20
            
            row = i // brick_h
            col = j // brick_w
            
            # Offset every other row
            if row % 2 == 1:
                col = (j + brick_w//2) // brick_w
            
            # Brick color variation
            if (row + col) % 2 == 0:
                test_texture[i, j] = [180, 120, 80]  # Brown brick
            else:
                test_texture[i, j] = [160, 100, 60]  # Darker brown
            
            # Mortar lines
            if i % brick_h == 0 or j % brick_w == 0:
                test_texture[i, j] = [200, 200, 200]  # Light gray mortar
    
    # Test different parameter combinations
    parameter_sets = [
        {'block_size': 20, 'overlap_ratio': 1/6, 'tolerance': 0.1},
        {'block_size': 30, 'overlap_ratio': 1/6, 'tolerance': 0.1},
        {'block_size': 20, 'overlap_ratio': 1/4, 'tolerance': 0.1},
        {'block_size': 30, 'overlap_ratio': 1/4, 'tolerance': 0.15},
    ]
    
    results = compare_parameters(test_texture, parameter_sets)
    
    # Visualize best result
    best_result = max(results.values(), key=lambda x: x['quality']['overall_score'])
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(test_texture)
    axes[0].set_title('Original Test Texture')
    axes[0].axis('off')
    
    axes[1].imshow(best_result['synthesized'])
    axes[1].set_title(f"Best Result\nScore: {best_result['quality']['overall_score']:.3f}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()