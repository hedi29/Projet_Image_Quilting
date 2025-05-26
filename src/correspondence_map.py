"""
Simplified correspondence maps - Essential for texture transfer.

This minimal module provides only the essential correspondence maps
needed for effective texture transfer, without the complexity.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def compute_luminance_map(image: np.ndarray) -> np.ndarray:
    """
    Compute luminance correspondence map (most important).
    
    This is the core correspondence map that works for 90% of cases.
    
    Parameters:
    -----------
    image : ndarray
        Input image (H, W, 3) or (H, W)
        
    Returns:
    --------
    ndarray
        Luminance map (H, W, 3) - replicated across channels for consistency
    """
    if image.ndim == 3 and image.shape[2] == 3:
        # Standard RGB to luminance conversion
        luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    elif image.ndim == 2: # Grayscale image
        luminance = image.copy()
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}. Expected (H, W, 3) or (H, W).")

    # Normalize to [0, 255] range
    min_val, max_val = luminance.min(), luminance.max()
    if max_val - min_val < 1e-8: # Avoid division by zero for flat images
        luminance = np.zeros_like(luminance, dtype=np.uint8)
    else:
        luminance = ((luminance - min_val) / 
                     (max_val - min_val) * 255).astype(np.uint8)
    
    # Replicate to 3 channels for compatibility with RGB patches
    return np.stack([luminance, luminance, luminance], axis=2)


def compute_blurred_luminance_map(image: np.ndarray, blur_sigma: float = 3.0) -> np.ndarray:
    """
    Compute blurred luminance map for smoother transfers.
    
    Useful when you want smoother, less detailed correspondence.
    Good for artistic effects or when target image has noise.
    
    Parameters:
    -----------
    image : ndarray
        Input image (H, W, 3) or (H, W)
    blur_sigma : float
        Gaussian blur sigma (higher = smoother)
        
    Returns:
    --------
    ndarray
        Blurred luminance map (H, W, 3)
    """
    # Get basic luminance (single channel for blurring)
    if image.ndim == 3 and image.shape[2] == 3:
        luminance_single_channel = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    elif image.ndim == 2:
        luminance_single_channel = image.copy()
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}. Expected (H, W, 3) or (H, W).")

    luminance_float = luminance_single_channel.astype(np.float32)
    
    # Apply Gaussian blur
    blurred = gaussian_filter(luminance_float, sigma=blur_sigma)
    
    # Normalize
    min_val, max_val = blurred.min(), blurred.max()
    if max_val - min_val < 1e-8:
        blurred_norm = np.zeros_like(blurred, dtype=np.uint8)
    else:
        blurred_norm = ((blurred - min_val) / 
                       (max_val - min_val) * 255).astype(np.uint8)
    
    return np.stack([blurred_norm, blurred_norm, blurred_norm], axis=2)


def get_correspondence_map(image: np.ndarray, map_type: str = 'luminance', blur_sigma: float = 3.0) -> np.ndarray:
    """
    Get correspondence map - simplified interface.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    map_type : str
        'luminance' (default, works best) or 'blurred'
    blur_sigma : float
        Gaussian blur sigma, only used if map_type is 'blurred'
        
    Returns:
    --------
    ndarray
        Correspondence map
    """
    if map_type == 'luminance':
        return compute_luminance_map(image)
    elif map_type == 'blurred':
        return compute_blurred_luminance_map(image, blur_sigma=blur_sigma)
    else:
        print(f"⚠️  Unknown map type '{map_type}', using luminance map as default.")
        return compute_luminance_map(image)


# Test the simplified correspondence
if __name__ == "__main__":
    # Create test image (RGB)
    test_image_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    # Add some structure
    for i in range(100):
        for j in range(100):
            if (i // 20 + j // 20) % 2 == 0:
                test_image_rgb[i, j] = [200, 150, 100]
    
    # Create test image (Grayscale)
    test_image_gray = np.mean(test_image_rgb, axis=2).astype(np.uint8)

    print("Testing RGB image:")
    lum_map_rgb = get_correspondence_map(test_image_rgb, 'luminance')
    blur_map_rgb = get_correspondence_map(test_image_rgb, 'blurred', blur_sigma=2.0)
    
    print("✅ Simplified correspondence maps for RGB image working!")
    print(f"   Luminance map shape: {lum_map_rgb.shape}, dtype: {lum_map_rgb.dtype}")
    assert lum_map_rgb.shape == (100,100,3)
    assert lum_map_rgb.dtype == np.uint8
    print(f"   Blurred map shape: {blur_map_rgb.shape}, dtype: {blur_map_rgb.dtype}")
    assert blur_map_rgb.shape == (100,100,3)
    assert blur_map_rgb.dtype == np.uint8

    print("\nTesting Grayscale image:")
    lum_map_gray = get_correspondence_map(test_image_gray, 'luminance')
    blur_map_gray = get_correspondence_map(test_image_gray, 'blurred', blur_sigma=2.0)

    print("✅ Simplified correspondence maps for Grayscale image working!")
    print(f"   Luminance map shape: {lum_map_gray.shape}, dtype: {lum_map_gray.dtype}")
    assert lum_map_gray.shape == (100,100,3)
    assert lum_map_gray.dtype == np.uint8
    print(f"   Blurred map shape: {blur_map_gray.shape}, dtype: {blur_map_gray.dtype}")
    assert blur_map_gray.shape == (100,100,3)
    assert blur_map_gray.dtype == np.uint8

    # Test with a flat image
    flat_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
    flat_lum_map = get_correspondence_map(flat_image, 'luminance')
    print("\nTesting flat image:")
    print(f"   Luminance map for flat image shape: {flat_lum_map.shape}")
    assert np.all(flat_lum_map == 0) # Should be all zeros due to normalization of flat input

    # Display images if OpenCV is available and not in a headless environment
    try:
        cv2.imshow("Test RGB", test_image_rgb)
        cv2.imshow("Luminance Map RGB", lum_map_rgb)
        cv2.imshow("Blurred Luminance Map RGB", blur_map_rgb)
        
        cv2.imshow("Test Gray", test_image_gray)
        cv2.imshow("Luminance Map Gray", lum_map_gray)
        cv2.imshow("Blurred Luminance Map Gray", blur_map_gray)
        
        print("\nDisplaying test images and maps. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"Could not display images (likely headless environment or OpenCV issue): {e}")

    print("All tests passed!") 