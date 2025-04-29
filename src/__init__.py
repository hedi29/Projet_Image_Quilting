"""
Image Quilting for Texture Synthesis and Transfer

This package implements the algorithm described in the paper:
'Image Quilting for Texture Synthesis and Transfer' by Alexei A. Efros and William T. Freeman.

The implementation provides functionality for:
- Texture synthesis using image quilting
- Texture transfer from one image to another
- Evaluation metrics for synthesized textures
"""

__version__ = '0.1.0'
__author__ = 'Your Name'

# Import core functionality
from .quilting import (
    synthesize_texture,
    random_patch,
    minimum_error_boundary_cut
)

from .transfer import (
    texture_transfer,
    find_best_block_for_transfer
)

from .evaluation import (
    evaluate_synthesis_quality,
    compute_ssim
)

from .utils import (
    load_texture,
    save_image,
    visualize_results
)

# Define public API
__all__ = [
    'synthesize_texture',
    'texture_transfer',
    'evaluate_synthesis_quality',
    'load_texture',
    'save_image',
    'visualize_results',
    'random_patch',
    'minimum_error_boundary_cut',
    'compute_ssim'
]