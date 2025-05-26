"""
Image Quilting for Texture Synthesis

This package implements the algorithm described in the paper:
'Image Quilting for Texture Synthesis and Transfer' by Alexei A. Efros and William T. Freeman.

The implementation provides functionality for:
- Texture synthesis using image quilting
"""

__version__ = '0.1.0'
__author__ = 'Your Name'

# Import core functionality
from .quilting import ImageQuilting

from .utils import (
    load_texture,
    save_image,
    visualize_results
)

# Define public API
__all__ = [
    'ImageQuilting',
    'load_texture',
    'save_image',
    'visualize_results'
]