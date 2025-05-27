"""Image Quilting for Texture Synthesis.

This package implements the Efros & Freeman Image Quilting algorithm
for texture synthesis and supports elements of texture transfer.

Core classes and functions are exposed for use.
"""

__version__ = '0.1.0'
__author__ = 'Your Name' # TODO: Update with actual author name

# Core class for quilting
from .quilting import ImageQuilting

# Utility functions
from .utils import (
    load_texture,
    save_image,
    visualize_results
)

# Public API exposed by `from src import *`
__all__ = [
    'ImageQuilting',
    'load_texture',
    'save_image',
    'visualize_results'
]