"""
Main script for Image Quilting texture synthesis.
Usage: python main.py --input texture.jpg --output result.png
"""

import argparse
import time
import os
import numpy as np
from src import ImageQuilting, load_texture, save_image, visualize_results

def main():
    parser = argparse.ArgumentParser(description='Image Quilting Texture Synthesis')
    parser.add_argument('--input', type=str, help='Input texture image path')
    parser.add_argument('--output', type=str, default='output.png', help='Output image path')
    parser.add_argument('--block-size', type=int, default=32, help='Block size')
    parser.add_argument('--overlap-ratio', type=float, default=1/6, help='Overlap ratio')
    parser.add_argument('--tolerance', type=float, default=0.1, help='Error tolerance')
    parser.add_argument('--output-width', type=int, default=None, help='Output width')
    parser.add_argument('--output-height', type=int, default=None, help='Output height')

    args = parser.parse_args()

    print(f"Loading texture from: {args.input}")
    try:
        input_texture = load_texture(args.input)
    except Exception as e:
        print(f"Error loading texture: {e}")
        # Fallback or exit might be needed here depending on desired behavior
        return 1 

    print(f"Input texture shape: {input_texture.shape}")

    # Determine output size, default to 2x input size if not specified
    output_width = args.output_width if args.output_width is not None else input_texture.shape[1] * 2
    output_height = args.output_height if args.output_height is not None else input_texture.shape[0] * 2
    output_size = (output_height, output_width)
    print(f"Output size: {output_size}")

    quilter = ImageQuilting(
        block_size=args.block_size,
        overlap_ratio=args.overlap_ratio,
        tolerance=args.tolerance
    )

    print(f"Algorithm parameters:")
    print(f"  Block size: {quilter.block_size}")
    print(f"  Overlap: {quilter.overlap}")
    print(f"  Tolerance: {quilter.tolerance}")

    print("\nStarting texture synthesis...")
    start_time = time.time()

    try:
        result = quilter.synthesize_texture(input_texture, output_size)
        synthesis_time = time.time() - start_time
        print(f"Synthesis completed in {synthesis_time:.2f} seconds")

        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_image(result, args.output)
        print(f"Saved result to: {args.output}")

        try:
            visualize_results(input_texture, result,
                            f"Image Quilting Results\nBlock: {args.block_size}, Overlap: {quilter.overlap}")
        except Exception as e:
            print(f"Visualization error: {e}. Result still saved.")

    except Exception as e:
        print(f"Error during synthesis: {e}")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())