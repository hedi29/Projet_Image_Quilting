"""
Main entry point for the Image Quilting project.

This script provides a command-line interface to run texture synthesis
and texture transfer using the Image Quilting algorithm.
"""

import os
import argparse
import time
import numpy as np
import cv2
from src.quilting import synthesize_texture
from src.transfer import texture_transfer
from src.evaluation import evaluate_synthesis_quality
from src.utils import load_texture, save_image, visualize_results, visualize_transfer_results


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Image Quilting for Texture Synthesis and Transfer')
    
    # Common parameters
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input texture image')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save output image')
    parser.add_argument('--mode', type=str, choices=['synthesis', 'transfer'], default='synthesis',
                        help='Mode: texture synthesis or texture transfer')
    parser.add_argument('--block-size', type=int, default=32,
                        help='Size of quilting blocks (default: 32)')
    parser.add_argument('--overlap', type=int, default=6,
                        help='Size of overlap between blocks (default: 6)')
    parser.add_argument('--tolerance', type=float, default=0.1,
                        help='Error tolerance for block selection (default: 0.1)')
    parser.add_argument('--output-width', type=int, default=None,
                        help='Width of output image (default: 2x input width)')
    parser.add_argument('--output-height', type=int, default=None,
                        help='Height of output image (default: 2x input height)')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization of original and result')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate synthesis quality using metrics')
    
    # Texture transfer specific parameters
    parser.add_argument('--target', type=str, default=None,
                        help='Path to target image for texture transfer')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Weight between texture and correspondence (default: 0.8)')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of refinement iterations for transfer (default: 3)')
    
    return parser.parse_args()


def main():
    """Main function to run the Image Quilting algorithm."""
    args = parse_arguments()
    
    # Load input texture
    input_texture = load_texture(args.input)
    print(f"Loaded input texture: {args.input}, shape: {input_texture.shape}")
    
    # Determine output size
    if args.output_width is None:
        output_width = input_texture.shape[1] * 2
    else:
        output_width = args.output_width
        
    if args.output_height is None:
        output_height = input_texture.shape[0] * 2
    else:
        output_height = args.output_height
    
    print(f"Output dimensions: {output_width}x{output_height}")
    
    # Run algorithm based on mode
    start_time = time.time()
    
    if args.mode == 'synthesis':
        print(f"Running texture synthesis with block size: {args.block_size}, overlap: {args.overlap}")
        
        result = synthesize_texture(
            input_texture,
            output_height,
            output_width,
            args.block_size,
            args.overlap,
            args.tolerance
        )
        
        # Save result
        save_image(result, args.output)
        print(f"Saved synthesized texture to: {args.output}")
        
        # Generate visualization if requested
        if args.visualize:
            vis_path = os.path.splitext(args.output)[0] + '_visualization.png'
            vis_image = visualize_results(
                input_texture, result,
                title=f"Texture Synthesis (Block: {args.block_size}, Overlap: {args.overlap})",
                save_path=vis_path
            )
            print(f"Saved visualization to: {vis_path}")
        
    elif args.mode == 'transfer':
        if args.target is None:
            raise ValueError("Target image must be specified for texture transfer mode")
        
        # Load target image
        target_image = load_texture(args.target)
        print(f"Loaded target image: {args.target}, shape: {target_image.shape}")
        
        print(f"Running texture transfer with alpha: {args.alpha}, iterations: {args.iterations}")
        
        result = texture_transfer(
            input_texture,
            target_image,
            output_size=(output_height, output_width),
            block_size=args.block_size,
            overlap_size=args.overlap,
            alpha=args.alpha,
            iterations=args.iterations,
            tolerance=args.tolerance
        )
        
        # Save result
        save_image(result, args.output)
        print(f"Saved transferred texture to: {args.output}")
        
        # Generate visualization if requested
        if args.visualize:
            vis_path = os.path.splitext(args.output)[0] + '_visualization.png'
            vis_image = visualize_transfer_results(
                input_texture, target_image, result,
                title=f"Texture Transfer (Block: {args.block_size}, Alpha: {args.alpha})",
                save_path=vis_path
            )
            print(f"Saved visualization to: {vis_path}")
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    
    # Evaluate synthesis quality if requested
    if args.evaluate:
        print("Evaluating synthesis quality...")
        metrics = evaluate_synthesis_quality(input_texture, result)
        
        print("\nQuality Metrics:")
        print(f"SSIM: {metrics['ssim']:.4f} (higher is better)")
        print(f"MSE: {metrics['mse']:.4f} (lower is better)")
        print(f"Histogram Distance: {metrics['histogram_distance']:.4f} (lower is better)")
        
        # Save metrics to file
        metrics_path = os.path.splitext(args.output)[0] + '_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write(f"Input: {args.input}\n")
            f.write(f"Output: {args.output}\n")
            f.write(f"Mode: {args.mode}\n")
            f.write(f"Block Size: {args.block_size}\n")
            f.write(f"Overlap: {args.overlap}\n")
            f.write(f"Tolerance: {args.tolerance}\n")
            if args.mode == 'transfer':
                f.write(f"Target: {args.target}\n")
                f.write(f"Alpha: {args.alpha}\n")
                f.write(f"Iterations: {args.iterations}\n")
            f.write("\nQuality Metrics:\n")
            f.write(f"SSIM: {metrics['ssim']:.6f}\n")
            f.write(f"MSE: {metrics['mse']:.6f}\n")
            f.write(f"Histogram Distance: {metrics['histogram_distance']:.6f}\n")
        
        print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()