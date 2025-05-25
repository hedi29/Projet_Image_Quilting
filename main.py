"""
Simple main script for Image Quilting texture synthesis.
Usage: python main_simple.py --input texture.jpg --output result.png
"""

import argparse
import time
import os
import numpy as np
from src import ImageQuilting, load_texture, save_image, visualize_results, evaluate_texture_quality


def create_test_texture():
    """Create a test texture with clear structure for testing."""
    size = 64
    texture = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create a wood-like grain pattern
    for i in range(size):
        for j in range(size):
            # Base wood color
            base_color = [139, 90, 43]  # Brown
            
            # Add vertical grain lines
            grain = np.sin(j * 0.3) * 20
            noise = np.random.randint(-10, 10)
            
            # Apply variations
            for c in range(3):
                val = base_color[c] + grain + noise
                texture[i, j, c] = np.clip(val, 0, 255)
    
    return texture


def main():
    parser = argparse.ArgumentParser(description='Image Quilting Texture Synthesis')
    parser.add_argument('--input', type=str, help='Input texture image path')
    parser.add_argument('--output', type=str, default='output.png', help='Output image path')
    parser.add_argument('--block-size', type=int, default=32, help='Block size (default: 32)')
    parser.add_argument('--overlap-ratio', type=float, default=1/6, help='Overlap ratio (default: 1/6)')
    parser.add_argument('--tolerance', type=float, default=0.1, help='Error tolerance (default: 0.1)')
    parser.add_argument('--output-width', type=int, default=None, help='Output width')
    parser.add_argument('--output-height', type=int, default=None, help='Output height')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate synthesis quality')
    parser.add_argument('--test', action='store_true', help='Use test texture instead of input file')
    
    args = parser.parse_args()
    
    # Load or create texture
    if args.test or args.input is None:
        print("Using test texture...")
        input_texture = create_test_texture()
        if args.input is None:
            args.input = "test_texture"
    else:
        print(f"Loading texture from: {args.input}")
        try:
            input_texture = load_texture(args.input)
        except Exception as e:
            print(f"Error loading texture: {e}")
            print("Using test texture instead...")
            input_texture = create_test_texture()
            args.input = "test_texture"
    
    print(f"Input texture shape: {input_texture.shape}")
    
    # Determine output size
    if args.output_width is None:
        output_width = input_texture.shape[1] * 2
    else:
        output_width = args.output_width
        
    if args.output_height is None:
        output_height = input_texture.shape[0] * 2
    else:
        output_height = args.output_height
    
    output_size = (output_height, output_width)
    print(f"Output size: {output_size}")
    
    # Initialize quilting algorithm
    quilter = ImageQuilting(
        block_size=args.block_size,
        overlap_ratio=args.overlap_ratio,
        tolerance=args.tolerance
    )
    
    print(f"Algorithm parameters:")
    print(f"  Block size: {quilter.block_size}")
    print(f"  Overlap: {quilter.overlap}")
    print(f"  Tolerance: {quilter.tolerance}")
    
    # Synthesize texture
    print("\nStarting texture synthesis...")
    start_time = time.time()
    
    try:
        result = quilter.synthesize_texture(input_texture, output_size)
        
        synthesis_time = time.time() - start_time
        print(f"Synthesis completed in {synthesis_time:.2f} seconds")
        
        # Save result
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        save_image(result, args.output)
        print(f"Saved result to: {args.output}")
        
        # Evaluate quality if requested
        if args.evaluate:
            print("\nEvaluating synthesis quality...")
            metrics = evaluate_texture_quality(input_texture, result)
            
            # Save metrics
            metrics_path = os.path.splitext(args.output)[0] + '_metrics.txt'
            with open(metrics_path, 'w') as f:
                f.write(f"Input: {args.input}\n")
                f.write(f"Output: {args.output}\n")
                f.write(f"Block size: {args.block_size}\n")
                f.write(f"Overlap ratio: {args.overlap_ratio}\n")
                f.write(f"Tolerance: {args.tolerance}\n")
                f.write(f"Synthesis time: {synthesis_time:.2f}s\n")
                f.write("\nQuality Metrics:\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.6f}\n")
            
            print(f"Saved evaluation metrics to: {metrics_path}")
        
        # Visualize results
        try:
            visualize_results(input_texture, result, 
                            f"Image Quilting Results\nBlock: {args.block_size}, Overlap: {quilter.overlap}")
        except Exception as e:
            print(f"Visualization error: {e}")
            print("Results saved successfully despite visualization error.")
    
    except Exception as e:
        print(f"Error during synthesis: {e}")
        return 1
    
    return 0


def run_parameter_study():
    """Run a parameter study to find optimal settings."""
    print("Running parameter study...")
    
    # Create test texture
    test_texture = create_test_texture()
    
    # Parameter combinations to test
    parameter_sets = [
        {'block_size': 16, 'overlap_ratio': 1/6, 'tolerance': 0.1},
        {'block_size': 24, 'overlap_ratio': 1/6, 'tolerance': 0.1},
        {'block_size': 32, 'overlap_ratio': 1/6, 'tolerance': 0.1},
        {'block_size': 24, 'overlap_ratio': 1/4, 'tolerance': 0.1},
        {'block_size': 32, 'overlap_ratio': 1/4, 'tolerance': 0.1},
        {'block_size': 32, 'overlap_ratio': 1/6, 'tolerance': 0.15},
    ]
    
    results = []
    
    for i, params in enumerate(parameter_sets):
        print(f"\nTesting parameter set {i+1}: {params}")
        
        quilter = ImageQuilting(**params)
        
        start_time = time.time()
        synthesized = quilter.synthesize_texture(test_texture, (128, 128))
        synthesis_time = time.time() - start_time
        
        quality = evaluate_texture_quality(test_texture, synthesized, verbose=False)
        
        results.append({
            'params': params,
            'quality': quality,
            'time': synthesis_time,
            'synthesized': synthesized
        })
        
        print(f"Score: {quality['overall_score']:.4f}, Time: {synthesis_time:.2f}s")
    
    # Find best result
    best_result = max(results, key=lambda x: x['quality']['overall_score'])
    
    print(f"\nBest parameters: {best_result['params']}")
    print(f"Best score: {best_result['quality']['overall_score']:.4f}")
    print(f"Time: {best_result['time']:.2f}s")
    
    # Save best result
    save_image(best_result['synthesized'], 'best_result.png')
    print("Saved best result to: best_result.png")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Check if user wants to run parameter study
    if len(sys.argv) > 1 and sys.argv[1] == '--parameter-study':
        run_parameter_study()
    else:
        exit_code = main()
        sys.exit(exit_code)