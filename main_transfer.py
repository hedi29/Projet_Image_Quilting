"""
Texture Transfer Main Script - Complete demonstration of texture transfer capabilities.

This script shows how to use the texture transfer implementation to render
one image using the texture style of another image.

Usage examples:
    # Basic texture transfer
    python main_transfer.py --source texture.jpg --target photo.jpg --output result.png
    
    # With custom parameters
    python main_transfer.py --source wood.jpg --target portrait.jpg --alpha 0.7 --block-size 24
    
    # Run demo with built-in examples
    python main_transfer.py --demo
"""

import argparse
import time
import os
import numpy as np
import cv2
from pathlib import Path

# Import our texture transfer modules
from src.transfer import TextureTransfer, transfer_texture_simplified
from src.utils import load_texture, save_image, visualize_results


def create_demo_source_texture():
    """Create a demo source texture with clear patterns."""
    size = 128
    texture = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create a brick-like pattern
    brick_height = 20
    brick_width = 40
    mortar_color = [200, 200, 200]  # Light gray
    brick_color = [150, 80, 60]     # Reddish brown
    
    for i in range(size):
        for j in range(size):
            # Determine if we're in mortar or brick
            brick_row = i // brick_height
            brick_col = j // brick_width
            
            # Offset every other row for realistic brick pattern
            if brick_row % 2 == 1:
                brick_col = (j + brick_width // 2) // brick_width
            
            # Mortar lines
            if (i % brick_height < 2) or (j % brick_width < 2):
                texture[i, j] = mortar_color
            else:
                # Add some variation to brick color
                variation = np.random.randint(-20, 20, 3)
                color = np.clip(np.array(brick_color) + variation, 0, 255)
                texture[i, j] = color
    
    return texture


def create_demo_target_image():
    """Create a demo target image with clear structure."""
    size = 200
    target = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create a simple geometric pattern
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            # Distance from center
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            
            # Create concentric circles with different intensities
            if dist < 30:
                intensity = 255
            elif dist < 60:
                intensity = 180
            elif dist < 90:
                intensity = 120
            else:
                intensity = 60
            
            # Add some angular variation
            angle = np.arctan2(i - center, j - center)
            angular_variation = int(30 * np.sin(4 * angle))
            
            final_intensity = np.clip(intensity + angular_variation, 0, 255)
            target[i, j] = [final_intensity, final_intensity, final_intensity]
    
    return target


def run_texture_transfer_demo():
    """Run a comprehensive demo of texture transfer capabilities."""
    print("🎨 Running Texture Transfer Demo")
    print("=" * 50)
    
    # Create demo images
    print("Creating demo source texture and target image...")
    source_texture = create_demo_source_texture()
    target_image = create_demo_target_image()
    
    # Save demo inputs
    os.makedirs("demo_results", exist_ok=True)
    save_image(source_texture, "demo_results/demo_source.png")
    save_image(target_image, "demo_results/demo_target.png")
    print("✅ Saved demo inputs to demo_results/")
    
    # Test different alpha values
    alpha_values = [0.9, 0.7, 0.5, 0.3]
    
    for alpha in alpha_values:
        print(f"\n🔄 Testing alpha = {alpha}")
        
        start_time = time.time()
        result = transfer_texture_simplified(
            source_texture=source_texture,
            target_image=target_image,
            alpha=alpha,
            block_size=24,
            correspondence_type='luminance',
            iterations=2,
            blur_sigma=3.0
        )
        transfer_time = time.time() - start_time
        
        # Save result
        output_path = f"demo_results/transfer_alpha_{alpha:.1f}.png"
        save_image(result, output_path)
        print(f"✅ Saved result to {output_path} (took {transfer_time:.1f}s)")
    
    print(f"\n🎉 Demo completed! Check the demo_results/ folder for outputs.")
    print("Try different alpha values to see the texture/correspondence balance:")
    print("  • α = 0.9: Strong texture influence, weak correspondence")
    print("  • α = 0.7: Balanced (recommended)")
    print("  • α = 0.5: Equal texture and correspondence")
    print("  • α = 0.3: Weak texture, strong correspondence")


def main():
    parser = argparse.ArgumentParser(
        description='Texture Transfer using Image Quilting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transfer
  python main_transfer.py --source wood.jpg --target portrait.jpg
  
  # Custom parameters
  python main_transfer.py --source fabric.png --target landscape.jpg --alpha 0.8 --block-size 32
  
  # Run demo
  python main_transfer.py --demo
        """
    )
    
    # Input/output arguments
    parser.add_argument('--source', type=str, help='Source texture image path')
    parser.add_argument('--target', type=str, help='Target image path (guides the transfer)')
    parser.add_argument('--output', type=str, default='transfer_result.png', help='Output image path')
    
    # Algorithm parameters
    parser.add_argument('--alpha', type=float, default=0.7, 
                       help='Texture weight (0.0=pure correspondence, 1.0=pure texture, default=0.7)')
    parser.add_argument('--block-size', type=int, default=32, 
                       help='Block size for quilting (default: 32)')
    parser.add_argument('--iterations', type=int, default=3, 
                       help='Number of refinement iterations (default: 3)')
    parser.add_argument('--correspondence', type=str, default='luminance', 
                       choices=['luminance', 'blurred'],
                       help='Correspondence map type (default: luminance)')
    parser.add_argument('--blur-sigma', type=float, default=3.0,
                       help='Gaussian blur sigma for \'blurred\' correspondence map (default: 3.0)')
    
    # Output options
    parser.add_argument('--output-width', type=int, default=None, 
                       help='Output width (default: target image width)')
    parser.add_argument('--output-height', type=int, default=None, 
                       help='Output height (default: target image height)')
    
    # Special modes
    parser.add_argument('--demo', action='store_true', 
                       help='Run demo with built-in examples')
    parser.add_argument('--visualize', action='store_true', 
                       help='Show visualization of results')
    
    args = parser.parse_args()
    
    # Run demo mode
    if args.demo:
        run_texture_transfer_demo()
        return 0
    
    # Validate required arguments
    if not args.source or not args.target:
        print("❌ Error: Both --source and --target are required (or use --demo)")
        parser.print_help()
        return 1
    
    # Load images
    print(f"🔄 Loading images...")
    try:
        source_texture = load_texture(args.source)
        target_image = load_texture(args.target)
        print(f"✅ Source texture: {source_texture.shape}")
        print(f"✅ Target image: {target_image.shape}")
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return 1
    
    # Determine output size
    if args.output_width and args.output_height:
        output_size = (args.output_height, args.output_width)
    else:
        output_size = target_image.shape[:2]
    
    print(f"🎯 Output size: {output_size}")
    
    # Validate parameters
    if not (0.0 <= args.alpha <= 1.0):
        print(f"❌ Error: alpha must be between 0.0 and 1.0, got {args.alpha}")
        return 1
    
    if args.block_size < 8 or args.block_size > 64:
        print(f"⚠️  Warning: block_size {args.block_size} is outside recommended range [8, 64]")
    
    # Print algorithm parameters
    print(f"\n🔧 Algorithm Parameters:")
    print(f"   Alpha (texture weight): {args.alpha}")
    print(f"   Block size: {args.block_size}")
    print(f"   Iterations: {args.iterations}")
    print(f"   Correspondence: {args.correspondence}")
    if args.correspondence == 'blurred':
        print(f"   Blur Sigma: {args.blur_sigma}")
    
    # Perform texture transfer
    print(f"\n🎨 Starting texture transfer...")
    start_time = time.time()
    
    try:
        # Use the simple transfer function
        result = transfer_texture_simplified(
            source_texture=source_texture,
            target_image=target_image,
            alpha=args.alpha,
            block_size=args.block_size,
            correspondence_type=args.correspondence,
            iterations=args.iterations,
            output_size=output_size,
            blur_sigma=args.blur_sigma
        )
        
        transfer_time = time.time() - start_time
        print(f"✅ Transfer completed in {transfer_time:.2f} seconds")
        
        # Save result
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        save_image(result, args.output)
        print(f"💾 Saved result to: {args.output}")
        
        # Visualize if requested
        if args.visualize:
            try:
                print("🖼️  Showing visualization...")
                # Create a comparison visualization
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(source_texture)
                axes[0].set_title('Source Texture')
                axes[0].axis('off')
                
                axes[1].imshow(target_image)
                axes[1].set_title('Target Image')
                axes[1].axis('off')
                
                axes[2].imshow(result)
                axes[2].set_title(f'Transfer Result (α={args.alpha})')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"⚠️  Visualization error: {e}")
                print("Result saved successfully despite visualization error.")
        
        # Print summary
        print(f"\n📊 Transfer Summary:")
        print(f"   Source: {args.source} ({source_texture.shape})")
        print(f"   Target: {args.target} ({target_image.shape})")
        print(f"   Result: {args.output} ({result.shape})")
        print(f"   Time: {transfer_time:.2f} seconds")
        print(f"   Alpha: {args.alpha} (texture influence)")
        
    except Exception as e:
        print(f"❌ Error during texture transfer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def run_batch_transfer():
    """Run texture transfer on multiple image pairs."""
    print("🔄 Batch Texture Transfer")
    
    # Define image pairs for batch processing
    # You can modify this list based on your available images
    image_pairs = [
        ("data/wood_texture.jpg", "data/portrait.jpg", "results/wood_portrait.png"),
        ("data/fabric_texture.jpg", "data/landscape.jpg", "results/fabric_landscape.png"),
        ("data/stone_texture.jpg", "data/building.jpg", "results/stone_building.png"),
    ]
    
    # Different alpha values to test
    alpha_values = [0.6, 0.7, 0.8]
    
    os.makedirs("batch_results", exist_ok=True)
    
    for source_path, target_path, base_output in image_pairs:
        if not (os.path.exists(source_path) and os.path.exists(target_path)):
            print(f"⚠️  Skipping {source_path} -> {target_path} (files not found)")
            continue
        
        print(f"\n🎨 Processing: {source_path} -> {target_path}")
        
        try:
            source = load_texture(source_path)
            target = load_texture(target_path)
            
            for alpha in alpha_values:
                output_path = f"batch_results/{Path(base_output).stem}_alpha_{alpha:.1f}.png"
                
                result = transfer_texture_simplified(
                    source_texture=source,
                    target_image=target,
                    alpha=alpha,
                    block_size=28,
                    iterations=2,
                    correspondence_type='luminance',
                    blur_sigma=3.0
                )
                
                save_image(result, output_path)
                print(f"✅ Saved: {output_path}")
                
        except Exception as e:
            print(f"❌ Error processing {source_path}: {e}")


if __name__ == "__main__":
    import sys
    
    # Check for special batch mode
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        run_batch_transfer()
    else:
        exit_code = main()
        sys.exit(exit_code) 