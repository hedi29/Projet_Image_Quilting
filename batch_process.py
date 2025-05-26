import argparse
import os
import time
from src import ImageQuilting, load_texture, save_image

def main():
    parser = argparse.ArgumentParser(description='Batch process images for texture synthesis.')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing input texture images.')
    parser.add_argument('--results_dir', type=str, default='results_batch',
                        help='Directory to save synthesized textures.')
    
    # Algorithm parameters
    parser.add_argument('--output_width', type=int, default=1024,
                        help='Width of the output synthesized texture.')
    parser.add_argument('--output_height', type=int, default=1024,
                        help='Height of the output synthesized texture.')
    parser.add_argument('--block_size', type=int, default=128,
                        help='Block size for image quilting.')
    parser.add_argument('--overlap_ratio', type=float, default=1/6,
                        help='Overlap ratio between blocks.')
    parser.add_argument('--tolerance', type=float, default=0.2,
                        help='Error tolerance for selecting matching blocks.')

    args = parser.parse_args()

    # Validate data directory
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Results will be saved in '{args.results_dir}'")

    # List image files
    image_files = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
    for item in os.listdir(args.data_dir):
        if item.lower().endswith(supported_extensions):
            image_files.append(os.path.join(args.data_dir, item))

    if not image_files:
        print(f"No images found in '{args.data_dir}'.")
        return

    print(f"Found {len(image_files)} images to process.")

    # Initialize Quilter
    quilter = ImageQuilting(
        block_size=args.block_size,
        overlap_ratio=args.overlap_ratio,
        tolerance=args.tolerance
    )
    
    output_size = (args.output_height, args.output_width)

    total_start_time = time.time()

    for i, img_path in enumerate(image_files):
        print(f"\nProcessing image {i+1}/{len(image_files)}: {img_path}")
        try:
            input_texture = load_texture(img_path)
            print(f"  Loaded input texture: {img_path} (Shape: {input_texture.shape})")
        except Exception as e:
            print(f"  Error loading texture {img_path}: {e}")
            continue

        start_time = time.time()
        try:
            print(f"  Synthesizing texture with output size {output_size}...")
            synthesized_texture = quilter.synthesize_texture(input_texture, output_size)
            
            synthesis_time = time.time() - start_time
            print(f"  Synthesis completed in {synthesis_time:.2f} seconds.")

            # Construct output path
            base_name = os.path.basename(img_path)
            name, ext = os.path.splitext(base_name)
            output_filename = f"{name}_quilted_w{args.output_width}_h{args.output_height}_b{args.block_size}{ext if ext else '.png'}"
            output_path = os.path.join(args.results_dir, output_filename)
            
            save_image(synthesized_texture, output_path)
            print(f"  Saved synthesized texture to: {output_path}")

        except Exception as e:
            print(f"  Error during synthesis for {img_path}: {e}")
            import traceback
            traceback.print_exc() # For more detailed error during development

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"\nBatch processing completed in {total_time:.2f} seconds.")

if __name__ == '__main__':
    # This ensures that the multiprocessing code (if any part of your imported src uses it)
    # behaves correctly when the script is run directly.
    main()