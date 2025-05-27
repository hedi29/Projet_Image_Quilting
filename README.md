# Image Quilting Project

This project implements the Image Quilting algorithm for texture synthesis and texture transfer, based on the paper "Image Quilting for Texture Synthesis and Transfer" by Alexei A. Efros and William T. Freeman.

## Project Structure

```
Projet_Image/
├── data_couleur/                   # Example data directory for color textures
├── data_couleur_2/                 # Another example data directory
├── data_texture_geometrique/       # Example data for geometric textures
├── data_texture_non_geometrique/   # Example data for non-geometric textures
├── data_transfer/                  # Example data for texture transfer
├── photo_article/                  # Images used in an article/report
├── resultat_article/               # Results for article images
│   └── runs.txt                    # Commands to reproduce these results
├── resultat_couleur/               # Results for color textures
│   └── runs.txt
├── resultat_couleur_2/
│   └── runs.txt
├── resultat_texture_geometrique/
│   └── runs.txt
├── resultat_texture_non_geometrique/
│   └── runs.txt
├── src/
│   ├── __init__.py               # Makes src a Python package
│   ├── quilting.py               # Core Image Quilting algorithm
│   └── utils.py                  # Utility functions for loading, saving, visualizing
├── batch_process.py                # Script for batch processing images
├── evaluation_chi2_LBP.py                   # Script for evaluating texture quality
├── main.py                         # Main script for single image texture synthesis
├── transfer.py                     # Main script for texture transfer
├── requirements.txt                # Python dependencies
├── runs.txt                        # Example evaluation runs
└── README.md                       # This file
```

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd Projet_Image
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Core Components

### 1. Texture Synthesis (`main.py` and `src/quilting.py`)

The `src/quilting.py` file contains the `ImageQuilting` class, which implements the core logic of the texture synthesis algorithm. This includes:
*   Dividing the input texture into blocks.
*   Finding the best matching block for a given location in the output texture based on overlap error.
*   Calculating a minimum error boundary cut (seam) to seamlessly stitch blocks together.
*   Parallelized patch evaluation for performance.

The `main.py` script provides a command-line interface to perform texture synthesis on a single input image.

**How to use `main.py`:**

```bash
python main.py --input <path_to_input_texture> --output <path_for_synthesized_output> [options]
```

**Common Options:**
*   `--input`: Path to the input texture image.
*   `--output`: Path to save the synthesized texture. Default: `output.png`.
*   `--block-size`: Size of the square blocks (e.g., 32, 64). Default: `32`.
*   `--overlap-ratio`: Overlap size as a ratio of `block_size` (e.g., 1/6, 0.2). Default: `1/6`.
*   `--tolerance`: Error tolerance for selecting among best matching blocks. Default: `0.1`.
*   `--output-width`: Desired width for the output texture. Default: 2x input width.
*   `--output-height`: Desired height for the output texture. Default: 2x input height.

**Examples:**

1.  **Basic usage (default output size will be 2x input image dimensions):**
    ```bash
    python main.py --input data_texture_geometrique/D1.jpeg --output results/D1_synthesized_default.png --block-size 64 --tolerance 0.1
    ```

2.  **Specify output dimensions (e.g., 512x512) with a larger block size and custom overlap/tolerance:**
    ```bash
    python main.py --input data_texture_non_geometrique/D2.jpeg --output results/D2_synthesized_512.png --block-size 128 --overlap-ratio 0.2 --tolerance 0.15 --output-width 512 --output-height 512
    ```

3.  **Using parameters similar to some batch processing runs (e.g., for color textures):**
    ```bash
    python main.py --input data_couleur/C1.jpeg --output results/C1_synthesized_large.png --block-size 128 --overlap-ratio 0.2 --tolerance 0.2 --output-width 1024 --output-height 1024
    ```

4.  **Another example with smaller block size, often good for detailed or non-geometric textures:**
    ```bash
    python main.py --input data_texture_non_geometrique/D15.jpeg --output results/D15_synthesized_detailed.png --block-size 64 --overlap-ratio 0.2 --tolerance 0.1 --output-width 768 --output-height 768 
    ```
    *(Note: Ensure the `results/` directory exists or is created by your `save_image` utility if it doesn't handle directory creation automatically. The provided `main.py` script was updated to handle directory creation.)*

### 2. Texture Transfer (`transfer.py`)

The `transfer.py` script implements texture transfer, which applies the style of a source texture to the appearance of a target image. It uses a modified quilting approach where block selection considers both the overlap error with neighboring blocks and a correspondence error with the target image's luminance.

**How to use `transfer.py`:**

```bash
python transfer.py --texture <path_to_source_texture> --target <path_to_target_image> --output <path_for_transfer_result> [options]
```

**Common Options:**
*   `--texture` (or `-t`): Path to the source texture image.
*   `--target` (or `-g`): Path to the target image (whose appearance will be matched).
*   `--output` (or `-o`): Path to save the texture transfer result.
*   `--block_size` (or `-b`): Size of the blocks. Default: `36`.
*   `--overlap_rate` (or `-r`): Overlap ratio. Default: `0.1`.
*   `--alpha` (or `-a`): Weighting factor between overlap error and correspondence error. Default: `0.1`.

**Example:**
```bash
python transfer.py --texture photo_article/rice.jpeg --target data_transfer/bill-big.jpg --output result.png --block_size 20 --overlap_rate 0.3 --alpha 0.1

python transfer.py --texture photo_article/yogurt.jpeg --target data_transfer/bill-big.jpg --output result.png --block_size 20 --overlap_rate 0.15 --alpha 0.1

python transfer.py --texture photo_article/9.jpeg --target data_transfer/bouga_2.png --output result_bouga_1.png --block_size 20 --overlap_rate 0.15 --alpha 0.1

python transfer.py --texture photo_article/yogurt.jpeg --target data_transfer/bouga.png --output result_bouga.png --block_size 20 --overlap_rate 0.15 --alpha 0.1

python transfer.py --texture photo_article/olives.jpeg --target data_transfer/bouga_2.png --output result_bouga_2.png --block_size 20 --overlap_rate 0.15 --alpha 0.1

python transfer.py --texture photo_article/rice.jpeg --target data_transfer/bill-big.jpg --output result.png --block_size 20 --overlap_rate 0.15 --alpha 0.1

```

### 3. Batch Processing (`batch_process.py`)

The `batch_process.py` script allows you to run texture synthesis on all images within a specified directory and save the results to an output directory.

**How to use `batch_process.py`:**

```bash
python batch_process.py --data_dir <input_directory_with_textures> --results_dir <output_directory_for_results> [options]
```

**Common Options:**
*   `--data_dir`: Directory containing input texture images. Default: `data`.
*   `--results_dir`: Directory to save synthesized textures. Default: `results_batch`.
*   `--output_width`: Width of the output synthesized textures. Default: `1024`.
*   `--output_height`: Height of the output synthesized textures. Default: `1024`.
*   `--block_size`: Block size for image quilting. Default: `128`.
*   `--overlap_ratio`: Overlap ratio between blocks. Default: `1/6`.
*   `--tolerance`: Error tolerance for selecting matching blocks. Default: `0.2`.

**Data and Results Folders:**
*   The `--data_dir` should contain the images you want to process (e.g., `data_couleur/`, `data_texture_geometrique/`).
*   The `--results_dir` is where the generated textures will be saved. The script will create this directory if it doesn't exist. Each output file will be named based on the input file and the synthesis parameters used.

**Example (commands from `runs.txt` files):**

*   To reproduce results in `resultat_couleur/` (original command pointed to `resultat_couleur_2` but seems to be for `resultat_couleur` based on directory name, using block size 128):
    ```bash
    python batch_process.py --data_dir data_couleur/ --results_dir resultat_couleur/ --output_width 1024 --output_height 1024 --block_size 128 --overlap_ratio 0.2 --tolerance 0.2
    ```
*   To reproduce results in `resultat_couleur_2/` (block size 64):
    ```bash
    python batch_process.py --data_dir data_couleur_2/ --results_dir resultat_couleur_2/ --output_width 1024 --output_height 1024 --block_size 64 --overlap_ratio 0.2 --tolerance 0.01
    ```
*   To reproduce results in `resultat_article/`:
    ```bash
    python batch_process.py --data_dir photo_article/ --results_dir resultat_article/ --output_width 640 --output_height 640 --block_size 64 --overlap_ratio 0.2 --tolerance 0.1
    ```
*   To reproduce results in `resultat_texture_non_geometrique/`:
    ```bash
    python batch_process.py --data_dir data_texture_non_geometrique/ --results_dir resultat_texture_non_geometrique/ --output_width 1024 --output_height 1024 --block_size 64 --overlap_ratio 0.2 --tolerance 0.1
    ```
*   To reproduce results in `resultat_texture_geometrique/`:
    ```bash
    python batch_process.py --data_dir data_texture_geometrique/ --results_dir resultat_texture_geometrique/ --output_width 1024 --output_height 1024 --block_size 128 --overlap_ratio 0.2 --tolerance 0.2
    ```

### 4. Evaluation (`evaluation_chi2_LBP.py`)

combine la distance de Chi² sur histogrammes BGR 3D
et la distance L2 entre histogrammes LBP.

**How to use `evaluation_chi2_LBP.py`:**

```bash
python evaluation_chi2.py --ref data_texture_geometrique/D8.jpeg --synth resultat_texture_geometrique/D8_quilted_w1024_h1024_b128.jpeg --bins 8 8 8 --P 8 --R 1.0 --lbp_bins 24
```

**Common Options:**
*   `--ref`: Path to the original exemplar texture image.
*   `--synth`: Path to the synthesized/quilted image.
*   `--patch`: Patch size (KxK) for comparison. Default: `40`.
*   `--num`: Number of random patches to sample for comparison. Default: `250`.
*   `--mode`: How to align images if they are different sizes. Choices: `tile` (tile reference to match synth size), `resize` (resize reference to match synth size), `crop` (crop both to smallest common dimensions). Default: `tile`.
*   `--seed`: RNG seed for reproducibility of patch sampling. Default: `0`.

**Example (from root `runs.txt`):**
```bash
python evaluation.py --ref data_texture_geometrique/D1.jpeg --synth resultat_texture_geometrique/D1_quilted_w1024_h1024_b128.png --patch 1000 --mode resize --num 20
```
This command evaluates the synthesized version of `D1.jpeg` against the original, using a patch size of 1000x1000, resizing the reference to match the synthesized image, and sampling 20 patches.

Additional example evaluation parameters mentioned in `runs.txt`:
```bash
# --mode  tile \
# --patch 40  \
# --num   200 \
# --seed  42
```

## Utility Functions (`src/utils.py`)

This module contains helper functions used across the project:
*   `load_texture(path)`: Loads an image from a file path into a NumPy array.
*   `save_image(image, path)`: Saves a NumPy array as an image file.
*   `visualize_results(...)`: Displays the original and synthesized textures side-by-side using Matplotlib (can also save the visualization).

## Contributing

(Optional: Add guidelines for contributing if this is an open project).

## License

(Optional: Specify a license, e.g., MIT, Apache 2.0). 
