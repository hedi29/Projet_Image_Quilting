# Texture Transfer with Image Quilting

This project implements texture transfer using the Image Quilting algorithm as described in the Efros & Freeman paper. It allows you to render one image using the texture style of another image while preserving the original structure.

## 🚀 Quick Start

### 1. Simple Example (Recommended for beginners)

```bash
python simple_transfer_example.py
```

This creates test images and demonstrates basic texture transfer. Perfect for understanding how it works!

### 2. Demo Mode (See different alpha values)

```bash
python main_transfer.py --demo
```

This runs a comprehensive demo showing how different alpha values affect the texture/correspondence balance.

### 3. Use Your Own Images

```bash
python main_transfer.py --source your_texture.jpg --target your_photo.jpg --output result.png
```

## 📁 Files Overview

- **`main_transfer.py`** - Complete main script with all options
- **`simple_transfer_example.py`** - Minimal example for beginners  
- **`src/transfer.py`** - Core texture transfer implementation
- **`src/corresmendance_map.py`** - Correspondence map computation

## 🎛️ Parameters Explained

### Alpha (α) - Texture vs Correspondence Balance
- **α = 1.0**: Pure texture (ignores target structure)
- **α = 0.8**: Strong texture influence (recommended for artistic effects)
- **α = 0.7**: Balanced (recommended default)
- **α = 0.5**: Equal texture and correspondence
- **α = 0.3**: Weak texture, strong correspondence (preserves target structure)
- **α = 0.0**: Pure correspondence (just copies target)

### Block Size
- **16-24**: Fine detail, slower processing
- **32**: Good balance (default)
- **40-48**: Faster, less detail

### Iterations
- **1**: Fast, basic quality
- **2-3**: Good quality (recommended)
- **4+**: Diminishing returns, much slower

## 📖 Usage Examples

### Basic Transfer
```bash
python main_transfer.py --source wood_texture.jpg --target portrait.jpg
```

### Custom Parameters
```bash
python main_transfer.py \
    --source fabric.png \
    --target landscape.jpg \
    --alpha 0.8 \
    --block-size 24 \
    --iterations 3 \
    --output my_result.png
```

### With Visualization
```bash
python main_transfer.py --source texture.jpg --target photo.jpg --visualize
```

### Batch Processing
```bash
python main_transfer.py --batch
```

## 🎨 How It Works

1. **Source Texture**: Provides the visual style/pattern
2. **Target Image**: Guides where different textures should be placed
3. **Correspondence Map**: Usually luminance (brightness) of target image
4. **Alpha Parameter**: Controls balance between texture and correspondence
5. **Result**: Target image rendered with source texture style

## 💡 Tips for Best Results

### Good Source Textures
- ✅ Repetitive patterns (wood grain, fabric, stone)
- ✅ Rich textures with clear structure
- ✅ High contrast details
- ❌ Photos with specific objects
- ❌ Very smooth/uniform textures

### Good Target Images
- ✅ Clear structure/shapes
- ✅ Good contrast between regions
- ✅ Not too much fine detail
- ❌ Very noisy images
- ❌ Extremely complex scenes

### Parameter Guidelines
- Start with `alpha=0.7`, `block_size=32`, `iterations=3`
- For artistic effects: increase alpha (0.8-0.9)
- For structure preservation: decrease alpha (0.3-0.5)
- For fine details: decrease block_size (16-24)
- For speed: increase block_size (40-48), decrease iterations

## 🔧 Troubleshooting

### "Module not found" errors
Make sure you're running from the project root directory:
```bash
cd /path/to/your/project
python main_transfer.py --demo
```

### Poor results
- Try different alpha values (0.3 to 0.9)
- Ensure source texture has clear patterns
- Check that target image has good contrast
- Try smaller block sizes for more detail

### Slow performance
- Increase block_size (32 → 48)
- Reduce iterations (3 → 2)
- Use smaller images
- Reduce output size with `--output-width` and `--output-height`

## 📊 Example Results

The demo mode will create several examples showing:
- `demo_source.png` - Source texture (brick pattern)
- `demo_target.png` - Target image (concentric circles)
- `transfer_alpha_0.9.png` - Strong texture influence
- `transfer_alpha_0.7.png` - Balanced result
- `transfer_alpha_0.5.png` - Equal influence
- `transfer_alpha_0.3.png` - Strong correspondence influence

## 🔬 Advanced Usage

### Custom Correspondence Maps
```python
from src.transfer import transfer_texture

result = transfer_texture(
    source_texture=source,
    target_image=target,
    correspondence='blurred'  # Smoother correspondence
)
```

### Using the Class Directly
```python
from src.transfer import TextureTransfer

transferer = TextureTransfer(
    block_size=32,
    alpha=0.7,
    iterations=3,
    tolerance=0.1
)

result = transferer.transfer_texture(source, target)
```

## 📚 References

- Efros, A. A., & Freeman, W. T. (2001). Image quilting for texture synthesis and transfer. SIGGRAPH 2001.

## 🐛 Common Issues

1. **Import errors**: Make sure all files are in the correct directories
2. **Memory issues**: Use smaller images or increase block size
3. **Poor quality**: Adjust alpha parameter and try different correspondence maps
4. **Slow performance**: Reduce iterations or increase block size

---

**Happy texture transferring! 🎨** 