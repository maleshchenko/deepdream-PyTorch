# DeepDream in PyTorch

DeepDream is a computer vision algorithm that produces psychedelic, dreamlike images by enhancing patterns in existing images using a deep neural network. It works by performing gradient ascent in feature space, causing the network to amplify patterns it recognizes. This helps visualize what features a deep network has learned, while producing visually striking artistic results.

DeepDream was invented by Google and applied to the Inception network (trained on ImageNet) in 2014. The original algorithm was implemented in Caffe. This repository is a **PyTorch implementation** using InceptionV3, making it easy to run and understand the core concepts behind the algorithm.

```python test_pytorch.py --content source.jpg --layer Mixed_5b --iterations 200 --step 1.6 --octaves 5```

<img width="400" height="400" alt="source" src="https://github.com/user-attachments/assets/e755cea5-75aa-4e21-a640-8c57aa7985dc" /> <img width="400" height="400" alt="deepdream_pytorch" src="https://github.com/user-attachments/assets/ff33b51e-02fc-480c-8667-dc0be42db1dd" />

## How It Works

1. Load a pre-trained neural network (InceptionV3) trained on ImageNet
2. Start with an input image (or random noise)
3. Forward pass through the network to extract activations from selected layers
4. Compute gradients with respect to the input image to maximize activations
5. Update the image in the direction that enhances detected patterns
6. Repeat to create increasingly "dreamy" results

## Requirements

- Python >= 3.13
- PyTorch >= 2.0.0
- PyTorchVision >= 0.15.0
- NumPy, Pillow, SciPy (for image processing)
- Matplotlib (optional, for visualization)

## Installation

### 1. Create a Virtual Environment

```bash
# Using venv (built-in to Python)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda (if you prefer)
conda create -n deepdream python=3.13
conda activate deepdream
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install Development Dependencies

For visualization support with Matplotlib:
```bash
pip install -e ".[dev]"
```

**Note:** The project does not rely on any bundled environments. Each user should create their own virtual environment as shown above.

## Usage

### Basic Usage (Random Noise)

Generate DeepDream from scratch using random noise:

```bash
python test_pytorch.py
```

### Apply to an Existing Image

```bash
python test_pytorch.py --content my_image.jpg --output result.jpg
```

### Advanced Options

```bash
python test_pytorch.py \
    --content input.jpg \
    --output output.jpg \
    --layer Mixed_7c \
    --iterations 30 \
    --step 0.8 \
    --octaves 4
```

## Parameters

- **--content**: Source image file (if not provided, generates from random noise)
- **--output**: Output filename for the generated image
- **--layer**: Which layer to target for activation maximization
  - Deeper layers (Mixed_7x): More complex, semantic features
  - Middle layers (Mixed_5x, Mixed_6x): Medium-level patterns
  - Default: `Mixed_5c`
- **--iterations**: How many gradient ascent steps per octave (higher = more detailed)
- **--step**: Gradient step size - controls how aggressively patterns are enhanced (0.1-2.0)
- **--octaves**: Number of scales to process - higher octaves enhance larger patterns first

## Available Layers

```
Mixed_5b, Mixed_5c, Mixed_6a, Mixed_6b, Mixed_6c, Mixed_7a, Mixed_7b, Mixed_7c
```

**Layer Selection Tips:**
- `Mixed_5b/5c`: Good for delicate, intricate patterns
- `Mixed_6b/6c`: Balanced results with both textures and objects
- `Mixed_7a/7b`: Strong object-like features, more pronounced patterns

## Algorithm Variants

The code includes three rendering algorithms:

### 1. Multi-Scale Octave Processing (Default - `render_deepdream`)
Highest quality. Processes image from coarse to fine resolution.
- **Pros**: Better results, captures patterns at multiple scales
- **Cons**: Slower, uses more memory

### 2. Naive Gradient Ascent (`render_naive`)
Simple single-scale approach.
- **Pros**: Fast, lower memory usage
- **Cons**: Lower quality results, can produce artifacts

### 3. Laplacian Normalization (`render_lapnorm`)
Advanced technique using frequency domain normalization.
- **Pros**: Coherent results, better gradient distribution
- **Cons**: More complex, moderate speed

## Technical Details

### Multi-Scale Octave Processing

The algorithm uses a Laplacian pyramid approach:

1. **Decomposition**: Split image into Laplacian bands (coarse → fine)
2. **Coarse-to-Fine Processing**: Start with low resolution, progressively add details
3. **Reconstruction**: Combine processed bands back into final image

This prevents small-scale artifacts from overwhelming the result and enables coherent pattern enhancement across scales.

### Gradient Tiling

To handle large images without GPU memory issues:
- Image is processed in overlapping tiles (512x512)
- Random shifts applied to reduce tile boundary artifacts
- Gradients accumulated across all tiles

### Normalization

Gradients are normalized to prevent explosion:
```
gradient = gradient / (std(gradient) + ε)
```

This ensures stable updates regardless of gradient magnitude.

## Output Files

The script generates:
- `original_image_pytorch.jpg`: Input image (for reference)
- `deepdream_pytorch.jpg`: Final DeepDream result (multi-scale)

## Examples

### Artistic Enhancement
```bash
python test_pytorch.py --content source.jpg --layer Mixed_6c \
    --iterations 25 --step 0.6 --octaves 5
```

### Subtle Effects (Lower Iterations)
```bash
python test_pytorch.py --content source.jpg --layer Mixed_5b \
    --iterations 10 --step 0.3 --octaves 2
```

### Extreme Psychedelic Effects
```bash
python test_pytorch.py --content source.jpg --layer Mixed_7c \
    --iterations 50 --step 1.5 --octaves 6
```

## Tips for Best Results

1. **Start with good source images**: Interesting textures and patterns work well
2. **Try different layers**: Each layer produces different visual styles
3. **Adjust iterations**: More iterations = more pronounced effects
4. **Use appropriate step size**: Too high causes artifacts, too low gives subtle effects
5. **Multiple octaves**: Usually 3-5 octaves gives best results
6. **GPU recommended**: CUDA GPU significantly speeds up processing

## Hardware Considerations

- **GPU Memory**: Full HD images need ~8GB VRAM
- **CPU Mode**: Works but very slow (10-50x slower than GPU)
- **Recommended**: NVIDIA GPU with 8GB+ VRAM for best performance

## References

- [Original DeepDream by Google](https://github.com/google/deepdream)
- [Original Implementation in Caffe](https://github.com/google/deepdream/blob/master/dream.ipynb)
- [TensorFlow Tutorial](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb)
- [Wikipedia on DeepDream](https://en.wikipedia.org/wiki/DeepDream)
- [InceptionV3 Architecture](https://arxiv.org/abs/1512.00567)
- [Deep Inside Convolutional Networks: Visualising Image Classification Models](https://arxiv.org/abs/1312.6034)


## References
* [Original Implementation in Caffe](https://github.com/google/deepdream/blob/master/dream.ipynb)
* [TensorFlow Tutorial](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb) (what I followed)
* [Wikipedia on DeepDream](https://en.wikipedia.org/wiki/DeepDream)
* [Inception TensorFlow Code](https://github.com/tensorflow/models/tree/master/inception) (the network commonly used in the algorithm)
* [A Nice Explanation of DeepDream](http://www.kpkaiser.com/machine-learning/diving-deeper-into-deep-dreams/)
* [A Nice Video on DeepDream](https://www.youtube.com/watch?v=MrBzgvUNr4w) (moves pretty fast when it gets to the code part, be warned)

