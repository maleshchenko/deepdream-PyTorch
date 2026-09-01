#!/usr/bin/env python3
"""
PyTorch DeepDream Implementation

This implementation generates DeepDream images by:
1. Loading a pre-trained InceptionV3 model
2. Iteratively computing gradients of layer activations with respect to the input image
3. Using multi-scale octave processing for better visual results
4. Applying Laplacian pyramid normalization for improved gradient distribution
"""

import os
import argparse
import logging
import numpy as np
import PIL.Image
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Setup: Device, Model, and Layer Configuration
# ============================================================================

# Automatically use GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Load pre-trained InceptionV3 model (trained on ImageNet)
model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
model = model.to(device)
model.eval()

# Disable auxiliary outputs and freeze all parameters (inference only)
model.aux_logits = False
for param in model.parameters():
    param.requires_grad = False

# ImageNet normalization parameters (used for preprocessing)
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

# Target layers in InceptionV3 that produce good DeepDream results
# These layers are in the middle/deeper part of the network
target_layers = ['Mixed_5b', 'Mixed_5c', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c']

# Hook system to capture intermediate layer activations
layer_outputs = {}

def create_hook(layer_name):
    """Create a forward hook to capture layer output"""
    def hook(module, input, output):
        layer_outputs[layer_name] = output
    return hook

# Register hooks for selected layers
hooks = []
for name, module in model.named_modules():
    if any(target in name for target in target_layers):
        hook = module.register_forward_hook(create_hook(name))
        hooks.append(hook)

# ============================================================================
# Image Utilities
# ============================================================================

# Initialize with random noise for generating from scratch
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0

def showarray(a, fname='out_pytorch.jpg'):
    """
    Save a numpy image array to file.
    
    Args:
        a: numpy array with values in [0, 1] or [0, 255]
        fname: output filename
    """
    a = np.uint8(np.clip(a, 0, 1) * 255)
    img = PIL.Image.fromarray(a)
    img.save(fname)
    logger.info(f'Saved image to {fname}')

def visstd(a, s=0.1):
    """
    Normalize image array for visualization.
    Centers the values around 0.5 with controlled standard deviation.
    
    Args:
        a: image array
        s: scale factor for standard deviation
    
    Returns:
        normalized image in [0, 1] range
    """
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5

def img_to_tensor(img_np):
    """
    Convert numpy image array to normalized PyTorch tensor.
    
    Args:
        img_np: numpy array [H, W, 3] with values in [0, 255]
    
    Returns:
        tensor [1, 3, H, W] normalized for ImageNet, ready for model input
    """
    img_tensor = torch.from_numpy(img_np).float().to(device)
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    img_tensor = (img_tensor - imagenet_mean) / imagenet_std
    return img_tensor

def tensor_to_img(img_tensor):
    """
    Convert PyTorch tensor back to numpy image array.
    
    Args:
        img_tensor: tensor [1, 3, H, W]
    
    Returns:
        numpy array [H, W, 3]
    """
    return img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

def resize(img, size):
    """
    Resize image using scipy interpolation.
    
    Args:
        img: input image array
        size: target (height, width)
    
    Returns:
        resized image array
    """
    factors = [1.0 * size[0] / img.shape[0], 1.0 * size[1] / img.shape[1]]
    factors += [1.0] * (len(img.shape) - 2)
    return scipy.ndimage.zoom(img, factors, order=1)

def compute_layer_activation(img_tensor, layer_name):
    """
    Forward pass through the model and retrieve a specific layer's activation.
    
    Args:
        img_tensor: input tensor [1, 3, H, W]
        layer_name: name of layer to extract output from
    
    Returns:
        layer activation tensor
    """
    layer_outputs.clear()
    # Note: we do NOT use torch.no_grad() here because we need gradients
    # to flow backward through the model for the DeepDream algorithm
    model(img_tensor)
    
    if layer_name in layer_outputs:
        return layer_outputs[layer_name]
    
    # Fallback: try partial name matching
    for key in layer_outputs.keys():
        if layer_name in key:
            return layer_outputs[key]
    
    raise ValueError(f"Layer {layer_name} not found in model")


# ============================================================================
# Gradient Computation
# ============================================================================

def calc_grad_tiled(img, layer_name, tile_size=512):
    """
    Compute gradients of layer activation with respect to input image.
    
    Uses a tiled approach to manage memory on GPUs with limited VRAM.
    Random shifts are applied to blur tile boundaries and reduce artifacts.
    
    Args:
        img: input image array [H, W, 3]
        layer_name: target layer name for activation maximization
        tile_size: size of processing tiles
    
    Returns:
        gradient array same shape as img
    """
    sz = tile_size
    h, w = img.shape[:2]
    
    # Apply random shift to reduce boundary artifacts
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    
    # Process image in tiles
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            # Extract tile
            sub = img_shift[y:y+sz, x:x+sz]
            
            # Convert to tensor and enable gradient computation
            img_tensor = img_to_tensor(sub)
            img_tensor.requires_grad_(True)
            
            # Forward pass through model
            layer_output = compute_layer_activation(img_tensor, layer_name)
            
            # Loss: maximize mean activation of the target layer
            loss = layer_output.mean()
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Extract gradient and convert back to numpy
            g = img_tensor.grad.data
            g = tensor_to_img(g)
            
            # Store in gradient array
            grad[y:y+sz, x:x+sz] = g
    
    # Undo the shift to align gradient with original image
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

# ============================================================================
# DeepDream Rendering Functions
# ============================================================================

def render_deepdream(layer_name, img0=img_noise,
                     iter_n=50, step=1.4, octave_n=5, octave_scale=1.3,
                     output='deepdream_pytorch.jpg'):
    """
    Generate DeepDream image using multi-scale octave processing.
    
    This is the main algorithm that produces high-quality results.
    The multi-scale approach processes the image from coarse to fine,
    allowing the algorithm to enhance patterns at multiple scales.
    
    Args:
        layer_name: which layer activations to maximize
        img0: starting image (defaults to random noise)
        iter_n: iterations per octave
        step: gradient ascent step size
        octave_n: number of octaves (scales) to process
        octave_scale: scaling factor between octaves
        output: output filename
    
    Returns:
        final generated image array
    """
    img = img0.copy()
    octaves = []
    
    # Build Laplacian pyramid: decompose into coarse + details at each scale
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)
    
    # Process from coarse to fine resolution
    for octave in range(octave_n):
        if octave > 0:
            # Combine low-frequency from previous octave with high-frequency details
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        
        logger.info(f'Processing octave {octave + 1}/{octave_n}')
        
        # Gradient ascent iterations at this scale
        for i in range(iter_n):
            # Compute gradient of layer activation
            g = calc_grad_tiled(img, layer_name, tile_size=256)
            
            # Normalize gradient to avoid explosions
            g /= g.std() + 1e-8
            
            # Update image in direction of increasing activation
            img += g * step
            
            if i % 10 == 0:
                logger.debug(f'  Iteration {i + 1}/{iter_n}')
    
    # Save result
    showarray(img / 255.0, output)
    return img

def render_naive(layer_name, img0=img_noise, iter_n=20, step=1.0):
    """
    Simple DeepDream without multi-scale processing (faster but lower quality).
    
    This is a baseline method for comparison.
    
    Args:
        layer_name: which layer activations to maximize
        img0: starting image
        iter_n: number of iterations
        step: gradient step size
    
    Returns:
        generated image array
    """
    img = img0.copy()
    
    for i in range(iter_n):
        # Convert image to tensor
        img_tensor = img_to_tensor(img)
        img_tensor.requires_grad_(True)
        
        # Forward pass
        layer_output = compute_layer_activation(img_tensor, layer_name)
        
        # Maximize layer activation
        loss = layer_output.mean()
        
        # Backward pass to get gradients
        loss.backward()
        
        # Extract gradient
        g = img_tensor.grad.data
        g = tensor_to_img(g)
        
        # Normalize and apply gradient
        g /= g.std() + 1e-8
        img += g * step
        
        if i % 5 == 0:
            logger.debug(f'Iteration {i + 1}/{iter_n}, Loss: {loss.item():.6f}')
    
    # Save result
    showarray(visstd(img), 'deepdream_naive_pytorch.jpg')
    return img


# ============================================================================
# Main Script
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch DeepDream: Generate psychedelic images by enhancing neural network patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage with random noise
  python test_pytorch.py
  
  # Apply DeepDream to an input image
  python test_pytorch.py --content myimage.jpg --output result.jpg
  
  # Use different layer and more iterations
  python test_pytorch.py --layer Mixed_7c --iterations 30 --octaves 4
        '''
    )
    
    parser.add_argument('--content', default='source.jpg', 
                        help='Source image to modify (default: random noise)')
    parser.add_argument('--output', default='deepdream_pytorch.jpg',
                        help='Output filename (default: deepdream_pytorch.jpg)')
    parser.add_argument('--layer', default='Mixed_5c', choices=target_layers,
                        help='Which network layer to maximize activations for')
    parser.add_argument('--iterations', type=int, default=15,
                        help='Iterations per octave (default: 15)')
    parser.add_argument('--step', type=float, default=0.5,
                        help='Gradient step size (default: 0.5)')
    parser.add_argument('--octaves', type=int, default=3,
                        help='Number of octaves/scales to process (default: 3)')
    
    args = parser.parse_args()

    # Validate arguments
    if args.iterations < 1 or args.octaves < 1 or args.step <= 0:
        parser.error('--iterations and --octaves must be positive and --step must be greater than zero')

    # Load input image
    if os.path.exists(args.content):
        logger.info(f'Loading content image from: {args.content}')
        img0 = np.float32(PIL.Image.open(args.content).convert('RGB'))
    else:
        logger.warning(f'Content image not found: {args.content}')
        logger.info('Generating from random noise instead')
        img0 = img_noise.copy()
    
    logger.info(f'Input image shape: {img0.shape}')
    showarray(img0 / 255.0, 'original_image_pytorch.jpg')
    
    # Generate DeepDream
    logger.info(f'Generating DeepDream for layer: {args.layer}')
    logger.info(f'Parameters: iterations={args.iterations}, step={args.step}, octaves={args.octaves}')
    
    render_deepdream(
        args.layer,
        img0,
        iter_n=args.iterations,
        step=args.step,
        octave_n=args.octaves,
        octave_scale=1.3,
        output=args.output,
    )
    
    logger.info(f'DeepDream image saved to: {args.output}')
    logger.info('DeepDream generation complete!')
