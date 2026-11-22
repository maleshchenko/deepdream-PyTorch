#!/usr/bin/env python3
"""
PyTorch DeepDream Implementation

Generates DeepDream images by:
1. Loading a pre-trained InceptionV3 model
2. Iteratively computing gradients of one layer's activations with respect to the image
3. Using multi-scale octave processing (Laplacian pyramid) for better visual results

The image is kept as a float tensor in [0, 255] (shape [1, 3, H, W]) on the compute
device for the whole run, so there are no CPU/GPU round trips inside the loop.
"""

import argparse
import logging
import os

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# InceptionV3 fails on inputs smaller than this
MIN_SIZE = 75

# Mixed blocks that can be targeted (names match torchvision's Inception3 attributes)
TARGET_LAYERS = [
    'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
    'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
    'Mixed_7a', 'Mixed_7b', 'Mixed_7c',
]

DEFAULT_ITERATIONS = 10
DEFAULT_STEP = 1.5
DEFAULT_OCTAVES = 3
DEFAULT_OCTAVE_SCALE = 1.3
DEFAULT_TILE_SIZE = 256


# ============================================================================
# Model
# ============================================================================

class _StopForward(Exception):
    """Raised by the hook to stop the forward pass once the target layer has run."""


class DreamModel:
    """InceptionV3 wrapper that returns the activation of one Mixed block.

    A forward hook is attached only to the requested layer, and it halts the
    forward pass right after that layer, so nothing deeper in the network
    (later Mixed blocks, pooling, classifier) is computed.
    """

    def __init__(self, device=None):
        from torchvision import models  # imported here so the module imports without torchvision

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')

        # With pretrained weights torchvision enables transform_input, so feeding
        # ImageNet-normalized input (below) is the correct preprocessing.
        net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        net = net.to(self.device).eval()
        for param in net.parameters():
            param.requires_grad_(False)
        self.net = net

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def normalize(self, x):
        """[0, 255] pixels -> ImageNet-normalized network input (differentiable)."""
        return (x / 255.0 - self.mean) / self.std

    def activation(self, x, layer_name):
        """Run `x` ([1, 3, H, W], values in [0, 255]) up to `layer_name` and return its output."""
        if layer_name not in TARGET_LAYERS:
            raise ValueError(f"Unknown layer {layer_name!r}; choose from {TARGET_LAYERS}")

        store = {}

        def hook(module, inputs, output):
            store['out'] = output
            raise _StopForward

        handle = getattr(self.net, layer_name).register_forward_hook(hook)
        try:
            try:
                self.net(self.normalize(x))
            except _StopForward:
                pass
        finally:
            handle.remove()

        if 'out' not in store:
            raise RuntimeError(f"Layer {layer_name} was never reached")
        return store['out']


# ============================================================================
# Image utilities
# ============================================================================

def make_noise(size=224):
    """Near-flat gray noise to dream from when no content image is given."""
    return np.random.uniform(size=(size, size, 3)).astype(np.float32) + 100.0


def to_tensor(img_np, device):
    """numpy [H, W, 3] in [0, 255] -> float tensor [1, 3, H, W]."""
    arr = np.ascontiguousarray(img_np, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def to_image_array(img):
    """float tensor [1, 3, H, W] -> uint8 numpy [H, W, 3] (clamped to [0, 255])."""
    return img.detach().clamp(0, 255).round().to(torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()


def save_image(img, fname):
    PIL.Image.fromarray(to_image_array(img)).save(fname)
    logger.info(f'Saved image to {fname}')


def resize(img, size):
    """Bilinear resize of a [1, C, H, W] tensor to (height, width), antialiased when shrinking."""
    h, w = img.shape[2:]
    shrinking = size[0] < h or size[1] < w
    return F.interpolate(img, size=tuple(size), mode='bilinear',
                         align_corners=False, antialias=shrinking)


# ============================================================================
# Gradient computation
# ============================================================================

def _tile_starts(length, size):
    """Start offsets of tiles covering [0, length) completely.

    The last tile is anchored to the end, so it overlaps the previous one
    instead of leaving an uncovered strip.
    """
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size))
    if starts[-1] + size < length:
        starts.append(length - size)
    return starts


def calc_grad(model, img, layer_name, tile_size=DEFAULT_TILE_SIZE, loss_kind='sum'):
    """Gradient of the layer-activation objective with respect to the image pixels.

    The image is randomly rolled (jitter) so tile boundaries land in different
    places every call, processed in tiles to bound memory, and overlapping tile
    gradients are averaged.
    """
    _, _, h, w = img.shape
    sy, sx = (int(v) for v in np.random.randint(tile_size, size=2))
    shifted = torch.roll(img, shifts=(sy, sx), dims=(2, 3))

    grad = torch.zeros_like(img)
    count = torch.zeros(1, 1, h, w, device=img.device)

    for y in _tile_starts(h, tile_size):
        for x in _tile_starts(w, tile_size):
            tile = shifted[:, :, y:y + tile_size, x:x + tile_size].detach().clone().requires_grad_(True)
            act = model.activation(tile, layer_name)

            # Sum (not mean) so each pixel's gradient doesn't depend on how big its tile is
            loss = act.sum() if loss_kind == 'sum' else (act ** 2).sum()
            g, = torch.autograd.grad(loss, tile)

            th, tw = tile.shape[2:]
            grad[:, :, y:y + th, x:x + tw] += g
            count[:, :, y:y + th, x:x + tw] += 1

    grad = grad / count.clamp_min(1)
    return torch.roll(grad, shifts=(-sy, -sx), dims=(2, 3))


def _max_octaves(h, w, octave_scale):
    """Most octaves whose smallest level is still at least MIN_SIZE pixels."""
    n = 1
    while min(h, w) / octave_scale ** n >= MIN_SIZE:
        n += 1
    return n


def _ascent_step(img, g, step):
    g = g / (g.std() + 1e-8)
    return (img + g * step).clamp(0, 255)


# ============================================================================
# DeepDream rendering
# ============================================================================

def render_deepdream(model, layer_name, img0,
                     iter_n=DEFAULT_ITERATIONS, step=DEFAULT_STEP,
                     octave_n=DEFAULT_OCTAVES, octave_scale=DEFAULT_OCTAVE_SCALE,
                     tile_size=DEFAULT_TILE_SIZE, loss_kind='sum'):
    """Multi-scale DeepDream. `img0` is a [1, 3, H, W] tensor in [0, 255]; returns the same."""
    h, w = img0.shape[2:]
    if min(h, w) < MIN_SIZE:
        raise ValueError(f'Image is {w}x{h}; InceptionV3 needs at least {MIN_SIZE}px per side')

    allowed = _max_octaves(h, w, octave_scale)
    if octave_n > allowed:
        logger.warning(f'Reducing octaves from {octave_n} to {allowed} so the smallest '
                       f'level stays at least {MIN_SIZE}px')
        octave_n = allowed

    # Laplacian pyramid: split into a coarse image plus the detail lost at each scale
    img = img0.clone()
    octaves = []
    for _ in range(octave_n - 1):
        hw = img.shape[2:]
        lo = resize(img, (max(1, round(hw[0] / octave_scale)), max(1, round(hw[1] / octave_scale))))
        octaves.append(img - resize(lo, hw))
        img = lo

    # Coarse to fine
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = (resize(img, hi.shape[2:]) + hi).clamp(0, 255)

        logger.info(f'Processing octave {octave + 1}/{octave_n} ({img.shape[3]}x{img.shape[2]})')
        for i in range(iter_n):
            g = calc_grad(model, img, layer_name, tile_size, loss_kind)
            img = _ascent_step(img, g, step)
            logger.debug(f'  Iteration {i + 1}/{iter_n}')

    return img


def render_naive(model, layer_name, img0, iter_n=20, step=1.0, loss_kind='sum'):
    """Single-scale baseline: whole image per gradient call, no octaves."""
    img = img0.clone()
    tile_size = max(img.shape[2:])
    for i in range(iter_n):
        g = calc_grad(model, img, layer_name, tile_size, loss_kind)
        img = _ascent_step(img, g, step)
        logger.debug(f'Iteration {i + 1}/{iter_n}')
    return img


# ============================================================================
# Main script
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description='PyTorch DeepDream: generate psychedelic images by enhancing neural network patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Dream from random noise
  python deepdream.py

  # Apply DeepDream to an input image
  python deepdream.py --content myimage.jpg --output result.jpg

  # Different layer, more iterations, fewer octaves
  python deepdream.py --content myimage.jpg --layer Mixed_6c --iterations 20 --octaves 2
        ''',
    )
    parser.add_argument('--content', default=None,
                        help='Source image to modify (default: start from random noise)')
    parser.add_argument('--output', default='deepdream_pytorch.jpg',
                        help='Output filename (default: deepdream_pytorch.jpg)')
    parser.add_argument('--layer', default='Mixed_5c', choices=TARGET_LAYERS,
                        help='Network layer whose activations to maximize (default: Mixed_5c)')
    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS,
                        help=f'Iterations per octave (default: {DEFAULT_ITERATIONS})')
    parser.add_argument('--step', type=float, default=DEFAULT_STEP,
                        help=f'Gradient step size (default: {DEFAULT_STEP})')
    parser.add_argument('--octaves', type=int, default=DEFAULT_OCTAVES,
                        help=f'Number of octaves/scales (default: {DEFAULT_OCTAVES})')
    parser.add_argument('--octave-scale', type=float, default=DEFAULT_OCTAVE_SCALE,
                        help=f'Scale factor between octaves (default: {DEFAULT_OCTAVE_SCALE})')
    parser.add_argument('--tile-size', type=int, default=DEFAULT_TILE_SIZE,
                        help=f'Tile size for gradient computation (default: {DEFAULT_TILE_SIZE})')
    parser.add_argument('--loss', choices=['sum', 'squared'], default='sum',
                        help='Objective: summed activations, or summed squared activations '
                             '(favors strongly firing units; default: sum)')
    parser.add_argument('--verbose', action='store_true', help='Log every iteration')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    if args.iterations < 1 or args.octaves < 1:
        parser.error('--iterations and --octaves must be positive')
    if args.step <= 0:
        parser.error('--step must be greater than zero')
    if args.octave_scale <= 1:
        parser.error('--octave-scale must be greater than 1')
    if args.tile_size < MIN_SIZE:
        parser.error(f'--tile-size must be at least {MIN_SIZE}')

    model = DreamModel()

    if args.content and os.path.exists(args.content):
        logger.info(f'Loading content image from: {args.content}')
        img_np = np.asarray(PIL.Image.open(args.content).convert('RGB'), dtype=np.float32)
    else:
        if args.content:
            logger.warning(f'Content image not found: {args.content}')
        logger.info('Generating from random noise')
        img_np = make_noise()

    img0 = to_tensor(img_np, model.device)
    logger.info(f'Input image shape: {tuple(img_np.shape)}')
    save_image(img0, 'original_image_pytorch.jpg')

    logger.info(f'Generating DeepDream for layer: {args.layer}')
    logger.info(f'Parameters: iterations={args.iterations}, step={args.step}, '
                f'octaves={args.octaves}, loss={args.loss}')

    try:
        result = render_deepdream(
            model, args.layer, img0,
            iter_n=args.iterations, step=args.step,
            octave_n=args.octaves, octave_scale=args.octave_scale,
            tile_size=args.tile_size, loss_kind=args.loss,
        )
    except ValueError as e:
        parser.error(str(e))

    save_image(result, args.output)
    logger.info('DeepDream generation complete!')


if __name__ == '__main__':
    main()
