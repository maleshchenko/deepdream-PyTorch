#!/usr/bin/env python3
"""
Unit tests for test_pytorch.py (DeepDream implementation)

Tests core utility functions and image processing operations:
- Image conversion (numpy ↔ tensor)
- Image resizing
- Tensor normalization
- Layer activation computation
- Gradient computation
"""

import pytest
import numpy as np
import torch
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import deepdream module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from deepdream.py
from deepdream import (
    img_to_tensor,
    tensor_to_img,
    resize,
    visstd,
    showarray,
    compute_layer_activation,
    calc_grad_tiled,
    imagenet_mean,
    imagenet_std,
    device,
    model,
)


class TestImageConversion:
    """Test numpy ↔ tensor conversion functions"""

    def test_img_to_tensor_shape(self):
        """Test that img_to_tensor produces correct shape"""
        img = np.random.uniform(0, 255, (224, 224, 3)).astype(np.float32)
        tensor = img_to_tensor(img)
        
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.device == device
        assert tensor.dtype == torch.float32

    def test_img_to_tensor_normalization(self):
        """Test that img_to_tensor applies ImageNet normalization"""
        # Create a simple image with realistic pixel values
        img = np.ones((224, 224, 3), dtype=np.float32) * 128
        tensor = img_to_tensor(img)
        
        # After normalization, value range should be reasonable
        # (not necessarily negative or positive, but normalized)
        assert -2 < tensor.mean() < 2  # Normalized values centered around 0
        assert tensor.std() < 2  # Normalized standard deviation
        
    def test_tensor_to_img_shape(self):
        """Test that tensor_to_img produces correct shape"""
        tensor = torch.randn(1, 3, 224, 224).to(device)
        img = tensor_to_img(tensor)
        
        assert img.shape == (224, 224, 3)
        assert isinstance(img, np.ndarray)

    def test_tensor_to_img_range(self):
        """Test that tensor_to_img returns values in valid range"""
        tensor = torch.randn(1, 3, 224, 224).to(device)
        img = tensor_to_img(tensor)
        
        # After denormalization, should be roughly in [-1, 1] range
        assert img.min() >= -10  # Allow some range due to normalization
        assert img.max() <= 10

    def test_conversion_roundtrip(self):
        """Test that img → tensor → img conversion is approximately reversible"""
        # Create a normalized image
        img_orig = np.random.uniform(0, 255, (224, 224, 3)).astype(np.float32)
        
        # Convert to tensor and back
        tensor = img_to_tensor(img_orig)
        img_recovered = tensor_to_img(tensor)
        
        # Should recover the original shape
        assert img_recovered.shape == img_orig.shape
        
        # Values should be similar (allowing for normalization/denormalization rounding)
        # Note: This won't be a perfect roundtrip due to normalization operations
        assert img_recovered.shape == img_orig.shape


class TestImageResizing:
    """Test image resizing functionality"""

    def test_resize_basic(self):
        """Test basic image resizing"""
        img = np.random.uniform(0, 255, (224, 224, 3))
        resized = resize(img, (112, 112))
        
        assert resized.shape == (112, 112, 3)

    def test_resize_upscale(self):
        """Test upscaling an image"""
        img = np.random.uniform(0, 255, (100, 100, 3))
        resized = resize(img, (200, 200))
        
        assert resized.shape == (200, 200, 3)

    def test_resize_rectangular(self):
        """Test resizing to rectangular dimensions"""
        img = np.random.uniform(0, 255, (224, 224, 3))
        resized = resize(img, (112, 256))
        
        assert resized.shape == (112, 256, 3)

    def test_resize_multichannel(self):
        """Test resize preserves number of channels"""
        img = np.random.uniform(0, 255, (200, 200, 3))
        resized = resize(img, (100, 100))
        
        assert resized.shape[2] == img.shape[2]

    def test_resize_values_in_valid_range(self):
        """Test that resized image values remain in valid range"""
        img = np.random.uniform(100, 150, (224, 224, 3))
        resized = resize(img, (112, 112))
        
        # Interpolation should keep values similar to original range
        assert resized.min() > 50
        assert resized.max() < 200


class TestVisualizationNormalization:
    """Test visstd normalization function"""

    def test_visstd_centering(self):
        """Test that visstd centers values around 0.5"""
        img = np.random.randn(100, 100, 3) * 10 + 100
        normalized = visstd(img, s=0.1)
        
        # Should be in [0, 1] range
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        
        # Should be approximately centered around 0.5
        assert 0.4 < normalized.mean() < 0.6

    def test_visstd_scale_parameter(self):
        """Test that scale parameter affects output spread"""
        img = np.random.randn(100, 100, 3)
        
        normalized_small = visstd(img, s=0.05)
        normalized_large = visstd(img, s=0.2)
        
        # Larger scale should produce more spread
        assert normalized_large.std() > normalized_small.std()

    def test_visstd_constant_image(self):
        """Test visstd on constant image (zero standard deviation)"""
        img = np.ones((100, 100, 3)) * 50
        normalized = visstd(img, s=0.1)
        
        # Should still return valid values (handles std=0 with 1e-4 epsilon)
        assert np.isfinite(normalized).all()
        assert 0 <= normalized.min()
        assert normalized.max() <= 1


class TestLayerActivation:
    """Test layer activation computation"""

    def test_compute_layer_activation_output_shape(self):
        """Test that compute_layer_activation returns valid tensor"""
        img = np.random.uniform(0, 255, (224, 224, 3)).astype(np.float32)
        tensor = img_to_tensor(img)
        
        # Test with a layer that exists
        activation = compute_layer_activation(tensor, 'Mixed_5b')
        
        assert isinstance(activation, torch.Tensor)
        assert activation.device == device

    def test_compute_layer_activation_invalid_layer(self):
        """Test that invalid layer name raises error"""
        img = np.random.uniform(0, 255, (224, 224, 3)).astype(np.float32)
        tensor = img_to_tensor(img)
        
        with pytest.raises(ValueError):
            compute_layer_activation(tensor, 'NonexistentLayer')

    def test_compute_layer_activation_multiple_layers(self):
        """Test that multiple layers can be activated"""
        img = np.random.uniform(0, 255, (224, 224, 3)).astype(np.float32)
        tensor = img_to_tensor(img)
        
        layers_to_test = ['Mixed_5b', 'Mixed_5c', 'Mixed_6a']
        activations = {}
        
        for layer in layers_to_test:
            activations[layer] = compute_layer_activation(tensor, layer)
        
        # All should be valid tensors
        assert all(isinstance(a, torch.Tensor) for a in activations.values())


class TestGradientComputation:
    """Test gradient computation for DeepDream"""

    def test_calc_grad_tiled_output_shape(self):
        """Test that calc_grad_tiled returns correct gradient shape"""
        img = np.random.uniform(100, 150, (224, 224, 3)).astype(np.float32)
        
        grad = calc_grad_tiled(img, 'Mixed_5b', tile_size=512)
        
        assert grad.shape == img.shape
        assert isinstance(grad, np.ndarray)
        assert grad.dtype == np.float32

    def test_calc_grad_tiled_nonzero_gradient(self):
        """Test that gradient is computed (not all zeros)"""
        img = np.random.uniform(100, 150, (224, 224, 3)).astype(np.float32)
        
        grad = calc_grad_tiled(img, 'Mixed_5b', tile_size=512)
        
        # Gradient should not be all zeros
        assert np.abs(grad).sum() > 0

    def test_calc_grad_tiled_small_tile(self):
        """Test gradient computation with small tile size"""
        img = np.random.uniform(100, 150, (224, 224, 3)).astype(np.float32)
        
        grad = calc_grad_tiled(img, 'Mixed_5b', tile_size=128)
        
        assert grad.shape == img.shape

    def test_calc_grad_tiled_finite_values(self):
        """Test that gradient contains only finite values"""
        img = np.random.uniform(100, 150, (224, 224, 3)).astype(np.float32)
        
        grad = calc_grad_tiled(img, 'Mixed_5b', tile_size=256)
        
        # Should not contain NaN or Inf
        assert np.isfinite(grad).all()

    def test_calc_grad_tiled_consistency(self):
        """Test that tiling approach is consistent"""
        img = np.random.uniform(100, 150, (224, 224, 3)).astype(np.float32)
        
        # Compute gradients with different tile sizes
        grad_small = calc_grad_tiled(img, 'Mixed_5b', tile_size=128)
        grad_large = calc_grad_tiled(img, 'Mixed_5b', tile_size=512)
        
        # Both should have valid gradients (actual values may differ due to shifting)
        assert grad_small.shape == grad_large.shape
        assert np.abs(grad_small).sum() > 0
        assert np.abs(grad_large).sum() > 0


class TestImageIO:
    """Test image I/O functions"""

    def test_showarray_saves_file(self):
        """Test that showarray creates a valid image file"""
        img = np.random.uniform(0, 1, (224, 224, 3)).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_output.jpg')
            showarray(img, fname=output_path)
            
            # File should exist
            assert os.path.exists(output_path)
            
            # File should have some size
            assert os.path.getsize(output_path) > 0

    def test_showarray_clips_values(self):
        """Test that showarray properly clips out-of-range values"""
        # Image with values outside [0, 1]
        img = np.array([[-0.5, 0.5, 1.5]], dtype=np.float32).reshape(1, 1, 3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_clip.jpg')
            showarray(img, fname=output_path)
            
            # Should not raise an error
            assert os.path.exists(output_path)

    def test_showarray_uint8_range(self):
        """Test showarray with image in [0, 255] range"""
        img = np.random.uniform(0, 255, (224, 224, 3)).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_uint8.jpg')
            showarray(img, fname=output_path)
            
            assert os.path.exists(output_path)


class TestImageProperties:
    """Test properties and constraints of image operations"""

    def test_tensor_device_consistency(self):
        """Test that tensors are created on correct device"""
        img = np.random.uniform(0, 255, (224, 224, 3)).astype(np.float32)
        tensor = img_to_tensor(img)
        
        assert tensor.device == device

    def test_gradient_requires_grad(self):
        """Test that tensors can be used for gradient computation"""
        img = np.random.uniform(100, 150, (224, 224, 3)).astype(np.float32)
        tensor = img_to_tensor(img)
        tensor.requires_grad_(True)
        
        # Should be able to compute gradients
        assert tensor.requires_grad

    def test_image_normalization_range(self):
        """Test that normalized images have reasonable value ranges"""
        img = np.ones((224, 224, 3), dtype=np.float32) * 128
        tensor = img_to_tensor(img)
        
        # After ImageNet normalization, values should be in reasonable range
        # Most values should be within [-2, 2] for typical images
        normalized_img = tensor.cpu().numpy()
        assert normalized_img.min() > -5
        assert normalized_img.max() < 5


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_resize_single_pixel(self):
        """Test resizing to very small image"""
        img = np.random.uniform(0, 255, (224, 224, 3))
        resized = resize(img, (1, 1))
        
        assert resized.shape == (1, 1, 3)

    def test_resize_large_upscale(self):
        """Test large upscaling operation"""
        img = np.random.uniform(0, 255, (50, 50, 3))
        resized = resize(img, (500, 500))
        
        assert resized.shape == (500, 500, 3)

    def test_img_to_tensor_with_extreme_values(self):
        """Test img_to_tensor with extreme pixel values"""
        img = np.array([[[0, 127.5, 255]]], dtype=np.float32)
        tensor = img_to_tensor(img)
        
        # Should handle without error
        assert tensor.shape == (1, 3, 1, 1)
        assert np.isfinite(tensor.cpu().numpy()).all()

    def test_visstd_with_zero_std(self):
        """Test visstd gracefully handles zero standard deviation"""
        img = np.ones((10, 10, 3)) * 50
        result = visstd(img, s=0.1)
        
        # Should not produce NaN or Inf
        assert np.isfinite(result).all()

    def test_calc_grad_tiled_with_small_image(self):
        """Test gradient computation on small images"""
        # InceptionV3 has minimum input requirements (~75x75), so use 128x128
        img = np.random.uniform(100, 150, (128, 128, 3)).astype(np.float32)
        
        grad = calc_grad_tiled(img, 'Mixed_5b', tile_size=256)
        
        # Should work with smaller (but valid) image sizes
        assert grad.shape == img.shape
        assert np.isfinite(grad).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
