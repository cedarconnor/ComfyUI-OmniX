"""
Unit tests for OmniX utility functions
"""

import unittest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from omnix.utils import (
    to_comfyui_image,
    from_comfyui_image,
    validate_panorama_aspect_ratio,
    visualize_depth_map,
    normalize_normal_map
)


class TestImageConversion(unittest.TestCase):
    """Test image format conversion utilities"""

    def test_pil_to_comfyui(self):
        """Test PIL Image to ComfyUI tensor conversion"""
        # Create test PIL image
        pil_image = Image.new('RGB', (512, 256), color='red')

        tensor = to_comfyui_image(pil_image)

        # Check shape (B, H, W, C)
        self.assertEqual(tensor.shape, (1, 256, 512, 3))
        # Check data type
        self.assertEqual(tensor.dtype, torch.float32)
        # Check value range [0, 1]
        self.assertTrue(tensor.min() >= 0.0)
        self.assertTrue(tensor.max() <= 1.0)

    def test_comfyui_to_pil(self):
        """Test ComfyUI tensor to PIL Image conversion"""
        # Create test tensor
        tensor = torch.ones(1, 256, 512, 3) * 0.5

        pil_image = from_comfyui_image(tensor)

        # Check type
        self.assertIsInstance(pil_image, Image.Image)
        # Check size
        self.assertEqual(pil_image.size, (512, 256))
        # Check mode
        self.assertEqual(pil_image.mode, 'RGB')

    def test_roundtrip_conversion(self):
        """Test PIL -> ComfyUI -> PIL roundtrip"""
        # Create original image
        original = Image.new('RGB', (256, 128))
        pixels = np.random.randint(0, 256, (128, 256, 3), dtype=np.uint8)
        original = Image.fromarray(pixels)

        # Convert to ComfyUI and back
        tensor = to_comfyui_image(original)
        reconstructed = from_comfyui_image(tensor)

        # Check dimensions match
        self.assertEqual(original.size, reconstructed.size)
        self.assertEqual(original.mode, reconstructed.mode)


class TestPanoramaValidation(unittest.TestCase):
    """Test panorama aspect ratio validation"""

    def test_valid_panorama_2_1_ratio(self):
        """Test valid 2:1 panorama"""
        panorama = torch.randn(1, 1024, 2048, 3)

        # Should not raise exception
        try:
            validate_panorama_aspect_ratio(panorama)
        except ValueError:
            self.fail("Valid 2:1 panorama raised ValueError")

    def test_invalid_panorama_ratio(self):
        """Test invalid aspect ratio"""
        panorama = torch.randn(1, 512, 512, 3)  # 1:1 ratio

        # Should raise ValueError
        with self.assertRaises(ValueError):
            validate_panorama_aspect_ratio(panorama)

    def test_valid_panorama_tolerance(self):
        """Test panorama within tolerance"""
        # Slightly off 2:1 but within 5% tolerance
        panorama = torch.randn(1, 1000, 2050, 3)  # ~2.05:1

        try:
            validate_panorama_aspect_ratio(panorama, tolerance=0.1)
        except ValueError:
            self.fail("Panorama within tolerance raised ValueError")


class TestDepthVisualization(unittest.TestCase):
    """Test depth map visualization"""

    def test_visualize_depth_basic(self):
        """Test basic depth visualization"""
        depth = torch.randn(1, 256, 512, 1)

        vis = visualize_depth_map(depth)

        # Check output shape (should add color channels)
        self.assertEqual(vis.shape, (1, 256, 512, 3))
        # Check value range
        self.assertTrue(vis.min() >= 0.0)
        self.assertTrue(vis.max() <= 1.0)

    def test_visualize_depth_range(self):
        """Test depth visualization with known range"""
        # Create depth with known range
        depth = torch.linspace(0, 1, 512).view(1, 1, 512, 1).repeat(1, 256, 1, 1)

        vis = visualize_depth_map(depth)

        # Should produce gradient visualization
        self.assertEqual(vis.shape, (1, 256, 512, 3))


class TestNormalMapNormalization(unittest.TestCase):
    """Test normal map normalization"""

    def test_normalize_normal_map(self):
        """Test normal map normalization"""
        # Create random normals (not unit length)
        normals = torch.randn(1, 256, 512, 3)

        normalized = normalize_normal_map(normals)

        # Check shape preserved
        self.assertEqual(normalized.shape, normals.shape)

        # Check unit length (approximately)
        lengths = torch.norm(normalized, dim=-1)
        self.assertTrue(torch.allclose(lengths, torch.ones_like(lengths), atol=1e-5))

    def test_normalize_converts_range(self):
        """Test that normalization converts to [0, 1] range"""
        normals = torch.randn(1, 64, 128, 3)

        normalized = normalize_normal_map(normals)

        # Output should be in [0, 1] for visualization
        self.assertTrue(normalized.min() >= 0.0)
        self.assertTrue(normalized.max() <= 1.0)


class TestMemoryManagement(unittest.TestCase):
    """Test memory management utilities"""

    def test_cleanup_function(self):
        """Test memory cleanup function"""
        from omnix.utils import cleanup_memory

        # Create some tensors
        tensors = [torch.randn(1000, 1000) for _ in range(10)]

        # Clean up
        try:
            cleanup_memory()
        except Exception as e:
            self.fail(f"cleanup_memory raised exception: {e}")


if __name__ == '__main__':
    unittest.main()
