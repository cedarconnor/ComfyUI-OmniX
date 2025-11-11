"""
Unit tests for OmniX perception engine
"""

import unittest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from omnix.perceiver import PanoramaEncoder, SimplePanoramaEncoder


class TestPanoramaEncoder(unittest.TestCase):
    """Test PanoramaEncoder"""

    def test_encoder_initialization(self):
        """Test encoder initialization"""
        encoder = PanoramaEncoder(
            in_channels=3,
            feature_channels=256,
            dtype=torch.float32
        )

        self.assertEqual(encoder.feature_channels, 256)
        self.assertEqual(encoder.dtype, torch.float32)

    def test_encoder_forward_comfyui_format(self):
        """Test encoder with ComfyUI image format (B, H, W, C)"""
        encoder = PanoramaEncoder()

        # ComfyUI format
        panorama = torch.randn(1, 512, 1024, 3)

        features = encoder(panorama)

        # Check output shape
        self.assertEqual(len(features.shape), 4)
        self.assertEqual(features.shape[0], 1)  # Batch size
        self.assertEqual(features.shape[1], 256)  # Feature channels

    def test_encoder_forward_torch_format(self):
        """Test encoder with PyTorch image format (B, C, H, W)"""
        encoder = PanoramaEncoder()

        # PyTorch format
        panorama = torch.randn(1, 3, 512, 1024)

        features = encoder(panorama)

        # Check output
        self.assertEqual(len(features.shape), 4)
        self.assertEqual(features.shape[0], 1)

    def test_encoder_dtype_handling(self):
        """Test encoder handles different dtypes"""
        encoder = PanoramaEncoder(dtype=torch.float16)

        panorama = torch.randn(1, 256, 512, 3).to(torch.float32)

        features = encoder(panorama)

        # Output should be in encoder's dtype
        self.assertEqual(features.dtype, torch.float16)


class TestSimplePanoramaEncoder(unittest.TestCase):
    """Test SimplePanoramaEncoder"""

    def test_simple_encoder_initialization(self):
        """Test simple encoder initialization"""
        encoder = SimplePanoramaEncoder(dtype=torch.float32)

        self.assertEqual(encoder.dtype, torch.float32)

    def test_simple_encoder_forward(self):
        """Test simple encoder forward pass"""
        encoder = SimplePanoramaEncoder()

        panorama = torch.randn(1, 256, 512, 3)

        features = encoder(panorama)

        # Check output
        self.assertEqual(len(features.shape), 4)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[1], 256)  # Output channels

    def test_simple_encoder_batch(self):
        """Test simple encoder with batch"""
        encoder = SimplePanoramaEncoder()

        panorama = torch.randn(4, 256, 512, 3)  # Batch of 4

        features = encoder(panorama)

        self.assertEqual(features.shape[0], 4)


if __name__ == '__main__':
    unittest.main()
