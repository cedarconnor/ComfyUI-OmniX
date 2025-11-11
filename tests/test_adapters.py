"""
Unit tests for OmniX adapter management
"""

import unittest
import torch
import tempfile
import os
from pathlib import Path
import safetensors.torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from omnix.adapters import AdapterManager, AdapterModule, OmniXAdapters


class TestAdapterModule(unittest.TestCase):
    """Test AdapterModule functionality"""

    def test_adapter_creation(self):
        """Test creating adapter from state dict"""
        state_dict = {
            'weight1': torch.randn(128, 256),
            'weight2': torch.randn(256, 512),
        }

        adapter = AdapterModule(state_dict, dtype=torch.float32)

        self.assertTrue(hasattr(adapter, 'weight1'))
        self.assertTrue(hasattr(adapter, 'weight2'))
        self.assertEqual(adapter.weight1.shape, (128, 256))
        self.assertEqual(adapter.weight2.dtype, torch.float32)

    def test_adapter_forward(self):
        """Test adapter forward pass"""
        state_dict = {'dummy': torch.zeros(1)}
        adapter = AdapterModule(state_dict)

        # Forward should pass through input unchanged (placeholder behavior)
        input_tensor = torch.randn(1, 3, 64, 64)
        output = adapter(input_tensor)

        self.assertEqual(input_tensor.shape, output.shape)
        torch.testing.assert_close(input_tensor, output)


class TestAdapterManager(unittest.TestCase):
    """Test AdapterManager functionality"""

    def setUp(self):
        """Create temporary adapter directory with mock weights"""
        self.temp_dir = tempfile.mkdtemp()
        self.adapter_dir = Path(self.temp_dir) / "omnix-base"
        self.adapter_dir.mkdir(parents=True)

        # Create mock adapter weights
        self.create_mock_adapter('rgb_generation')
        self.create_mock_adapter('distance')

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_mock_adapter(self, adapter_type: str):
        """Create mock adapter weights file"""
        state_dict = {
            f'{adapter_type}_weight': torch.randn(256, 256),
            f'{adapter_type}_bias': torch.randn(256),
        }
        path = self.adapter_dir / f"{adapter_type}_adapter.safetensors"
        safetensors.torch.save_file(state_dict, str(path))

    def test_manager_initialization(self):
        """Test adapter manager initialization"""
        manager = AdapterManager(str(self.adapter_dir), dtype=torch.float16)

        self.assertEqual(manager.adapter_dir, self.adapter_dir)
        self.assertEqual(manager.dtype, torch.float16)

    def test_manager_invalid_directory(self):
        """Test manager with invalid directory"""
        with self.assertRaises(FileNotFoundError):
            AdapterManager("/nonexistent/path")

    def test_load_adapter(self):
        """Test loading adapter"""
        manager = AdapterManager(str(self.adapter_dir))
        adapter = manager.get_adapter('rgb_generation')

        self.assertIsInstance(adapter, AdapterModule)
        self.assertTrue(hasattr(adapter, 'rgb_generation_weight'))

    def test_adapter_caching(self):
        """Test that adapters are cached"""
        manager = AdapterManager(str(self.adapter_dir))

        # Load adapter twice
        adapter1 = manager.get_adapter('rgb_generation')
        adapter2 = manager.get_adapter('rgb_generation')

        # Should be the same object (cached)
        self.assertIs(adapter1, adapter2)

    def test_list_available_adapters(self):
        """Test listing available adapters"""
        manager = AdapterManager(str(self.adapter_dir))
        available = manager.list_available_adapters()

        self.assertIn('rgb_generation', available)
        self.assertIn('distance', available)
        self.assertEqual(len(available), 2)

    def test_unload_adapter(self):
        """Test unloading adapter from memory"""
        manager = AdapterManager(str(self.adapter_dir))

        # Load and then unload
        manager.get_adapter('rgb_generation')
        self.assertIn('rgb_generation', manager._loaded_adapters)

        manager.unload_adapter('rgb_generation')
        self.assertNotIn('rgb_generation', manager._loaded_adapters)


class TestOmniXAdapters(unittest.TestCase):
    """Test high-level OmniXAdapters interface"""

    def setUp(self):
        """Create temporary adapter setup"""
        self.temp_dir = tempfile.mkdtemp()
        self.adapter_dir = Path(self.temp_dir) / "omnix-base"
        self.adapter_dir.mkdir(parents=True)

        # Create mock adapter
        state_dict = {'test_weight': torch.randn(128, 128)}
        path = self.adapter_dir / "rgb_generation_adapter.safetensors"
        safetensors.torch.save_file(state_dict, str(path))

        self.manager = AdapterManager(str(self.adapter_dir))
        self.adapters = OmniXAdapters(self.manager)

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_get_adapter(self):
        """Test getting adapter through high-level interface"""
        adapter = self.adapters.get_adapter('rgb_generation')
        self.assertIsInstance(adapter, AdapterModule)

    def test_list_available(self):
        """Test listing available adapters"""
        available = self.adapters.list_available()
        self.assertIn('rgb_generation', available)


if __name__ == '__main__':
    unittest.main()
