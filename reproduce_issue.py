
import torch
import sys
import os

# Add the custom node directory to path so we can import modules
sys.path.append("c:\\ComfyUI\\custom_nodes\\ComfyUI-OmniX")
# sys.path.append("c:\\ComfyUI")

from unittest.mock import MagicMock
sys.modules["folder_paths"] = MagicMock()

# Load omni_x_utils first
import omni_x_utils
sys.modules["omni_x_utils"] = omni_x_utils

# Load omni_x_nodes with patched import
with open("c:\\ComfyUI\\custom_nodes\\ComfyUI-OmniX\\omni_x_nodes.py", "r", encoding="utf-8") as f:
    code = f.read()
    code = code.replace("from . import omni_x_utils", "import omni_x_utils")

import types
omni_x_nodes = types.ModuleType("omni_x_nodes")
exec(code, omni_x_nodes.__dict__)

OmniX_PanoPerception_Normal = omni_x_nodes.OmniX_PanoPerception_Normal

class MockVAE:
    def decode(self, latents):
        # Simulate VAE decoding
        # Latents: (B, 16, H, W) -> Decoded: (B, C, H*8, W*8)
        # The error is (1, 1, 1024) in SaveImage, which implies the output tensor was (1, 1024, 1, 1) or similar?
        # Wait, the error "Cannot handle this data type: (1, 1, 1024)" from PIL means the array shape is (1, 1, 1024).
        # PIL.Image.fromarray((1, 1, 1024)) -> Wait, PIL expects (H, W, C).
        # If the tensor is (1, 1024, 1, 1) and we permute it...
        
        # Let's look at process_normal:
        # normals = decoded.detach().cpu().float()  -> (B, C, H, W)
        # normal_map = normals[i].permute(1, 2, 0).numpy() -> (H, W, C)
        
        # If C=1024, H=1, W=1. Then normal_map is (1, 1, 1024).
        # Then normal_vis = (normal_map + 1.0) / 2.0
        # output = torch.stack(...) -> (B, H, W, C) = (1, 1, 1, 1024)
        
        # SaveImage receives this.
        # SaveImage does: img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        # i is (1, 1, 1024).
        # PIL says "Cannot handle this data type".
        
        # So we need to simulate a VAE decode that returns (B, 1024, 1, 1).
        return torch.randn(1, 1024, 1, 1)

def reproduce():
    print("Reproducing issue...")
    node = OmniX_PanoPerception_Normal()
    
    vae = MockVAE()
    samples = {"samples": torch.randn(1, 16, 1, 1)} # Dummy latents
    
    try:
        # This should return a tensor of shape (1, 1, 1, 1024)
        result = node.process_normal(vae, samples)
        output_tensor = result[0]
        print(f"Node output shape: {output_tensor.shape}")
        
        # Simulate SaveImage behavior
        import numpy as np
        from PIL import Image
        
        # ComfyUI passes individual images from the batch to SaveImage
        img_tensor = output_tensor[0] # (1, 1, 1024)
        img_np = (img_tensor.numpy() * 255).astype(np.uint8)
        
        print(f"Image numpy shape: {img_np.shape}")
        
        # This should crash
        Image.fromarray(img_np)
        print("Did not crash! (Unexpected)")
        
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {e}")

if __name__ == "__main__":
    reproduce()
