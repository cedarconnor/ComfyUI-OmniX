"""
OmniX Panorama Generator

Handles text-to-panorama and image-to-panorama generation using Flux + OmniX.
Implements the full generation pipeline including text encoding, conditioning,
diffusion sampling, and VAE decoding.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from .model_loader import OmniXModelLoader

# Import ComfyUI's sampling utilities
import comfy.sample
import comfy.samplers
import comfy.model_management as mm


class OmniXPanoramaGenerator:
    """
    OmniX Panorama Generation Pipeline

    Generates 360° equirectangular panoramas from:
    - Text prompts (text-to-panorama)
    - Text + reference image (image-to-panorama)
    - Text + depth/normal maps (controlled generation)

    Uses Flux.1-dev + OmniX RGB generation adapter.
    """

    def __init__(self, model_loader: OmniXModelLoader):
        """
        Initialize panorama generator.

        Args:
            model_loader: OmniXModelLoader with Flux model and adapters
        """
        self.model_loader = model_loader
        self.model = model_loader.get_model()
        self.vae = model_loader.get_vae()
        self.clip = getattr(model_loader, 'clip', None)

        # Ensure RGB generation adapter is applied
        if 'rgb_generation' not in model_loader.active_adapters:
            model_loader.apply_adapter('rgb_generation', strength=1.0)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = -1,
        steps: int = 20,
        cfg: float = 3.5,
        sampler_name: str = "euler",
        scheduler: str = "simple",
        width: int = 1024,
        height: int = 512,
        denoise: float = 1.0,
        conditioning_image: Optional[torch.Tensor] = None,
        conditioning_strength: float = 0.8,
    ) -> torch.Tensor:
        """
        Generate a panorama from text prompt.

        Args:
            prompt: Text description of the scene
            negative_prompt: Negative prompt for guidance
            seed: Random seed (-1 for random)
            steps: Number of diffusion steps
            cfg: Classifier-free guidance scale
            sampler_name: Sampler algorithm
            scheduler: Noise scheduler
            width: Output width (should be 2x height for equirectangular)
            height: Output height
            denoise: Denoising strength (1.0 = full generation)
            conditioning_image: Optional reference image for conditioning
            conditioning_strength: Strength of image conditioning

        Returns:
            Generated panorama tensor in ComfyUI format (B, H, W, C)
        """
        # Set seed
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        print(f"Generating panorama...")
        print(f"  Prompt: {prompt}")
        print(f"  Size: {width}x{height}")
        print(f"  Steps: {steps}, CFG: {cfg}")
        print(f"  Seed: {seed}")

        # Validate aspect ratio (panoramas should be 2:1)
        aspect_ratio = width / height
        if not (1.8 <= aspect_ratio <= 2.2):
            print(f"⚠️  Warning: Aspect ratio {aspect_ratio:.2f} is not standard for panoramas (2:1)")

        # Encode text prompt
        if self.clip is None:
            raise RuntimeError("CLIP model not available. Cannot encode text prompt.")

        # Encode positive prompt
        tokens = self.clip.tokenize(prompt)
        cond, pooled = self.clip.encode_from_tokens(tokens, return_pooled=True)
        positive_cond = [[cond, {"pooled_output": pooled}]]

        # Encode negative prompt
        if negative_prompt:
            neg_tokens = self.clip.tokenize(negative_prompt)
            neg_cond, neg_pooled = self.clip.encode_from_tokens(neg_tokens, return_pooled=True)
            negative_cond = [[neg_cond, {"pooled_output": neg_pooled}]]
        else:
            # Empty conditioning
            negative_cond = [[torch.zeros_like(cond), {"pooled_output": torch.zeros_like(pooled)}]]

        # Prepare latent
        # Flux works in latent space, so we need to create initial noise
        latent_width = width // 8  # VAE downsampling factor
        latent_height = height // 8
        batch_size = 1

        # Create noise tensor
        device = mm.get_torch_device()
        latent = torch.randn(
            (batch_size, 4, latent_height, latent_width),  # Flux uses 4 latent channels
            device=device,
            dtype=torch.float32
        )

        # If conditioning image is provided, encode it
        if conditioning_image is not None:
            print(f"  Using image conditioning (strength: {conditioning_strength:.2f})")

            # conditioning_image is already a tensor in ComfyUI format (B, H, W, C)
            # Convert to (B, C, H, W) for processing
            if conditioning_image.ndim == 4 and conditioning_image.shape[-1] == 3:
                conditioning_image = conditioning_image.permute(0, 3, 1, 2)

            conditioning_image = conditioning_image.to(device)

            # Resize to match target size
            if conditioning_image.shape[2:] != (height, width):
                conditioning_image = torch.nn.functional.interpolate(
                    conditioning_image,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                )

            # Encode with VAE
            with torch.no_grad():
                conditioning_latent = self.vae.encode(conditioning_image)

            # Blend noise with conditioning latent based on strength
            # Higher strength = more influence from conditioning image
            noise_strength = 1.0 - conditioning_strength
            latent = noise_strength * latent + conditioning_strength * conditioning_latent

        # Prepare latent dict for ComfyUI sampler
        latent_image = {"samples": latent}

        # Run sampling
        print("  Running diffusion sampling...")

        samples = comfy.sample.sample(
            model=self.model,
            noise=latent,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive_cond,
            negative=negative_cond,
            latent_image=latent_image,
            denoise=denoise,
            seed=seed
        )

        # Decode latent to image
        print("  Decoding latent to image...")

        with torch.no_grad():
            decoded = self.vae.decode(samples)

        # decoded is already a tensor in (B, C, H, W) format
        # Convert to ComfyUI format (B, H, W, C)
        if decoded.ndim == 4 and decoded.shape[1] == 3:
            output = decoded.permute(0, 2, 3, 1)
        else:
            output = decoded

        print(f"✓ Generated panorama: {output.shape}")

        return output

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> torch.Tensor:
        """
        Generate multiple panoramas from a list of prompts.

        Args:
            prompts: List of text prompts
            **kwargs: Additional arguments passed to generate()

        Returns:
            Batch of panoramas (B, H, W, C)
        """
        results = []

        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] Generating panorama for: {prompt}")

            # Generate single panorama
            panorama = self.generate(prompt=prompt, **kwargs)

            results.append(panorama)

        # Stack into batch
        batch = torch.cat(results, dim=0)

        print(f"\n✓ Generated {len(prompts)} panoramas")

        return batch

    def text_to_panorama(
        self,
        prompt: str,
        width: int = 2048,
        height: int = 1024,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate panorama from text prompt (convenience method).

        Args:
            prompt: Text description
            width: Output width (default: 2048)
            height: Output height (default: 1024)
            **kwargs: Additional generation parameters

        Returns:
            Generated panorama (1, H, W, C)
        """
        return self.generate(
            prompt=prompt,
            width=width,
            height=height,
            **kwargs
        )

    def image_to_panorama(
        self,
        prompt: str,
        image: torch.Tensor,
        conditioning_strength: float = 0.8,
        width: int = 2048,
        height: int = 1024,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate panorama from text + reference image (convenience method).

        Args:
            prompt: Text description
            image: Reference image (B, H, W, C)
            conditioning_strength: How much to follow the reference image
            width: Output width
            height: Output height
            **kwargs: Additional generation parameters

        Returns:
            Generated panorama (1, H, W, C)
        """
        return self.generate(
            prompt=prompt,
            conditioning_image=image,
            conditioning_strength=conditioning_strength,
            width=width,
            height=height,
            **kwargs
        )


def create_generator(
    flux_checkpoint: str = "flux1-dev.safetensors",
    adapter_preset: str = "omnix-base",
    dtype: torch.dtype = torch.float16
) -> OmniXPanoramaGenerator:
    """
    Create a panorama generator (convenience function).

    Args:
        flux_checkpoint: Flux checkpoint name
        adapter_preset: OmniX adapter preset
        dtype: Inference dtype

    Returns:
        OmniXPanoramaGenerator ready for generation
    """
    from .model_loader import load_omnix_model

    # Load model
    model_loader = load_omnix_model(
        flux_checkpoint=flux_checkpoint,
        adapter_preset=adapter_preset,
        dtype=dtype
    )

    # Create generator
    generator = OmniXPanoramaGenerator(model_loader)

    print("✓ OmniX panorama generator ready")

    return generator
