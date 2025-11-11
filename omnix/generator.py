"""
OmniX Panorama Generator

Implements text-to-panorama and image-to-panorama generation using
OmniX adapters with Flux.1-dev diffusion model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass
from PIL import Image

from .utils import to_comfyui_image, from_comfyui_image


@dataclass
class GenerationConfig:
    """Configuration for panorama generation"""
    # Image dimensions (must be 2:1 ratio for equirectangular)
    width: int = 2048
    height: int = 1024

    # Diffusion parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    # OmniX-specific
    adapter_strength: float = 1.0
    panorama_mode: bool = True  # Enable equirectangular awareness

    # Performance
    batch_size: int = 1
    use_fp16: bool = True

    # Random seed
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        # Check 2:1 aspect ratio
        ratio = self.width / self.height
        if abs(ratio - 2.0) > 0.01:
            raise ValueError(
                f"Panorama must have 2:1 aspect ratio. "
                f"Got {self.width}x{self.height} (ratio: {ratio:.2f})"
            )

        # Check dimensions are reasonable
        if self.width < 512 or self.height < 256:
            raise ValueError(
                f"Dimensions too small: {self.width}x{self.height}. "
                f"Minimum: 512x256"
            )

        if self.width > 4096 or self.height > 2048:
            print(f"Warning: Large dimensions ({self.width}x{self.height}) may require significant VRAM")


class OmniXPanoramaGenerator:
    """
    High-level interface for panorama generation with OmniX.

    This generator wraps ComfyUI's Flux infrastructure and adds
    OmniX-specific panorama generation capabilities.
    """

    def __init__(
        self,
        model: Any,  # ComfyUI MODEL object (with OmniX adapters injected)
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize panorama generator.

        Args:
            model: ComfyUI MODEL with OmniX adapters applied
            config: Generation configuration
        """
        self.model = model
        self.config = config or GenerationConfig()
        self.config.validate()

    def generate_from_text(
        self,
        prompt: str,
        negative_prompt: str = "",
        conditioning: Optional[Any] = None,
        negative_conditioning: Optional[Any] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate panorama from text prompt.

        Args:
            prompt: Text description of desired panorama
            negative_prompt: Things to avoid in generation
            conditioning: Pre-encoded conditioning (optional, from CLIPTextEncode)
            negative_conditioning: Pre-encoded negative conditioning (optional)
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters

        Returns:
            Generated panorama as ComfyUI IMAGE tensor (B, H, W, C)
        """
        # Update config with any runtime parameters
        gen_config = self._update_config(seed=seed, **kwargs)

        # Set random seed
        if gen_config.seed is not None:
            torch.manual_seed(gen_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(gen_config.seed)

        print(f"Generating {gen_config.width}x{gen_config.height} panorama...")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Steps: {gen_config.num_inference_steps}, Guidance: {gen_config.guidance_scale}")

        # In ComfyUI workflow, conditioning is already done by CLIPTextEncode nodes
        # This function documents the full pipeline, but ComfyUI handles most of it

        # The actual generation happens through ComfyUI's KSampler node
        # which uses the model we've already patched with OmniX adapters

        # This method serves as documentation and can be used for
        # programmatic generation outside of ComfyUI graph

        raise NotImplementedError(
            "Direct text-to-panorama generation should use ComfyUI workflow:\n"
            "1. CLIPTextEncode -> conditioning\n"
            "2. EmptyLatentImage -> latent (2048x1024)\n"
            "3. KSampler with OmniX-patched model\n"
            "4. VAEDecode -> panorama image\n"
            "\n"
            "See workflows/example_text_to_panorama.json for reference."
        )

    def generate_from_image(
        self,
        init_image: Union[torch.Tensor, Image.Image],
        prompt: str = "",
        strength: float = 0.75,
        seed: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate panorama from initial image (img2img).

        Args:
            init_image: Starting image (will be resized to panorama dimensions)
            prompt: Text guidance for generation
            strength: How much to transform the image (0.0 = keep original, 1.0 = full generation)
            seed: Random seed
            **kwargs: Additional parameters

        Returns:
            Generated panorama as ComfyUI IMAGE tensor
        """
        # Update config
        gen_config = self._update_config(seed=seed, **kwargs)

        # Convert image to tensor if needed
        if isinstance(init_image, Image.Image):
            init_image = to_comfyui_image(init_image)

        # Validate and resize image to panorama dimensions
        init_image = self._prepare_init_image(init_image, gen_config)

        print(f"Generating panorama from image (strength: {strength})...")

        # Similar to text-to-image, this should use ComfyUI's workflow
        # with img2img nodes (VAEEncode -> KSampler with denoise < 1.0)

        raise NotImplementedError(
            "Image-to-panorama generation should use ComfyUI workflow:\n"
            "1. LoadImage -> IMAGE\n"
            "2. VAEEncode -> LATENT\n"
            "3. KSampler with OmniX-patched model (denoise=strength)\n"
            "4. VAEDecode -> panorama image\n"
            "\n"
            "Ensure input image has 2:1 aspect ratio or use OmniXPanoramaValidator."
        )

    def _update_config(self, **kwargs) -> GenerationConfig:
        """Update generation config with runtime parameters"""
        config_dict = {
            'width': kwargs.get('width', self.config.width),
            'height': kwargs.get('height', self.config.height),
            'num_inference_steps': kwargs.get('num_inference_steps', self.config.num_inference_steps),
            'guidance_scale': kwargs.get('guidance_scale', self.config.guidance_scale),
            'adapter_strength': kwargs.get('adapter_strength', self.config.adapter_strength),
            'panorama_mode': kwargs.get('panorama_mode', self.config.panorama_mode),
            'batch_size': kwargs.get('batch_size', self.config.batch_size),
            'use_fp16': kwargs.get('use_fp16', self.config.use_fp16),
            'seed': kwargs.get('seed', self.config.seed),
        }

        config = GenerationConfig(**config_dict)
        config.validate()
        return config

    def _prepare_init_image(
        self,
        image: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Prepare initial image for img2img generation.

        Args:
            image: Input image tensor (B, H, W, C)
            config: Generation config with target dimensions

        Returns:
            Resized and validated image tensor
        """
        batch, height, width, channels = image.shape

        # Check if resize needed
        if width != config.width or height != config.height:
            print(f"Resizing init image: {width}x{height} -> {config.width}x{config.height}")

            # Convert to (B, C, H, W) for interpolation
            image = image.permute(0, 3, 1, 2)

            # Resize
            image = torch.nn.functional.interpolate(
                image,
                size=(config.height, config.width),
                mode='bilinear',
                align_corners=False
            )

            # Convert back to (B, H, W, C)
            image = image.permute(0, 2, 3, 1)

        return image


class PanoramaPostProcessor:
    """
    Post-processing utilities for generated panoramas.

    Handles:
    - Seam blending (left-right edge continuity)
    - Color correction
    - Sharpening
    - Artifact removal
    """

    @staticmethod
    def ensure_seamless(
        panorama: torch.Tensor,
        blend_width: int = 32
    ) -> torch.Tensor:
        """
        Ensure panorama is seamless (left edge matches right edge).

        This is crucial for 360° viewing without visible seams.

        Args:
            panorama: Panorama tensor (B, H, W, C)
            blend_width: Width of blending region in pixels

        Returns:
            Seamless panorama with blended edges
        """
        batch, height, width, channels = panorama.shape

        # Extract left and right edge regions
        left_edge = panorama[:, :, :blend_width, :]
        right_edge = panorama[:, :, -blend_width:, :]

        # Create blend mask (linear fade)
        blend_mask = torch.linspace(0, 1, blend_width, device=panorama.device)
        blend_mask = blend_mask.view(1, 1, -1, 1)

        # Blend edges
        blended_left = left_edge * (1 - blend_mask) + right_edge * blend_mask
        blended_right = right_edge * blend_mask + left_edge * (1 - blend_mask)

        # Apply blended regions
        panorama = panorama.clone()
        panorama[:, :, :blend_width, :] = blended_left
        panorama[:, :, -blend_width:, :] = blended_right

        return panorama

    @staticmethod
    def correct_spherical_distortion(
        panorama: torch.Tensor,
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        Correct for spherical distortion in equirectangular projection.

        Equirectangular projections have inherent distortion at poles.
        This applies compensation to improve visual quality.

        Args:
            panorama: Panorama tensor (B, H, W, C)
            strength: Correction strength (0.0 = none, 1.0 = full)

        Returns:
            Corrected panorama
        """
        if strength == 0.0:
            return panorama

        batch, height, width, channels = panorama.shape

        # Create latitude-based correction map
        # Equirectangular projection stretches poles horizontally
        lat = torch.linspace(-np.pi/2, np.pi/2, height, device=panorama.device)
        correction = torch.cos(lat).view(1, -1, 1, 1)

        # Apply correction with strength factor
        correction = 1.0 + strength * (correction - 1.0)

        # Apply to panorama (modulates intensity based on latitude)
        panorama = panorama * correction

        # Clamp to valid range
        panorama = torch.clamp(panorama, 0.0, 1.0)

        return panorama

    @staticmethod
    def enhance_colors(
        panorama: torch.Tensor,
        saturation: float = 1.1,
        brightness: float = 1.0,
        contrast: float = 1.0
    ) -> torch.Tensor:
        """
        Enhance colors in panorama.

        Args:
            panorama: Panorama tensor (B, H, W, C)
            saturation: Saturation multiplier (1.0 = no change)
            brightness: Brightness multiplier (1.0 = no change)
            contrast: Contrast multiplier (1.0 = no change)

        Returns:
            Enhanced panorama
        """
        # Apply brightness
        if brightness != 1.0:
            panorama = panorama * brightness

        # Apply contrast
        if contrast != 1.0:
            mean = panorama.mean(dim=(1, 2, 3), keepdim=True)
            panorama = (panorama - mean) * contrast + mean

        # Apply saturation
        if saturation != 1.0:
            # Convert to grayscale
            grayscale = panorama.mean(dim=3, keepdim=True)
            # Interpolate between grayscale and color
            panorama = grayscale + saturation * (panorama - grayscale)

        # Clamp to valid range
        panorama = torch.clamp(panorama, 0.0, 1.0)

        return panorama


class BatchPanoramaGenerator:
    """
    Batch processing for multiple panorama generations.

    Optimizes memory usage and throughput for generating
    many panoramas from different prompts.
    """

    def __init__(
        self,
        generator: OmniXPanoramaGenerator,
        max_batch_size: int = 4
    ):
        """
        Initialize batch generator.

        Args:
            generator: OmniXPanoramaGenerator instance
            max_batch_size: Maximum batch size (depends on available VRAM)
        """
        self.generator = generator
        self.max_batch_size = max_batch_size

    def generate_batch(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Generate multiple panoramas in batches.

        Args:
            prompts: List of text prompts
            negative_prompts: List of negative prompts (optional)
            seeds: List of random seeds (optional)
            **kwargs: Additional generation parameters

        Returns:
            List of generated panorama tensors
        """
        num_prompts = len(prompts)

        if negative_prompts is None:
            negative_prompts = [""] * num_prompts

        if seeds is None:
            seeds = [None] * num_prompts

        # Validate input lengths
        if len(negative_prompts) != num_prompts:
            raise ValueError("Number of negative prompts must match number of prompts")
        if len(seeds) != num_prompts:
            raise ValueError("Number of seeds must match number of prompts")

        results = []

        # Process in batches
        for i in range(0, num_prompts, self.max_batch_size):
            batch_end = min(i + self.max_batch_size, num_prompts)
            batch_prompts = prompts[i:batch_end]
            batch_negatives = negative_prompts[i:batch_end]
            batch_seeds = seeds[i:batch_end]

            print(f"Processing batch {i//self.max_batch_size + 1}/{(num_prompts + self.max_batch_size - 1)//self.max_batch_size}")

            # Generate each in batch
            for prompt, negative, seed in zip(batch_prompts, batch_negatives, batch_seeds):
                panorama = self.generator.generate_from_text(
                    prompt=prompt,
                    negative_prompt=negative,
                    seed=seed,
                    **kwargs
                )
                results.append(panorama)

        return results


def create_empty_panorama_latent(
    width: int = 2048,
    height: int = 1024,
    batch_size: int = 1,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Create empty latent for panorama generation.

    This is a helper function that mimics ComfyUI's EmptyLatentImage node
    for programmatic use.

    Args:
        width: Latent width (pixel width // 8 for Flux VAE)
        height: Latent height (pixel height // 8)
        batch_size: Number of latents in batch
        device: Target device

    Returns:
        Dictionary with 'samples' key containing latent tensor
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Flux VAE uses 8x downsampling
    latent_height = height // 8
    latent_width = width // 8

    # Flux latents have 16 channels
    latent_channels = 16

    # Create random latent noise
    latent = torch.randn(
        batch_size,
        latent_channels,
        latent_height,
        latent_width,
        device=device
    )

    return {"samples": latent}
