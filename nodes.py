"""
ComfyUI-OmniX Node Definitions

This module defines all custom nodes for OmniX integration with ComfyUI.
Works with ComfyUI's existing Flux pipeline infrastructure.
"""

import torch
import folder_paths
from typing import Tuple, Dict, Any, Optional
import os

from .omnix.adapters import AdapterManager, OmniXAdapters
from .omnix.perceiver import OmniXPerceiver
from .omnix.model_loader import OmniXModelLoader as ModelLoader, load_flux_model
from .omnix.generator import OmniXPanoramaGenerator
from .omnix.utils import (
    to_comfyui_image,
    from_comfyui_image,
    validate_panorama_aspect_ratio,
    visualize_depth_map,
    normalize_normal_map
)


class OmniXAdapterLoader:
    """
    Loads OmniX adapter weights for panorama generation and perception.
    Works with ComfyUI's existing Flux model loaders.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter_preset": (["omnix-base", "omnix-large"], {"default": "omnix-base"}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("OMNIX_ADAPTERS",)
    RETURN_NAMES = ("adapters",)
    FUNCTION = "load_adapters"
    CATEGORY = "OmniX"
    DESCRIPTION = "Load OmniX adapter weights for panorama generation and perception"

    def load_adapters(self, adapter_preset: str, precision: str) -> Tuple[OmniXAdapters]:
        """Load OmniX adapters with specified precision"""
        try:
            # Determine dtype
            dtype_map = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16
            }
            dtype = dtype_map.get(precision, torch.float16)

            # Find adapter path
            adapter_base_path = folder_paths.get_folder_paths("omnix")
            if not adapter_base_path:
                # Fallback to custom_nodes directory
                adapter_base_path = [os.path.join(
                    os.path.dirname(__file__),
                    "models",
                    "omnix"
                )]

            adapter_path = os.path.join(adapter_base_path[0], adapter_preset)

            # Load adapters
            adapter_manager = AdapterManager(adapter_path, dtype=dtype)
            adapters = OmniXAdapters(adapter_manager)

            print(f"✓ Loaded OmniX adapters: {adapter_preset} ({precision})")
            print(f"  Path: {adapter_path}")

            return (adapters,)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load OmniX adapters '{adapter_preset}': {str(e)}\n"
                f"Please ensure adapter weights are installed in ComfyUI/models/omnix/{adapter_preset}/"
            )


class OmniXApplyAdapters:
    """
    Applies OmniX adapters to a Flux model for panorama generation.
    Insert this between model loading and sampling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "adapters": ("OMNIX_ADAPTERS",),
                "adapter_type": (["rgb_generation"], {"default": "rgb_generation"}),
                "adapter_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_adapters"
    CATEGORY = "OmniX"
    DESCRIPTION = "Apply OmniX adapters to Flux model for panorama generation"

    def apply_adapters(
        self,
        model: Any,
        adapters: OmniXAdapters,
        adapter_type: str,
        adapter_strength: float
    ) -> Tuple[Any]:
        """Apply adapters to model"""
        try:
            # Clone model to avoid modifying original
            model_patched = model.clone()

            # Get adapter weights
            adapter_weights = adapters.get_adapter(adapter_type)

            # Apply adapter to model (this patches the model's forward pass)
            adapters.inject_into_model(
                model_patched,
                adapter_type,
                strength=adapter_strength
            )

            print(f"✓ Applied OmniX {adapter_type} adapter (strength: {adapter_strength:.2f})")

            return (model_patched,)

        except Exception as e:
            raise RuntimeError(
                f"Failed to apply OmniX adapters: {str(e)}\n"
                f"Ensure the model is a valid Flux model."
            )


class OmniXPanoramaPerception:
    """
    Extract geometric and material properties from panoramas.
    Outputs: depth, normals, albedo, roughness, metallic maps.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapters": ("OMNIX_ADAPTERS",),
                "panorama": ("IMAGE",),
            },
            "optional": {
                "extract_distance": ("BOOLEAN", {"default": True}),
                "extract_normal": ("BOOLEAN", {"default": True}),
                "extract_albedo": ("BOOLEAN", {"default": True}),
                "extract_roughness": ("BOOLEAN", {"default": True}),
                "extract_metallic": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("distance", "normal", "albedo", "roughness", "metallic")
    FUNCTION = "extract_properties"
    CATEGORY = "OmniX"
    DESCRIPTION = "Extract depth, normals, and PBR material properties from panoramas"

    def extract_properties(
        self,
        adapters: OmniXAdapters,
        panorama: torch.Tensor,
        extract_distance: bool = True,
        extract_normal: bool = True,
        extract_albedo: bool = True,
        extract_roughness: bool = True,
        extract_metallic: bool = True,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Extract properties from panorama using OmniX adapters"""

        try:
            # Validate input
            validate_panorama_aspect_ratio(panorama)

            # Create perceiver
            perceiver = OmniXPerceiver(adapters)

            # Build list of properties to extract
            extract_modes = []
            if extract_distance:
                extract_modes.append("distance")
            if extract_normal:
                extract_modes.append("normal")
            if extract_albedo:
                extract_modes.append("albedo")
            if extract_roughness:
                extract_modes.append("roughness")
            if extract_metallic:
                extract_modes.append("metallic")

            if not extract_modes:
                raise ValueError("At least one extraction mode must be enabled")

            # Run perception
            print(f"Running OmniX perception: {', '.join(extract_modes)}")
            results = perceiver.perceive(panorama, extract_modes)

            # Create placeholder for disabled outputs (black image with same dimensions)
            batch_size, height, width, channels = panorama.shape
            device = panorama.device
            dtype = panorama.dtype

            def create_placeholder(num_channels=1):
                """Create black placeholder tensor for disabled outputs"""
                return torch.zeros((batch_size, height, width, num_channels), device=device, dtype=dtype)

            # Format outputs (black placeholder if not extracted)
            distance = results.get("distance")
            if distance is not None:
                distance = visualize_depth_map(distance)
            else:
                distance = create_placeholder(1)  # Single channel for distance

            normal = results.get("normal")
            if normal is not None:
                normal = normalize_normal_map(normal)
            else:
                normal = create_placeholder(3)  # Three channels for normal (RGB)

            albedo = results.get("albedo")
            if albedo is None:
                albedo = create_placeholder(3)  # Three channels for albedo (RGB)

            roughness = results.get("roughness")
            if roughness is None:
                roughness = create_placeholder(1)  # Single channel for roughness

            metallic = results.get("metallic")
            if metallic is None:
                metallic = create_placeholder(1)  # Single channel for metallic

            print(f"✓ Extracted {len(results)} property maps from panorama")

            return (distance, normal, albedo, roughness, metallic)

        except Exception as e:
            raise RuntimeError(
                f"Failed to extract properties from panorama: {str(e)}"
            )


class OmniXPanoramaValidator:
    """
    Validates and fixes panorama aspect ratios for OmniX processing.
    Ensures panoramas are in equirectangular format (2:1 aspect ratio).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_aspect_ratio": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.8,
                    "max": 2.2,
                    "step": 0.1
                }),
                "fix_method": (["crop", "pad", "stretch"], {"default": "crop"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "validate"
    CATEGORY = "OmniX/utils"
    DESCRIPTION = "Validate and fix panorama aspect ratios for OmniX"

    def validate(
        self,
        image: torch.Tensor,
        target_aspect_ratio: float,
        fix_method: str
    ) -> Tuple[torch.Tensor, str]:
        """Validate and optionally fix panorama aspect ratio"""

        batch, height, width, channels = image.shape
        current_ratio = width / height

        info = f"Input: {width}x{height} (ratio: {current_ratio:.2f})"

        # Check if aspect ratio is acceptable
        ratio_diff = abs(current_ratio - target_aspect_ratio)

        if ratio_diff < 0.1:  # Within 5% tolerance
            info += " ✓ Valid panorama aspect ratio"
            return (image, info)

        # Fix aspect ratio
        if fix_method == "crop":
            # Crop to target ratio
            if current_ratio > target_aspect_ratio:
                # Too wide, crop width
                new_width = int(height * target_aspect_ratio)
                start_x = (width - new_width) // 2
                image = image[:, :, start_x:start_x + new_width, :]
            else:
                # Too tall, crop height
                new_height = int(width / target_aspect_ratio)
                start_y = (height - new_height) // 2
                image = image[:, start_y:start_y + new_height, :, :]

            info += f" → Cropped to {image.shape[2]}x{image.shape[1]}"

        elif fix_method == "pad":
            # Pad to target ratio
            if current_ratio > target_aspect_ratio:
                # Too wide, pad height
                new_height = int(width / target_aspect_ratio)
                pad_total = new_height - height
                pad_top = pad_total // 2
                pad_bottom = pad_total - pad_top
                image = torch.nn.functional.pad(
                    image.permute(0, 3, 1, 2),
                    (0, 0, pad_top, pad_bottom),
                    mode='replicate'
                ).permute(0, 2, 3, 1)
            else:
                # Too tall, pad width
                new_width = int(height * target_aspect_ratio)
                pad_total = new_width - width
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                image = torch.nn.functional.pad(
                    image.permute(0, 3, 1, 2),
                    (pad_left, pad_right, 0, 0),
                    mode='replicate'
                ).permute(0, 2, 3, 1)

            info += f" → Padded to {image.shape[2]}x{image.shape[1]}"

        elif fix_method == "stretch":
            # Stretch to target ratio
            import torch.nn.functional as F
            new_width = width if current_ratio > target_aspect_ratio else int(height * target_aspect_ratio)
            new_height = height if current_ratio < target_aspect_ratio else int(width / target_aspect_ratio)

            image = F.interpolate(
                image.permute(0, 3, 1, 2),
                size=(new_height, new_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)

            info += f" → Stretched to {new_width}x{new_height}"

        return (image, info)


class OmniXModelLoaderNode:
    """
    Loads Flux.1-dev model with OmniX adapters for panorama generation.
    This is the main model loader that combines Flux + OmniX into a unified model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "flux_checkpoint": (folder_paths.get_filename_list("checkpoints"),),
                "adapter_preset": (["omnix-base", "omnix-large"], {"default": "omnix-base"}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("OMNIX_MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("omnix_model", "clip", "vae")
    FUNCTION = "load_model"
    CATEGORY = "OmniX"
    DESCRIPTION = "Load Flux.1-dev with OmniX adapters for panorama generation"

    def load_model(self, flux_checkpoint: str, adapter_preset: str, precision: str):
        """Load Flux model with OmniX adapters"""
        try:
            # Determine dtype
            dtype_map = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16
            }
            dtype = dtype_map.get(precision, torch.float16)

            # Load Flux model
            print(f"Loading Flux checkpoint: {flux_checkpoint}")
            model, clip, vae = load_flux_model(flux_checkpoint)

            # Find adapter path
            adapter_base_path = folder_paths.get_folder_paths("omnix")
            if not adapter_base_path:
                # Fallback to custom_nodes directory
                adapter_base_path = [os.path.join(
                    os.path.dirname(__file__),
                    "models",
                    "omnix"
                )]

            adapter_path = os.path.join(adapter_base_path[0], adapter_preset)

            # Create OmniX model loader
            omnix_model = ModelLoader.from_comfyui(
                model=model,
                adapter_path=adapter_path,
                vae=vae,
                dtype=dtype
            )

            # Store CLIP for later use
            omnix_model.clip = clip

            print(f"✓ Loaded OmniX model: {flux_checkpoint} + {adapter_preset}")

            return (omnix_model, clip, vae)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load OmniX model: {str(e)}\n"
                f"Ensure Flux checkpoint and OmniX adapters are installed correctly."
            )


class OmniXPanoramaGeneratorNode:
    """
    Generates 360° panoramas from text prompts using OmniX.
    Supports both text-to-panorama and image-to-panorama workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "omnix_model": ("OMNIX_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "width": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 8192,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64
                }),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "conditioning_image": ("IMAGE",),
                "conditioning_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("panorama",)
    FUNCTION = "generate"
    CATEGORY = "OmniX"
    DESCRIPTION = "Generate 360° panoramas from text prompts using OmniX"

    def generate(
        self,
        omnix_model: ModelLoader,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        negative_prompt: str = "",
        conditioning_image: Optional[torch.Tensor] = None,
        conditioning_strength: float = 0.8,
    ) -> Tuple[torch.Tensor]:
        """Generate panorama from text prompt"""

        try:
            # Create generator
            generator = OmniXPanoramaGenerator(omnix_model)

            # Generate panorama
            panorama = generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                width=width,
                height=height,
                denoise=denoise,
                conditioning_image=conditioning_image,
                conditioning_strength=conditioning_strength,
            )

            return (panorama,)

        except Exception as e:
            raise RuntimeError(
                f"Failed to generate panorama: {str(e)}\n"
                f"Check that the model is loaded correctly and parameters are valid."
            )


# Node registration
NODE_CLASS_MAPPINGS = {
    "OmniXModelLoader": OmniXModelLoaderNode,
    "OmniXPanoramaGenerator": OmniXPanoramaGeneratorNode,
    "OmniXAdapterLoader": OmniXAdapterLoader,
    "OmniXApplyAdapters": OmniXApplyAdapters,
    "OmniXPanoramaPerception": OmniXPanoramaPerception,
    "OmniXPanoramaValidator": OmniXPanoramaValidator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniXModelLoader": "OmniX Model Loader",
    "OmniXPanoramaGenerator": "OmniX Panorama Generator",
    "OmniXAdapterLoader": "OmniX Adapter Loader",
    "OmniXApplyAdapters": "OmniX Apply Adapters",
    "OmniXPanoramaPerception": "OmniX Panorama Perception",
    "OmniXPanoramaValidator": "OmniX Panorama Validator",
}
