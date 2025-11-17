"""
Diffusers-based Flux nodes for OmniX perception.

Provides an alternative execution path that uses HuggingFace Diffusers'
FluxPipeline to run OmniX perception adapters without relying on ComfyUI's
native Flux implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import folder_paths
import comfy.model_management as model_management
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from safetensors.torch import load_file

from .omnix.diffusers_key_remap import remap_omnix_to_diffusers_flux, verify_key_format

logger = logging.getLogger(__name__)

REQUIRED_DIFFUSERS_FILES = (
    ("", "model_index.json"),
    ("scheduler", "scheduler_config.json"),
    ("transformer", "config.json"),
    ("text_encoder", "config.json"),
    ("text_encoder_2", "config.json"),
    ("tokenizer", "merges.txt"),
    ("tokenizer", "vocab.json"),
    ("tokenizer_2", "spiece.model"),
    ("vae", "config.json"),
)


class FluxDiffusersLoader:
    """
    Load Flux.1 models through Diffusers, optionally from local checkpoints.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import os

        checkpoint_files: List[str] = []
        diffusers_path = os.path.join(folder_paths.models_dir, "diffusers")
        unet_path = os.path.join(folder_paths.models_dir, "unet")
        checkpoints = folder_paths.get_folder_paths("checkpoints")
        checkpoint_dir = checkpoints[0] if checkpoints else None

        if os.path.exists(diffusers_path):
            checkpoint_files.extend(
                f for f in os.listdir(diffusers_path) if f.endswith((".safetensors", ".sft"))
            )
        if os.path.exists(unet_path):
            checkpoint_files.extend(
                f for f in os.listdir(unet_path) if f.endswith((".safetensors", ".sft"))
            )
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            checkpoint_files.extend(
                f
                for f in os.listdir(checkpoint_dir)
                if "flux" in f.lower() and f.endswith(".safetensors")
            )

        checkpoint_files = sorted(list(set(checkpoint_files)))
        default_checkpoint = (
            "flux1-dev.safetensors"
            if "flux1-dev.safetensors" in checkpoint_files
            else (checkpoint_files[0] if checkpoint_files else "flux1-dev.safetensors")
        )
        checkpoint_files = checkpoint_files or [default_checkpoint]

        return {
            "required": {
                "torch_dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            },
            "optional": {
                "local_checkpoint": (checkpoint_files, {"default": default_checkpoint}),
            },
        }

    RETURN_TYPES = ("FLUX_PIPELINE", "VAE", "TEXT_ENCODER")
    RETURN_NAMES = ("flux_pipeline", "vae", "text_encoder")
    FUNCTION = "load_pipeline"
    CATEGORY = "OmniX/Diffusers"

    @staticmethod
    def _find_checkpoint(filename: str) -> Path:
        search_paths = [
            Path(folder_paths.models_dir) / "diffusers" / filename,
            Path(folder_paths.models_dir) / "unet" / filename,
        ]
        checkpoint_dirs = folder_paths.get_folder_paths("checkpoints")
        if checkpoint_dirs:
            search_paths.append(Path(checkpoint_dirs[0]) / filename)

        for candidate in search_paths:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Local checkpoint not found: {filename}\nChecked:\n"
            + "\n".join(f"- {p}" for p in search_paths)
        )

    @staticmethod
    def _load_single_file(path: Path, torch_dtype: torch.dtype) -> FluxPipeline:
        """
        Load Flux pipeline from single file with fallback for weights_only restriction.

        First attempts standard loading. If that fails due to weights_only restriction,
        temporarily overrides torch.load in a controlled manner.
        """
        config_dir = path.parent
        missing = []
        for subdir, filename in REQUIRED_DIFFUSERS_FILES:
            target = config_dir / subdir / filename if subdir else config_dir / filename
            if not target.exists():
                missing.append(target)

        if not missing:
            logger.info("Loading Flux Diffusers repo from %s", config_dir)
            return FluxPipeline.from_pretrained(
                str(config_dir),
                torch_dtype=torch_dtype,
                local_files_only=True,
            )

        if missing and config_dir.name == "diffusers":
            missing_desc = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "FluxDiffusersLoader requires the Diffusers repo files. "
                f"Missing entries under {config_dir}:\n{missing_desc}\n"
                "Copy the full `black-forest-labs/FLUX.1-dev` repository into this folder."
            )

        common_kwargs = {
            "torch_dtype": torch_dtype,
            "local_files_only": True,
        }
        try:
            return FluxPipeline.from_single_file(path, **common_kwargs)
        except Exception as err:
            def _matches_weights_only(exc: Exception) -> bool:
                seen = set()
                stack = [exc]
                while stack:
                    current = stack.pop()
                    if current is None:
                        continue
                    if id(current) in seen:
                        continue
                    seen.add(id(current))
                    message = str(current)
                    lowered = message.lower()
                    if (
                        "weights_only" in lowered
                        or "weights only" in lowered
                        or "unpicklingerror" in lowered
                    ):
                        return True
                    stack.append(getattr(current, "__cause__", None))
                    stack.append(getattr(current, "__context__", None))
                return False

            if not _matches_weights_only(err):
                raise

            logging.warning(
                "Diffusers requires weights_only=False for %s. Using safe fallback loading.",
                path.name,
            )

            # Use a safer approach: patch with functools.partial in local scope
            import functools

            # Create a wrapper that sets weights_only=False
            safe_torch_load = functools.partial(torch.load, weights_only=False)

            # Temporarily store original for restoration
            _original_torch_load = torch.load

            try:
                # Replace torch.load with our safe version
                torch.load = safe_torch_load
                result = FluxPipeline.from_single_file(path, **common_kwargs)
                return result
            finally:
                # Always restore original torch.load, even on exception
                torch.load = _original_torch_load

    def load_pipeline(
        self,
        torch_dtype: str,
        local_checkpoint: str = "flux1-dev.safetensors",
    ):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map[torch_dtype]
        device = model_management.get_torch_device()

        checkpoint = self._find_checkpoint(local_checkpoint)
        logger.info("Loading Flux from %s", checkpoint)

        logger.info("Using Diffusers config from %s", checkpoint.parent)
        pipeline = self._load_single_file(checkpoint, dtype)

        pipeline = pipeline.to(device)
        pipeline.vae.enable_tiling()
        logger.info(f"Pipeline ready on {device}")
        return (pipeline, pipeline.vae, pipeline.text_encoder)


class OmniXLoRALoader:
    """
    Load OmniX LoRA adapters into a FluxPipeline instance.
    """

    ADAPTER_FILENAMES = {
        "distance": "rgb_to_depth_depth.safetensors",
        "normal": "rgb_to_normal_normal.safetensors",
        "albedo": "rgb_to_albedo_albedo.safetensors",
        "pbr": "rgb_to_pbr_pbr.safetensors",
    }

    @classmethod
    def INPUT_TYPES(cls):
        import os

        # Auto-detect adapter directory
        default_adapter_dir = "models/loras/omnix"

        # Try to find ComfyUI loras directory
        loras_paths = folder_paths.get_folder_paths("loras")
        if loras_paths and len(loras_paths) > 0:
            # Use first loras path + omnix subdirectory
            default_adapter_dir = os.path.join(loras_paths[0], "omnix")

        return {
            "required": {
                "flux_pipeline": ("FLUX_PIPELINE",),
                "adapter_dir": ("STRING", {"default": default_adapter_dir}),
                "enable_distance": ("BOOLEAN", {"default": True}),
                "enable_normal": ("BOOLEAN", {"default": True}),
                "enable_albedo": ("BOOLEAN", {"default": True}),
                "enable_pbr": ("BOOLEAN", {"default": False}),
                "lora_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("FLUX_PIPELINE", "OMNIX_ADAPTERS")
    RETURN_NAMES = ("flux_pipeline", "loaded_adapters")
    FUNCTION = "load_adapters"
    CATEGORY = "OmniX/Diffusers"

    def load_adapters(
        self,
        flux_pipeline,
        adapter_dir,
        enable_distance,
        enable_normal,
        enable_albedo,
        enable_pbr,
        lora_scale,
    ):
        adapter_dir = Path(adapter_dir)
        requested: List[str] = []
        if enable_distance:
            requested.append("distance")
        if enable_normal:
            requested.append("normal")
        if enable_albedo:
            requested.append("albedo")
        if enable_pbr:
            requested.append("pbr")

        if not requested:
            logger.warning("No adapters selected")
            return flux_pipeline, {}

        # Clear previously loaded adapters so we can re-use adapter names safely
        # Try multiple methods to ensure adapters are properly cleared
        adapters_cleared = False

        # Method 1: Try to get list of existing adapters and delete them individually
        if hasattr(flux_pipeline, "get_list_adapters"):
            try:
                existing = flux_pipeline.get_list_adapters()
                if existing:
                    logger.info(f"Found existing adapters: {existing}")
                    for adapter_info in existing:
                        adapter_name = adapter_info if isinstance(adapter_info, str) else adapter_info.get("name", "")
                        if adapter_name:
                            try:
                                flux_pipeline.delete_adapters(adapter_name)
                                logger.debug(f"Deleted existing adapter: {adapter_name}")
                            except Exception as e:
                                logger.debug(f"Could not delete adapter {adapter_name}: {e}")
                    adapters_cleared = True
            except Exception as exc:
                logger.debug(f"get_list_adapters failed: {exc}")

        # Method 2: Try to delete all adapters
        if not adapters_cleared and hasattr(flux_pipeline, "delete_adapters"):
            try:
                flux_pipeline.delete_adapters()
                logger.info("Cleared all existing adapters")
                adapters_cleared = True
            except Exception as exc:
                logger.debug(f"delete_adapters failed: {exc}")

        # Method 3: Try to unload LoRA weights
        if not adapters_cleared and hasattr(flux_pipeline, "unload_lora_weights"):
            try:
                flux_pipeline.unload_lora_weights()
                logger.info("Unloaded existing LoRA weights")
                adapters_cleared = True
            except Exception as exc:
                logger.debug(f"unload_lora_weights failed: {exc}")

        # Method 4: Try to delete individual adapters we're about to load
        if not adapters_cleared:
            for adapter_name in requested:
                if hasattr(flux_pipeline, "delete_adapters"):
                    try:
                        flux_pipeline.delete_adapters(adapter_name)
                        logger.debug(f"Deleted adapter: {adapter_name}")
                    except Exception as exc:
                        logger.debug(f"Could not delete adapter {adapter_name}: {exc}")

        loaded: Dict[str, Dict[str, torch.Tensor]] = {}
        adapter_names: List[str] = []
        adapter_scales: List[float] = []

        for adapter_name in requested:
            filename = self.ADAPTER_FILENAMES[adapter_name]
            adapter_path = adapter_dir / filename
            if not adapter_path.exists():
                logger.warning(f"Missing adapter: {adapter_path}")
                continue

            logger.info(f"Loading {adapter_name} from {adapter_path}")
            state = load_file(str(adapter_path))

            # CRITICAL: Check if keys need remapping for Diffusers compatibility
            key_patterns = verify_key_format(state)
            needs_remap = (
                key_patterns.get("transformer_blocks", 0) > 0 or
                key_patterns.get("single_transformer_blocks", 0) > 0
            )

            if needs_remap:
                logger.warning(f"Adapter '{adapter_name}' uses OmniX key format - remapping to Diffusers format")
                logger.debug(f"  Original format: {key_patterns}")
                state = remap_omnix_to_diffusers_flux(state)
                key_patterns_new = verify_key_format(state)
                logger.info(f"  Remapped format: {key_patterns_new}")
            else:
                logger.info(f"Adapter '{adapter_name}' already in Diffusers format")

            # Debug: Log adapter weight structure
            logger.debug(f"Adapter '{adapter_name}' has {len(state)} weight tensors")
            sample_keys = list(state.keys())[:5]
            logger.debug(f"Sample weight keys: {sample_keys}")
            if sample_keys:
                first_key = sample_keys[0]
                logger.debug(f"First weight shape: {first_key} -> {state[first_key].shape}")

            # Save remapped state to temporary file for load_lora_weights
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp_file:
                temp_adapter_path = tmp_file.name
                from safetensors.torch import save_file
                save_file(state, temp_adapter_path)

            try:
                # Load from temporary remapped file
                flux_pipeline.load_lora_weights(
                    Path(temp_adapter_path).parent,
                    weight_name=Path(temp_adapter_path).name,
                    adapter_name=adapter_name,
                )
                logger.info(f"Loaded adapter '{adapter_name}' successfully")
            except ValueError as e:
                if "already in use" in str(e):
                    logger.warning(f"Adapter {adapter_name} already exists, attempting to delete and retry...")
                    try:
                        # Try to delete the specific adapter
                        if hasattr(flux_pipeline, "delete_adapters"):
                            flux_pipeline.delete_adapters(adapter_name)
                            logger.info(f"Deleted existing adapter: {adapter_name}")

                        # Retry loading from temp file
                        flux_pipeline.load_lora_weights(
                            Path(temp_adapter_path).parent,
                            weight_name=Path(temp_adapter_path).name,
                            adapter_name=adapter_name,
                        )
                        logger.info(f"Successfully loaded {adapter_name} after retry")
                    except Exception as retry_error:
                        logger.error(f"Failed to load {adapter_name} even after deleting: {retry_error}")
                        logger.error(f"Please restart ComfyUI to clear adapter cache")
                        raise
                else:
                    raise
            finally:
                # Clean up temporary file
                import os
                try:
                    os.unlink(temp_adapter_path)
                except Exception:
                    pass  # Ignore cleanup errors

            loaded[adapter_name] = {
                "path": str(adapter_path),
                "weights": len(state),
                "scale": lora_scale,
            }
            adapter_names.append(adapter_name)
            adapter_scales.append(lora_scale)

        if adapter_names:
            flux_pipeline.set_adapters(adapter_names, adapter_weights=adapter_scales)
            logger.info(f"Active adapters: {adapter_names}")

            # CRITICAL: Verify LoRA layers were actually injected into the model
            lora_layers_found = []
            if hasattr(flux_pipeline.transformer, 'peft_config'):
                logger.info(f"PEFT config loaded: {list(flux_pipeline.transformer.peft_config.keys())}")

                # Check if actual LoRA modules exist in the transformer
                for name, module in flux_pipeline.transformer.named_modules():
                    if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                        lora_layers_found.append(name)

                if lora_layers_found:
                    logger.info(f"✓ Found {len(lora_layers_found)} LoRA-injected layers")
                    logger.debug(f"  Sample layers: {lora_layers_found[:5]}")
                else:
                    logger.error("✗ CRITICAL: No LoRA layers found in transformer!")
                    logger.error("  Adapters loaded but NOT applied - likely key mismatch")
                    logger.error("  Outputs will be incorrect. See CRITICAL_ADAPTER_FAILURE_ANALYSIS.md")
            else:
                logger.error("✗ CRITICAL: PEFT not initialized on transformer!")
                logger.error("  Adapters will NOT be applied during inference")

        return flux_pipeline, loaded


class OmniXPerceptionDiffusers:
    """
    Run OmniX perception tasks using the Diffusers Flux pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_pipeline": ("FLUX_PIPELINE",),
                "loaded_adapters": ("OMNIX_ADAPTERS",),
                "panorama": ("IMAGE",),
                "task": (
                    ["distance", "normal", "albedo", "pbr"],
                    {
                        "default": "distance",
                        "tooltip": "Which perception head to run: distance (depth), normal, albedo, or pbr (roughness/metallic).",
                    },
                ),
                "num_steps": (
                    "INT",
                    {
                        "default": 28,
                        "min": 1,
                        "max": 80,
                        "step": 1,
                        "tooltip": "Diffusion steps. 20-32 is typical for perception; higher values are slower but reduce noise.",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 3.5,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "Classifier-free guidance. 2.5-4.0 keeps perception stable; >8 behaves more like text-to-image.",
                    },
                ),
                "noise_strength": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.05,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Img2img strength: 0.1-0.3 preserves the panorama; 0.5+ introduces hallucinated structure.",
                    },
                ),
                "prompt_mode": (
                    ["empty", "task_name", "minimal", "descriptive"],
                    {
                        "default": "task_name",
                        "tooltip": "Prompt format: empty='', task_name='distance', minimal='perception: distance', descriptive=full description. Try 'empty' or 'task_name' if quality is poor.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("perception_output",)
    FUNCTION = "run_perception"
    CATEGORY = "OmniX/Diffusers"

    @staticmethod
    def _image_to_tensor(panorama: torch.Tensor, flux_pipeline: FluxPipeline) -> torch.Tensor:
        img = panorama[0].permute(2, 0, 1).unsqueeze(0)
        img = img * 2.0 - 1.0
        return img.to(device=flux_pipeline.device, dtype=flux_pipeline.vae.dtype)

    def _encode_latents(
        self, flux_pipeline: FluxPipeline, panorama: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        image_tensor = self._image_to_tensor(panorama, flux_pipeline)
        with torch.no_grad():
            latents = flux_pipeline.vae.encode(image_tensor).latent_dist.sample()
        vae_config = flux_pipeline.vae.config
        shift = getattr(vae_config, "shift_factor", 0.0)
        scale = getattr(vae_config, "scaling_factor", 1.0)
        latents = (latents - shift) * scale

        vae_scale = getattr(flux_pipeline, "vae_scale_factor", 8)
        image_height = panorama.shape[1]
        image_width = panorama.shape[2]
        latent_height = max(2, 2 * (int(image_height) // (vae_scale * 2)))
        latent_width = max(2, 2 * (int(image_width) // (vae_scale * 2)))
        image_height = latent_height * vae_scale
        image_width = latent_width * vae_scale

        image_latents = latents.to(device=flux_pipeline.device, dtype=flux_pipeline.dtype)
        latent_image_ids = flux_pipeline._prepare_latent_image_ids(
            1, latent_height // 2, latent_width // 2, flux_pipeline.device, flux_pipeline.dtype
        )
        return image_latents, latent_image_ids, latent_height, latent_width, image_height, image_width

    def _prepare_schedule(
        self,
        flux_pipeline: FluxPipeline,
        image_latents: torch.Tensor,
        latent_height: int,
        latent_width: int,
        num_steps: int,
        noise_strength: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scheduler = flux_pipeline.scheduler
        device = flux_pipeline.device
        num_channels_latents = flux_pipeline.transformer.config.in_channels // 4
        image_seq_len = (latent_height // 2) * (latent_width // 2)

        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        scheduler.set_timesteps(num_steps, device=device, mu=mu)
        timesteps = scheduler.timesteps
        order = getattr(scheduler, "order", 1)

        strength = float(np.clip(noise_strength, 0.05, 0.99))
        init_timestep = min(num_steps * strength, num_steps)
        t_start = int(max(num_steps - init_timestep, 0))
        start_index = min(t_start * order, len(timesteps) - 1)
        trimmed_timesteps = timesteps[start_index:]
        if trimmed_timesteps.numel() == 0:
            trimmed_timesteps = timesteps[-1:]
            start_index = len(timesteps) - 1

        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(start_index)

        noise = randn_tensor(image_latents.shape, device=device, dtype=image_latents.dtype)
        timestep_tensor = trimmed_timesteps[:1].to(device=device)

        if hasattr(scheduler, "scale_noise"):
            latents = scheduler.scale_noise(
                image_latents,
                timestep_tensor,
                noise=noise,
            )
        elif hasattr(scheduler, "sigmas"):
            sigma = scheduler.sigmas[start_index]
            if not torch.is_tensor(sigma):
                sigma = torch.tensor(sigma, device=device, dtype=image_latents.dtype)
            while len(sigma.shape) < len(image_latents.shape):
                sigma = sigma.unsqueeze(-1)
            latents = sigma * noise + (1.0 - sigma) * image_latents
        else:
            latents = image_latents + noise * noise_strength

        packed_latents = flux_pipeline._pack_latents(
            latents.to(device=flux_pipeline.device, dtype=flux_pipeline.dtype),
            batch_size=image_latents.shape[0],
            num_channels_latents=num_channels_latents,
            height=latent_height,
            width=latent_width,
        )

        return packed_latents, trimmed_timesteps

    def run_perception(
        self,
        flux_pipeline,
        loaded_adapters,
        panorama,
        task,
        num_steps,
        guidance_scale,
        noise_strength,
        prompt_mode,
    ):
        # Validate adapter availability
        if task not in loaded_adapters:
            raise ValueError(f"Adapter '{task}' not loaded. Available: {list(loaded_adapters)}")

        # Validate panorama input
        if panorama.dim() != 4 or panorama.shape[0] < 1:
            raise ValueError(f"Invalid panorama shape: {panorama.shape}. Expected (B, H, W, C)")

        height, width = panorama.shape[1], panorama.shape[2]

        # Validate panorama aspect ratio (should be 2:1 for equirectangular)
        aspect_ratio = width / height
        if not (1.8 <= aspect_ratio <= 2.2):
            logger.warning(
                f"Aspect ratio {aspect_ratio:.2f} is not standard for panoramas (2:1). "
                f"Got {width}×{height}. Output quality may be degraded."
            )

        # Validate tensor range
        pmin, pmax = panorama.min().item(), panorama.max().item()
        if not (-0.1 <= pmin and pmax <= 1.1):
            raise ValueError(f"Invalid input range: [{pmin:.2f}, {pmax:.2f}]. Expected [0, 1]")

        # Validate parameters
        if not (0.05 <= noise_strength <= 1.0):
            raise ValueError(f"Invalid noise_strength: {noise_strength}. Expected [0.05, 1.0]")

        if not (1.0 <= guidance_scale <= 20.0):
            raise ValueError(f"Invalid guidance_scale: {guidance_scale}. Expected [1.0, 20.0]")

        flux_pipeline.set_adapters([task], adapter_weights=[loaded_adapters[task]["scale"]])
        logger.info(f"Running perception - Task={task}, steps={num_steps}, noise={noise_strength}, lora_scale={loaded_adapters[task]['scale']}")

        # Debug: Verify adapter is active
        if hasattr(flux_pipeline, 'get_active_adapters'):
            active = flux_pipeline.get_active_adapters()
            logger.debug(f"Active adapters: {active}")
        if hasattr(flux_pipeline.transformer, '_hf_peft_config_loaded'):
            logger.debug(f"PEFT config loaded: {flux_pipeline.transformer._hf_peft_config_loaded}")
        if hasattr(flux_pipeline.transformer, 'peft_config'):
            logger.debug(f"PEFT adapters: {list(flux_pipeline.transformer.peft_config.keys())}")

        (
            image_latents,
            latent_image_ids,
            latent_height,
            latent_width,
            image_height,
            image_width,
        ) = self._encode_latents(flux_pipeline, panorama)
        latents, timesteps = self._prepare_schedule(
            flux_pipeline,
            image_latents,
            latent_height,
            latent_width,
            num_steps=num_steps,
            noise_strength=noise_strength,
        )

        # Generate prompt based on selected mode
        if prompt_mode == "empty":
            prompt = ""
        elif prompt_mode == "task_name":
            prompt = task  # Just "distance", "normal", "albedo", or "pbr"
        elif prompt_mode == "minimal":
            prompt = f"perception: {task}"
        else:  # descriptive
            prompt = f"perception task: {task}, equirectangular projection, 360 degree panoramic view"

        logger.debug(f"Using prompt (mode={prompt_mode}): '{prompt}'")
        prompt_embeds, pooled_prompt_embeds, text_ids = flux_pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=flux_pipeline.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=None,
        )

        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        negative_text_ids = None
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = (
                flux_pipeline.encode_prompt(
                    prompt="",
                    prompt_2=None,
                    device=flux_pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=512,
                    lora_scale=None,
                )
            )

        scheduler = flux_pipeline.scheduler
        transformer = flux_pipeline.transformer
        joint_kwargs = getattr(flux_pipeline, "joint_attention_kwargs", None) or {}

        if transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=flux_pipeline.device, dtype=torch.float32
            ).expand(latents.shape[0])
        else:
            guidance = None

        with flux_pipeline.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                with transformer.cache_context("cond"):
                    noise_pred = transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=joint_kwargs,
                        return_dict=False,
                    )[0]

                if do_cfg:
                    with transformer.cache_context("uncond"):
                        neg_noise_pred = transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=joint_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                progress_bar.update()

        latents = flux_pipeline._unpack_latents(
            latents,
            height=image_height,
            width=image_width,
            vae_scale_factor=flux_pipeline.vae_scale_factor,
        )
        latents = (latents / flux_pipeline.vae.config.scaling_factor) + flux_pipeline.vae.config.shift_factor

        with torch.no_grad():
            decoded = flux_pipeline.vae.decode(latents, return_dict=False)[0]

        image = flux_pipeline.image_processor.postprocess(decoded, output_type="pt")
        image = image.permute(0, 2, 3, 1).contiguous()
        image = torch.clamp(image, 0.0, 1.0).to(dtype=torch.float32)

        return (image,)


NODE_CLASS_MAPPINGS = {
    "FluxDiffusersLoader": FluxDiffusersLoader,
    "OmniXLoRALoader": OmniXLoRALoader,
    "OmniXPerceptionDiffusers": OmniXPerceptionDiffusers,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxDiffusersLoader": "Flux Loader (Diffusers)",
    "OmniXLoRALoader": "OmniX LoRA Loader",
    "OmniXPerceptionDiffusers": "OmniX Perception (Diffusers)",
}
