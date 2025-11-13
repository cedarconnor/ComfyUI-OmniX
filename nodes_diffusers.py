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


class FluxDiffusersLoader:
    """
    Load Flux.1 models through Diffusers, optionally from local checkpoints.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import os

        checkpoint_files: List[str] = []
        unet_path = os.path.join(folder_paths.models_dir, "unet")
        checkpoints = folder_paths.get_folder_paths("checkpoints")
        checkpoint_dir = checkpoints[0] if checkpoints else None

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

        checkpoint_files = sorted(list(set(checkpoint_files))) or ["flux1-dev.sft"]

        return {
            "required": {
                "load_method": (["huggingface", "local_file"], {"default": "local_file"}),
                "torch_dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            },
            "optional": {
                "model_id": (
                    "STRING",
                    {
                        "default": "black-forest-labs/FLUX.1-dev",
                        "multiline": False,
                    },
                ),
                "local_checkpoint": (checkpoint_files, {"default": checkpoint_files[0]}),
            },
        }

    RETURN_TYPES = ("FLUX_PIPELINE", "VAE", "TEXT_ENCODER")
    RETURN_NAMES = ("flux_pipeline", "vae", "text_encoder")
    FUNCTION = "load_pipeline"
    CATEGORY = "OmniX/Diffusers"

    @staticmethod
    def _find_checkpoint(filename: str) -> Path:
        search_paths = [
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
        try:
            return FluxPipeline.from_single_file(path, torch_dtype=torch_dtype)
        except Exception as err:
            if "weights_only" not in str(err):
                raise

            logging.warning(
                "Diffusers refused to load %s with weights_only restriction; retrying with safe override.",
                path,
            )
            import torch.serialization

            original_load = torch.load

            def patched_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return original_load(*args, **kwargs)

            torch.load = patched_load
            try:
                return FluxPipeline.from_single_file(path, torch_dtype=torch_dtype)
            finally:
                torch.load = original_load

    def load_pipeline(
        self,
        load_method: str,
        torch_dtype: str,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        local_checkpoint: str = "flux1-dev.sft",
    ):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map[torch_dtype]
        device = model_management.get_torch_device()

        if load_method == "local_file":
            checkpoint = self._find_checkpoint(local_checkpoint)
            print(f"[FluxDiffusers] Loading Flux from {checkpoint}")
            pipeline = self._load_single_file(checkpoint, dtype)
        else:
            print(f"[FluxDiffusers] Downloading Flux model: {model_id}")
            pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)

        pipeline = pipeline.to(device)
        pipeline.vae.enable_tiling()
        print(f"[FluxDiffusers] Pipeline ready on {device}")
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
        return {
            "required": {
                "flux_pipeline": ("FLUX_PIPELINE",),
                "adapter_dir": ("STRING", {"default": "C:/ComfyUI/models/loras/omnix"}),
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
            print("[OmniXLoRA] Warning: no adapters selected.")
            return flux_pipeline, {}

        loaded: Dict[str, Dict[str, torch.Tensor]] = {}
        adapter_names: List[str] = []
        adapter_scales: List[float] = []

        for adapter_name in requested:
            filename = self.ADAPTER_FILENAMES[adapter_name]
            adapter_path = adapter_dir / filename
            if not adapter_path.exists():
                print(f"[OmniXLoRA] Missing adapter: {adapter_path}")
                continue

            print(f"[OmniXLoRA] Loading {adapter_name} from {adapter_path}")
            state = load_file(str(adapter_path))
            flux_pipeline.load_lora_weights(
                adapter_dir,
                weight_name=filename,
                adapter_name=adapter_name,
            )
            loaded[adapter_name] = {
                "path": str(adapter_path),
                "weights": len(state),
                "scale": lora_scale,
            }
            adapter_names.append(adapter_name)
            adapter_scales.append(lora_scale)

        if adapter_names:
            flux_pipeline.set_adapters(adapter_names, adapter_weights=adapter_scales)
            print(f"[OmniXLoRA] Active adapters: {adapter_names}")

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
                "task": (["distance", "normal", "albedo", "pbr"], {"default": "distance"}),
                "num_steps": ("INT", {"default": 28, "min": 1, "max": 80, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.5}),
                "noise_strength": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05}),
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
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        image_tensor = self._image_to_tensor(panorama, flux_pipeline)
        with torch.no_grad():
            latents = flux_pipeline.vae.encode(image_tensor).latent_dist.sample()
        latents = latents * flux_pipeline.vae.config.scaling_factor

        vae_scale = getattr(flux_pipeline, "vae_scale_factor", 8)
        height = panorama.shape[1]
        width = panorama.shape[2]
        height = 2 * (int(height) // (vae_scale * 2))
        width = 2 * (int(width) // (vae_scale * 2))

        num_channels_latents = flux_pipeline.transformer.config.in_channels // 4
        packed = flux_pipeline._pack_latents(
            latents.to(device=flux_pipeline.device, dtype=flux_pipeline.dtype),
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
        )
        latent_image_ids = flux_pipeline._prepare_latent_image_ids(
            1, height // 2, width // 2, flux_pipeline.device, flux_pipeline.dtype
        )
        return packed, latent_image_ids, height, width

    def _prepare_schedule(
        self,
        flux_pipeline: FluxPipeline,
        packed_latents: torch.Tensor,
        num_steps: int,
        noise_strength: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scheduler = flux_pipeline.scheduler
        device = flux_pipeline.device
        image_seq_len = packed_latents.shape[1]

        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        base_sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        scheduler.set_timesteps(num_steps, device=device, sigmas=base_sigmas, mu=mu)
        timesteps = scheduler.timesteps
        order = getattr(scheduler, "order", 1)

        strength = float(np.clip(noise_strength, 0.05, 0.99))
        init_timestep = max(1, min(int(num_steps * strength), num_steps))
        t_start = max(num_steps - init_timestep, 0)
        start_index = min(t_start * order, len(timesteps) - 1)
        trimmed_timesteps = timesteps[start_index:]
        if trimmed_timesteps.numel() == 0:
            trimmed_timesteps = timesteps[-1:]
            start_index = len(timesteps) - 1

        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(start_index)

        noise = randn_tensor(packed_latents.shape, device=device, dtype=packed_latents.dtype)
        latents = scheduler.add_noise(packed_latents, noise, trimmed_timesteps[:1])
        return latents, trimmed_timesteps

    def run_perception(
        self,
        flux_pipeline,
        loaded_adapters,
        panorama,
        task,
        num_steps,
        guidance_scale,
        noise_strength,
    ):
        if task not in loaded_adapters:
            raise ValueError(f"Adapter '{task}' not loaded. Available: {list(loaded_adapters)}")

        flux_pipeline.set_adapters([task], adapter_weights=[loaded_adapters[task]["scale"]])
        print(f"[OmniXPerception-Diffusers] Task={task}, steps={num_steps}, noise={noise_strength}")

        packed_latents, latent_image_ids, height, width = self._encode_latents(flux_pipeline, panorama)
        latents, timesteps = self._prepare_schedule(
            flux_pipeline,
            packed_latents,
            num_steps=num_steps,
            noise_strength=noise_strength,
        )

        prompt = f"perception task: {task}"
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
        joint_kwargs = flux_pipeline.joint_attention_kwargs or {}

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
            height=height,
            width=width,
            vae_scale_factor=flux_pipeline.vae_scale_factor,
        )
        latents = (latents / flux_pipeline.vae.config.scaling_factor) + flux_pipeline.vae.config.shift_factor

        with torch.no_grad():
            decoded = flux_pipeline.vae.decode(latents, return_dict=False)[0]

        image = flux_pipeline.image_processor.postprocess(decoded, output_type="pt")
        image = image.permute(0, 2, 3, 1).contiguous()
        image = torch.clamp(image, 0.0, 1.0)

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
