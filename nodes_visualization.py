"""
Visualization utilities for OmniX perception outputs.
"""

from __future__ import annotations

import torch


def _luminance(tensor: torch.Tensor) -> torch.Tensor:
    return (
        0.299 * tensor[..., 0:1]
        + 0.587 * tensor[..., 1:2]
        + 0.114 * tensor[..., 2:3]
    )


def _viridis(depth: torch.Tensor) -> torch.Tensor:
    stops = depth.new_tensor(
        [
            [0.267, 0.005, 0.329],
            [0.283, 0.141, 0.458],
            [0.254, 0.265, 0.530],
            [0.207, 0.372, 0.553],
            [0.164, 0.471, 0.558],
            [0.134, 0.658, 0.517],
            [0.477, 0.821, 0.318],
            [0.993, 0.906, 0.144],
        ]
    )
    n_segments = stops.shape[0] - 1

    flat = depth.reshape(-1)
    scaled = torch.clamp(flat * n_segments, 0, n_segments - 1 + 1e-6)
    idx = scaled.floor().long()
    frac = (scaled - idx.float()).unsqueeze(-1)

    low = stops[idx]
    high = stops[torch.clamp(idx + 1, max=n_segments)]
    colored = torch.lerp(low, high, frac)
    return colored.view(*depth.shape[:-1], 3)


def _normalize_percentile(tensor: torch.Tensor, low: float, high: float) -> torch.Tensor:
    q_low = torch.quantile(tensor, low)
    q_high = torch.quantile(tensor, high)
    if q_high - q_low < 1e-6:
        return torch.zeros_like(tensor)
    return torch.clamp((tensor - q_low) / (q_high - q_low), 0.0, 1.0)


def _apply_gamma(tensor: torch.Tensor, gamma: float) -> torch.Tensor:
    if abs(gamma - 1.0) < 1e-6:
        return tensor
    return torch.pow(torch.clamp(tensor, 0.0, 1.0), 1.0 / gamma)


def _adjust_saturation(image: torch.Tensor, saturation: float) -> torch.Tensor:
    if abs(saturation - 1.0) < 1e-6:
        return image
    luma = _luminance(image)
    return torch.clamp(luma + (image - luma) * saturation, 0.0, 1.0)


class DepthVisualization:
    """
    Normalize and colorize raw OmniX depth outputs for easier inspection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (
                    ["grayscale", "viridis"],
                    {"default": "viridis", "tooltip": "Choose a grayscale or viridis false-color ramp for the normalized depth."},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 1.2, "min": 0.1, "max": 4.0, "step": 0.05, "tooltip": "Apply gamma to the normalized depth (values <1 boost contrast)."},
                ),
                "percentile_low": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.01, "tooltip": "Clamp the lower percentile to suppress outliers (e.g., 0.02)."},
                ),
                "percentile_high": (
                    "FLOAT",
                    {"default": 0.98, "min": 0.8, "max": 1.0, "step": 0.01, "tooltip": "Clamp the upper percentile (e.g., 0.98) to highlight mid-range depth."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualized",)
    FUNCTION = "visualize"
    CATEGORY = "OmniX/Visualization"

    def visualize(self, image, mode, gamma, percentile_low, percentile_high):
        tensor = image.clone().float()

        luminance = _luminance(tensor)
        normalized = _normalize_percentile(luminance, percentile_low, percentile_high)
        normalized = _apply_gamma(normalized, gamma)

        if mode == "grayscale":
            visualized = normalized.repeat(1, 1, 1, 3)
        else:
            visualized = _viridis(normalized)

        return (visualized.clamp(0.0, 1.0),)


class NormalVisualization:
    """
    Re-normalize OmniX normal outputs to [-1, 1] and map to [0, 1] RGB.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "normalize_vectors": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Normalize each XYZ vector back to unit length before visualization."},
                ),
                "invert_y": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Flip the Y axis (enable for OpenGL-style normals)."},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05, "tooltip": "Gamma correction on the final RGB normal map."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualized",)
    FUNCTION = "visualize"
    CATEGORY = "OmniX/Visualization"

    def visualize(self, image, normalize_vectors, invert_y, gamma):
        normals = image.clone().float() * 2.0 - 1.0
        if invert_y:
            normals[..., 1] = -normals[..., 1]

        if normalize_vectors:
            norm = torch.linalg.norm(normals, dim=-1, keepdim=True).clamp(min=1e-6)
            normals = normals / norm

        mapped = torch.clamp(normals * 0.5 + 0.5, 0.0, 1.0)
        mapped = _apply_gamma(mapped, gamma)
        return (mapped, )


class AlbedoVisualization:
    """
    Simple tone-mapping for OmniX albedo outputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05, "tooltip": "Gamma correction applied to the albedo map."},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.5, "step": 0.05, "tooltip": "Blend between grayscale (0) and boosted color (>1)."},
                ),
                "exposure": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "tooltip": "Linear exposure gain before gamma (use <1 to darken)."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualized",)
    FUNCTION = "visualize"
    CATEGORY = "OmniX/Visualization"

    def visualize(self, image, gamma, saturation, exposure):
        tensor = torch.clamp(image.clone().float(), 0.0, 1.0)
        tensor = torch.clamp(tensor * exposure, 0.0, 1.0)
        tensor = _apply_gamma(tensor, gamma)
        tensor = _adjust_saturation(tensor, saturation)
        return (tensor, )


class PBRVisualization:
    """
    Visualize OmniX roughness/metallic outputs.
    """

    MODES = ["roughness", "metallic", "combined"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (
                    cls.MODES,
                    {"default": "roughness", "tooltip": "Pick which channel to inspect: roughness, metallic, or the combined RG/B view."},
                ),
                "palette": (
                    ["grayscale", "viridis"],
                    {"default": "viridis", "tooltip": "Color mapping used for the selected channel."},
                ),
                "percentile_low": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.01, "tooltip": "Lower percentile clamp for roughness/metallic normalization."},
                ),
                "percentile_high": (
                    "FLOAT",
                    {"default": 0.98, "min": 0.8, "max": 1.0, "step": 0.01, "tooltip": "Upper percentile clamp for roughness/metallic normalization."},
                )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualized",)
    FUNCTION = "visualize"
    CATEGORY = "OmniX/Visualization"

    def visualize(self, image, mode, palette, percentile_low, percentile_high):
        tensor = torch.clamp(image.clone().float(), 0.0, 1.0)

        if mode == "combined":
            channels = tensor.clone()
            if channels.shape[-1] == 1:
                channels = channels.repeat(1, 1, 1, 3)
            elif channels.shape[-1] == 2:
                rough = channels[..., 0:1]
                metal = channels[..., 1:2]
                channels = torch.cat([rough, torch.zeros_like(rough), metal], dim=-1)
            return (channels.clamp(0.0, 1.0),)

        channel_idx = 0 if mode == "roughness" else 1
        if tensor.shape[-1] <= channel_idx:
            channel_idx = 0

        selected = tensor[..., channel_idx:channel_idx + 1]
        normalized = _normalize_percentile(selected, percentile_low, percentile_high)

        if palette == "grayscale":
            visualized = normalized.repeat(1, 1, 1, 3)
        else:
            visualized = _viridis(normalized)

        return (visualized.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "DepthVisualization": DepthVisualization,
    "NormalVisualization": NormalVisualization,
    "AlbedoVisualization": AlbedoVisualization,
    "PBRVisualization": PBRVisualization,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthVisualization": "Depth Visualization",
    "NormalVisualization": "Normal Visualization",
    "AlbedoVisualization": "Albedo Visualization",
    "PBRVisualization": "PBR Visualization",
}
