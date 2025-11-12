"""
OmniX Panorama Perception

Extracts geometric and material properties from panoramas using OmniX adapters.
Outputs: distance (depth), normals, albedo, roughness, metallic maps.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .adapters import OmniXAdapters, AdapterModule
from .utils import from_comfyui_image, to_comfyui_image


class PanoramaEncoder(nn.Module):
    """
    Encodes panorama images into feature representations.
    This feeds into perception adapters for property extraction.

    Uses a lightweight CNN-based encoder to extract multi-scale features
    that are suitable for perception tasks (depth, normals, materials).
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_channels: int = 256,
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.dtype = dtype
        self.feature_channels = feature_channels

        # Multi-scale feature extraction network
        # This creates hierarchical features suitable for perception tasks

        # Initial convolution
        self.conv_in = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.norm_in = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Encoder blocks (progressive downsampling)
        self.encoder_block1 = self._make_encoder_block(64, 128, stride=2)
        self.encoder_block2 = self._make_encoder_block(128, 256, stride=2)
        self.encoder_block3 = self._make_encoder_block(256, feature_channels, stride=1)

        # Global context aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

        # Initialize weights
        self._initialize_weights()

        # Convert to target dtype
        self.to(dtype=dtype)

    def _make_encoder_block(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ) -> nn.Module:
        """Create an encoder block with residual connections"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, panorama: torch.Tensor) -> torch.Tensor:
        """
        Encode panorama to feature representation.

        Args:
            panorama: Input panorama (B, C, H, W) or (B, H, W, C)

        Returns:
            Encoded features (B, feature_channels, H', W')
        """
        # Ensure correct channel order (B, C, H, W)
        if panorama.shape[-1] == 3:  # ComfyUI format (B, H, W, C)
            panorama = panorama.permute(0, 3, 1, 2)

        # Move to correct device and dtype
        device = self.conv_in.weight.device
        panorama = panorama.to(device=device, dtype=self.dtype)

        # Normalize input to [-1, 1] range
        panorama = panorama * 2.0 - 1.0

        # Encode through network
        x = self.conv_in(panorama)
        x = self.norm_in(x)
        x = self.relu(x)

        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        x = self.encoder_block3(x)

        # Project features
        features = self.feature_proj(x)

        return features


class SimplePanoramaEncoder(nn.Module):
    """
    Simplified panorama encoder for when full encoder is not needed.

    This is a lightweight alternative that applies minimal processing
    and is used when perception adapters can work directly with image features.
    """

    def __init__(self, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.dtype = dtype

        # Simple convolution for basic feature extraction
        self.feature_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
        )

        self.to(dtype=dtype)

    def forward(self, panorama: torch.Tensor) -> torch.Tensor:
        """
        Encode panorama to feature representation.

        Args:
            panorama: Input panorama (B, C, H, W) or (B, H, W, C)

        Returns:
            Encoded features
        """
        # Ensure correct channel order (B, C, H, W)
        if panorama.shape[-1] == 3:  # ComfyUI format (B, H, W, C)
            panorama = panorama.permute(0, 3, 1, 2)

        # Convert to correct dtype
        panorama = panorama.to(dtype=self.dtype)

        # Extract features
        features = self.feature_conv(panorama)

        return features


class OmniXPerceiver:
    """
    OmniX Panorama Perception Engine

    Extracts multiple properties from panoramas:
    - Distance (depth) maps
    - Normal maps (surface normals)
    - Albedo (base color/diffuse)
    - Roughness (PBR property)
    - Metallic (PBR property)
    """

    def __init__(self, adapters: OmniXAdapters):
        """
        Initialize perceiver.

        Args:
            adapters: OmniXAdapters instance with loaded adapter weights
        """
        self.adapters = adapters
        self.encoder = PanoramaEncoder()

        # Move encoder to GPU if available
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()

    def perceive(
        self,
        panorama: torch.Tensor,
        extract_modes: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract properties from panorama.

        Args:
            panorama: Input panorama in ComfyUI format (B, H, W, C)
            extract_modes: List of properties to extract
                          ["distance", "normal", "albedo", "roughness", "metallic"]

        Returns:
            Dictionary of extracted properties
        """
        if len(extract_modes) == 0:
            raise ValueError("No extraction modes specified")

        device = panorama.device
        results = {}

        # Encode panorama once (shared features)
        with torch.no_grad():
            features = self.encoder(panorama)

        # Run requested adapters
        for mode in extract_modes:
            if mode == "distance":
                results["distance"] = self._extract_distance(features)
            elif mode == "normal":
                results["normal"] = self._extract_normal(features)
            elif mode == "albedo":
                results["albedo"] = self._extract_albedo(features)
            elif mode == "roughness":
                results["roughness"] = self._extract_roughness(features)
            elif mode == "metallic":
                results["metallic"] = self._extract_metallic(features)
            else:
                print(f"Warning: Unknown perception mode '{mode}', skipping")

        return results

    def _extract_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract distance (depth) map.

        Returns:
            Distance map (B, H, W, 1) in ComfyUI format
        """
        adapter = self.adapters.get_adapter("distance")

        with torch.no_grad():
            # Run adapter
            distance = adapter(features)

            # Ensure single channel output
            if distance.shape[1] != 1:  # If multiple channels, average
                distance = distance.mean(dim=1, keepdim=True)

            # Convert to ComfyUI format (B, H, W, C)
            distance = distance.permute(0, 2, 3, 1)

            # Normalize to reasonable range
            distance = self._normalize_distance(distance)

        return distance

    def _extract_normal(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract normal map (surface normals).

        Returns:
            Normal map (B, H, W, 3) in ComfyUI format, range [-1, 1]
        """
        adapter = self.adapters.get_adapter("normal")

        with torch.no_grad():
            # Run adapter
            normal = adapter(features)

            # Ensure 3 channels (x, y, z)
            if normal.shape[1] != 3:
                raise ValueError(f"Normal adapter output unexpected shape: {normal.shape}")

            # Convert to ComfyUI format (B, H, W, C)
            normal = normal.permute(0, 2, 3, 1)

            # Normalize to unit vectors
            normal = torch.nn.functional.normalize(normal, dim=-1, p=2)

        return normal

    def _extract_albedo(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract albedo (base color/diffuse).

        Returns:
            Albedo map (B, H, W, 3) in ComfyUI format, range [0, 1]
        """
        adapter = self.adapters.get_adapter("albedo")

        with torch.no_grad():
            # Run adapter
            albedo = adapter(features)

            # Ensure 3 channels (RGB)
            if albedo.shape[1] != 3:
                # If single channel, replicate to 3
                albedo = albedo.repeat(1, 3, 1, 1)

            # Convert to ComfyUI format (B, H, W, C)
            albedo = albedo.permute(0, 2, 3, 1)

            # Clamp to [0, 1]
            albedo = torch.clamp(albedo, 0.0, 1.0)

        return albedo

    def _extract_roughness(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract roughness map (PBR material property).

        Returns:
            Roughness map (B, H, W, 1) in ComfyUI format, range [0, 1]
        """
        adapter = self.adapters.get_adapter("roughness")

        with torch.no_grad():
            # Run adapter
            roughness = adapter(features)

            # Ensure single channel output
            if roughness.shape[1] != 1:
                roughness = roughness.mean(dim=1, keepdim=True)

            # Convert to ComfyUI format (B, H, W, C)
            roughness = roughness.permute(0, 2, 3, 1)

            # Clamp to [0, 1]
            roughness = torch.clamp(roughness, 0.0, 1.0)

        return roughness

    def _extract_metallic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract metallic map (PBR material property).

        Returns:
            Metallic map (B, H, W, 1) in ComfyUI format, range [0, 1]
        """
        adapter = self.adapters.get_adapter("metallic")

        with torch.no_grad():
            # Run adapter
            metallic = adapter(features)

            # Ensure single channel output
            if metallic.shape[1] != 1:
                metallic = metallic.mean(dim=1, keepdim=True)

            # Convert to ComfyUI format (B, H, W, C)
            metallic = metallic.permute(0, 2, 3, 1)

            # Clamp to [0, 1]
            metallic = torch.clamp(metallic, 0.0, 1.0)

        return metallic

    def _normalize_distance(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Normalize distance map to reasonable range.

        Args:
            distance: Raw distance values

        Returns:
            Normalized distance in [0, 1] range
        """
        # Remove outliers (clip to percentiles)
        distance_flat = distance.flatten()
        p1 = torch.quantile(distance_flat, 0.01)
        p99 = torch.quantile(distance_flat, 0.99)

        distance = torch.clamp(distance, p1, p99)

        # Normalize to [0, 1]
        min_val = distance.min()
        max_val = distance.max()

        if max_val > min_val:
            distance = (distance - min_val) / (max_val - min_val)
        else:
            distance = torch.zeros_like(distance)

        return distance
