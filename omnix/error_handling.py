"""
OmniX Error Handling

Enhanced error handling for common failure scenarios:
- Missing adapter weights
- Out of memory (OOM) errors
- Model compatibility issues
- Invalid inputs
"""

import torch
import functools
from typing import Callable, Any
from pathlib import Path


class OmniXError(Exception):
    """Base exception for OmniX-specific errors"""
    pass


class AdapterWeightsNotFoundError(OmniXError):
    """Raised when adapter weights cannot be found"""

    def __init__(self, adapter_type: str, expected_path: Path):
        self.adapter_type = adapter_type
        self.expected_path = expected_path
        super().__init__(
            f"Adapter weights not found: {adapter_type}\n"
            f"Expected location: {expected_path}\n"
            f"\n"
            f"To fix this error:\n"
            f"1. Run: python download_models.py\n"
            f"2. Or manually download adapter weights from OmniX HuggingFace repository\n"
            f"3. Place weights in: {expected_path.parent}/\n"
            f"4. Ensure file is named: {adapter_type}_adapter.safetensors"
        )


class OutOfMemoryError(OmniXError):
    """Raised when GPU runs out of memory"""

    def __init__(self, operation: str, required_vram_gb: float = None):
        message = f"Out of memory during: {operation}\n\n"

        if required_vram_gb:
            message += f"Estimated VRAM required: {required_vram_gb:.1f} GB\n"

        message += (
            "To fix this error:\n"
            "1. Close other applications using GPU\n"
            "2. Reduce image resolution (try 1024x512 instead of 2048x1024)\n"
            "3. Use lower precision (fp16 instead of fp32)\n"
            "4. Reduce batch size to 1\n"
            "5. Unload unused models from memory\n"
            "6. Consider upgrading GPU or using CPU (slower)\n"
            "\n"
            "Check current memory: use omnix.utils.diagnose_memory()"
        )

        super().__init__(message)


class ModelCompatibilityError(OmniXError):
    """Raised when model is not compatible with OmniX"""

    def __init__(self, model_type: str, expected_type: str = "Flux.1-dev"):
        super().__init__(
            f"Model compatibility error\n"
            f"Model type: {model_type}\n"
            f"Expected: {expected_type}\n"
            f"\n"
            f"OmniX adapters are designed for Flux.1-dev models.\n"
            f"Using other models may produce unpredictable results.\n"
            f"\n"
            f"To fix:\n"
            f"1. Load Flux.1-dev model using ComfyUI's CheckpointLoaderSimple\n"
            f"2. Pass the Flux model to OmniXModelLoader\n"
            f"3. Then apply OmniX adapters"
        )


class InvalidPanoramaError(OmniXError):
    """Raised when panorama has invalid dimensions or format"""

    def __init__(self, width: int, height: int, expected_ratio: float = 2.0):
        current_ratio = width / height
        super().__init__(
            f"Invalid panorama dimensions: {width}x{height}\n"
            f"Current aspect ratio: {current_ratio:.2f}:1\n"
            f"Expected aspect ratio: {expected_ratio:.1f}:1 (equirectangular)\n"
            f"\n"
            f"To fix:\n"
            f"1. Crop or pad the panorama so width is exactly 2x height\n"
            f"2. Or manually resize to 2:1 ratio (e.g., 2048x1024, 4096x2048)\n"
            f"3. Choose fix method: crop, pad, or stretch"
        )


def handle_oom(operation_name: str = "operation"):
    """
    Decorator to catch and handle OOM errors gracefully.

    Usage:
        @handle_oom("adapter loading")
        def load_adapter(self):
            # ... code that might OOM
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                error_msg = str(e).lower()

                # Check if it's an OOM error
                if 'out of memory' in error_msg or 'cuda out of memory' in error_msg:
                    # Try to estimate required memory
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1e9
                        print(f"VRAM allocated when error occurred: {allocated:.2f} GB")

                    raise OutOfMemoryError(operation_name) from e
                else:
                    # Re-raise original error if not OOM
                    raise

        return wrapper
    return decorator


def check_vram_available(required_gb: float, operation: str = "operation"):
    """
    Check if sufficient VRAM is available before running operation.

    Args:
        required_gb: Required VRAM in GB
        operation: Name of operation (for error message)

    Raises:
        OutOfMemoryError if insufficient VRAM
    """
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU (will be slower)")
        return

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_vram_gb = props.total_memory / 1e9

    allocated_gb = torch.cuda.memory_allocated(device) / 1e9
    reserved_gb = torch.cuda.memory_reserved(device) / 1e9
    available_gb = total_vram_gb - reserved_gb

    print(f"VRAM check for {operation}:")
    print(f"  Required: {required_gb:.2f} GB")
    print(f"  Available: {available_gb:.2f} GB")
    print(f"  Total: {total_vram_gb:.2f} GB")

    # Allow 20% safety margin
    safety_margin = 0.2
    if available_gb < required_gb * (1 + safety_margin):
        raise OutOfMemoryError(operation, required_vram_gb=required_gb)


def validate_adapter_weights_exist(adapter_path: Path, adapter_type: str):
    """
    Validate that adapter weights file exists.

    Args:
        adapter_path: Path to adapter weights file
        adapter_type: Type of adapter

    Raises:
        AdapterWeightsNotFoundError if weights not found
    """
    if not adapter_path.exists():
        raise AdapterWeightsNotFoundError(adapter_type, adapter_path)

    # Check file is not empty
    if adapter_path.stat().st_size == 0:
        raise AdapterWeightsNotFoundError(
            adapter_type,
            adapter_path
        )


def safe_load_safetensors(path: Path, device: str = "cpu") -> dict:
    """
    Safely load safetensors file with error handling.

    Args:
        path: Path to .safetensors file
        device: Target device

    Returns:
        Dictionary of tensors

    Raises:
        OmniXError on failure
    """
    try:
        import safetensors.torch
        return safetensors.torch.load_file(str(path), device=device)
    except FileNotFoundError:
        raise OmniXError(
            f"Safetensors file not found: {path}\n"
            f"Ensure the file exists and path is correct."
        )
    except Exception as e:
        raise OmniXError(
            f"Failed to load safetensors file: {path}\n"
            f"Error: {str(e)}\n"
            f"\n"
            f"Possible causes:\n"
            f"1. File is corrupted\n"
            f"2. File is not a valid .safetensors file\n"
            f"3. Insufficient memory to load weights\n"
            f"\n"
            f"Try re-downloading the file: python download_models.py"
        )


def provide_helpful_error(error: Exception, context: str = "") -> str:
    """
    Convert generic errors to helpful error messages.

    Args:
        error: Original exception
        context: Additional context about what was being done

    Returns:
        Helpful error message string
    """
    error_type = type(error).__name__
    error_msg = str(error)

    if context:
        message = f"Error during {context}:\n"
    else:
        message = "Error occurred:\n"

    message += f"{error_type}: {error_msg}\n\n"

    # Add helpful suggestions based on error type
    if "FileNotFoundError" in error_type:
        message += (
            "File not found. Suggestions:\n"
            "1. Check that all model files are downloaded\n"
            "2. Run: python download_models.py\n"
            "3. Verify file paths in ComfyUI configuration\n"
        )
    elif "CUDA" in error_msg or "cuda" in error_msg:
        message += (
            "CUDA error. Suggestions:\n"
            "1. Update GPU drivers\n"
            "2. Check CUDA installation\n"
            "3. Try running on CPU (slower)\n"
            "4. Reduce batch size or resolution\n"
        )
    elif "shape" in error_msg.lower() or "size" in error_msg.lower():
        message += (
            "Tensor shape mismatch. Suggestions:\n"
            "1. Check input image dimensions\n"
            "2. Ensure panorama is 2:1 aspect ratio\n"
            "3. Ensure the panorama is 2:1 before running perception\n"
        )
    elif "dtype" in error_msg.lower():
        message += (
            "Data type mismatch. Suggestions:\n"
            "1. Ensure all inputs use same precision (fp16/fp32)\n"
            "2. Check adapter and model precision match\n"
        )

    return message
