"""Flash attention utilities with graceful fallback.

Handles ABI mismatches when flash_attn is installed but incompatible
with the current PyTorch version.
"""

import torch
from transformers.modeling_flash_attention_utils import (
    is_flash_attn_available,
    flash_attn_supports_top_left_mask,
)

# Check if flash_attn is actually usable (not just installed but ABI-compatible)
_flash_attn_available = False
flash_attn_varlen_func = None
_flash_attention_forward = None

if is_flash_attn_available():
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
        from flash_attn import flash_attn_varlen_func
        _flash_attn_available = True
    except (ImportError, OSError) as e:
        # ImportError: module not found
        # OSError: ABI mismatch (undefined symbol errors)
        import warnings
        warnings.warn(f"flash_attn installed but not usable (ABI mismatch?): {e}")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.

    Removed from transformers 5.x modeling_flash_attention_utils,
    so we provide a local implementation.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x_embed = (x * cos) + (_rotate_half(x) * sin)
    return x_embed
