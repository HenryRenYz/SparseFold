# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

import torch
import einops
from sparsefold.runtime import config
from sparsefold.runtime import util
from sparsefold.runtime import trace


def _get_backend():
  match config.sdpa_implementation:
    case "math":
      return torch.nn.attention.SDPBackend.MATH
    case "flash":
      return torch.nn.attention.SDPBackend.FLASH_ATTENTION
    case "efficient":
      return torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
    case _:
      raise ValueError(f"Unsupported SDPA implementation: {config.sdpa_implementation!r}")

def dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    mask_q: torch.Tensor | None = None,
    mask_k: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes Scaled Dot Product Attention with support for masks, biases.

    This function fuses batch dimensions to optimize kernel execution and utilizes 
    PyTorch's SDPA (Scaled Dot Product Attention) for hardware acceleration.

    Args:
        q: Query tensor of shape (..., Q, H, Dqk).
        k: Key tensor of shape (..., Kv, H, Dqk).
        v: Value tensor of shape (..., Kv, H, Dv).
        scale: Scaling factor for attention scores. If None, defaults to 1 / sqrt(Dqk).
        mask_q: Optional boolean mask for queries of shape (..., Q). False elements are masked.
        mask_k: Optional boolean mask for keys of shape (..., Kv). False elements are masked.
        bias: Optional additive bias tensor of shape (..., H, Q, Kv).

    Returns:
        Output tensor of shape (..., Q, H, Dv).
    """
    dtype = q.dtype
    device = q.device
    
    # Capture original batch shape for restoration later
    # q shape: (Batch..., Q, H, D)
    input_shape = q.shape
    batch_shape = input_shape[:-3]
    
    # --- 1. Efficient Mask and Bias Construction ---
    attn_bias = None
    
    # Initialize attn_bias only if masks or biases are present
    if mask_q is not None or mask_k is not None or bias is not None:
        min_val = -float(torch.finfo(dtype).max)

        # Initialize bias tensor with shape (Batch..., 1, Q, K) to allow broadcasting
        attn_bias = torch.zeros(batch_shape + (1, q.shape[-3], k.shape[-3]), 
                                device=device, dtype=dtype)
        if mask_q is not None:
            # Expand (..., Q) -> (..., 1, Q, 1) to broadcast against (..., H, Q, K)
            mask_q_view = mask_q[..., None, :, None]
            attn_bias.masked_fill_(~mask_q_view.bool(), min_val)
            
        if mask_k is not None:
            # Expand (..., K) -> (..., 1, 1, K)
            mask_k_view = mask_k[..., None, None, :]
            attn_bias.masked_fill_(~mask_k_view.bool(), min_val)

        # Add external bias
        if bias is not None:
            # bias: (..., H, Q, K)
            attn_bias = attn_bias + bias.to(dtype)

    # --- 2. Dimension Fusion ---
    # Flatten batch dims and permute: (..., Q, H, D) -> (FlatBatch, H, Q, D)
    # This ensures memory is contiguous and compatible with SDPA
    q = einops.rearrange(q, '... q h d -> (...) h q d')
    k = einops.rearrange(k, '... k h d -> (...) h k d')
    v = einops.rearrange(v, '... v h d -> (...) h v d')

    # Align attn_bias dimensions to the flattened batch structure
    if attn_bias is not None:  
        # Flatten: (Batch..., H, Q, K) -> (FlatBatch, H, Q, K)
        if attn_bias.shape[:-3] != batch_shape:
             attn_bias = attn_bias.expand(*batch_shape, *attn_bias.shape[-3:])
        
        attn_bias = einops.rearrange(attn_bias, '... h q k -> (...) h q k')

    # --- 3. SDPA Execution ---
    with torch.nn.attention.sdpa_kernel(backends=[_get_backend()]):
        def _trace_probs(slice: util.Slice):
            _scale = scale if scale is not None else 1 / q.shape[-1] ** 0.5
            _q, _k = util.slice_tensor(q, k, slice=slice)
            _b = util.slice_tensor(attn_bias, slice=slice) if attn_bias is not None else 0
            logits = einops.einsum(_q, _k * _scale, "... l e, ... s e -> ... l s") + _b
            logits = torch.softmax(logits, dim=-1)
            return {"q": _q, "k": _k, "bias": _b, "logits": logits}
        
        trace.log_tensor(probs=_trace_probs)
        output = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            scale=scale,
            attn_mask=attn_bias,
        )

    # --- 4. Restore Dimensions ---
    # (FlatBatch, H, Q, D) -> (Batch..., Q, H, D)
    output = einops.rearrange(output, '(b) h q d -> b q h d', b=q.shape[0])
    output = output.view(*batch_shape, *output.shape[-3:])

    return output
