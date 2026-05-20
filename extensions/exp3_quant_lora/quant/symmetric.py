from typing import Tuple

import torch


def _rowwise_scale(weight: torch.Tensor, max_abs_q: float, eps: float = 1e-8) -> torch.Tensor:
  max_abs = weight.abs().amax(dim=1, keepdim=True).clamp_min(eps)
  return max_abs / max_abs_q


def quantize_weight(weight: torch.Tensor, num_bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Quantize linear weight tensor [out_features, in_features].

  Returns:
    - qweight: int8 (4-bit path) or uint8 codebook index (2-bit path)
    - scale: row-wise scale tensor [out_features, 1]
  """
  if num_bits not in (2, 4):
    raise ValueError(f"num_bits must be 2 or 4, got {num_bits}")

  w = weight.detach().to(torch.float32)

  if num_bits == 4:
    # Symmetric int4 represented in int8 container: values in [-8, 7].
    scale = _rowwise_scale(w, max_abs_q=7.0)
    q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)
    return q, scale

  # 2-bit: store nearest index in 4-level symmetric codebook.
  scale = _rowwise_scale(w, max_abs_q=1.5)
  levels = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32, device=w.device)
  normalized = w / scale
  # [out, in, 4], nearest codebook level by absolute distance.
  distances = (normalized.unsqueeze(-1) - levels).abs()
  idx = torch.argmin(distances, dim=-1).to(torch.uint8)
  return idx, scale


def dequantize_weight(qweight: torch.Tensor, scale: torch.Tensor, num_bits: int) -> torch.Tensor:
  if num_bits not in (2, 4):
    raise ValueError(f"num_bits must be 2 or 4, got {num_bits}")

  if num_bits == 4:
    return qweight.to(torch.float32) * scale

  levels = torch.tensor(
    [-1.5, -0.5, 0.5, 1.5],
    dtype=torch.float32,
    device=qweight.device,
  )
  deq = levels[qweight.long()]
  return deq * scale
