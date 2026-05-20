from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from init.loftq_init import init_loftq
from init.qlora_init import init_qlora
from quant.symmetric import dequantize_weight, quantize_weight


class QuantLoRALinear(nn.Module):
  def __init__(
    self,
    original_linear: nn.Linear,
    rank: int = 8,
    alpha: float = 16.0,
    num_bits: int = 4,
    method: str = "qlora",
    init_fn: Optional[Callable] = None,
  ):
    super().__init__()
    if num_bits not in (2, 4):
      raise ValueError(f"num_bits must be 2 or 4, got {num_bits}")
    if rank <= 0:
      raise ValueError(f"rank must be positive, got {rank}")

    weight = original_linear.weight.detach()
    qweight, scale = quantize_weight(weight, num_bits=num_bits)

    self.rank = rank
    self.alpha = alpha
    self.num_bits = num_bits
    self.method = method
    self.in_features = original_linear.in_features
    self.out_features = original_linear.out_features

    self.register_buffer("qweight", qweight)
    self.register_buffer("scale", scale)
    if original_linear.bias is not None:
      self.register_buffer("base_bias", original_linear.bias.detach().clone())
    else:
      self.base_bias = None

    self.A = nn.Parameter(torch.empty(self.in_features, rank))
    self.B = nn.Parameter(torch.empty(rank, self.out_features))

    fn = init_fn
    if fn is None:
      fn = init_loftq if method == "loftq" else init_qlora
    fn(self.A, self.B, weight=weight, rank=rank, num_bits=num_bits)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    base_weight = dequantize_weight(self.qweight, self.scale, self.num_bits)
    base_weight = base_weight.to(dtype=x.dtype)
    bias = self.base_bias.to(dtype=x.dtype) if self.base_bias is not None else None
    base_out = F.linear(x, base_weight, bias)
    lora_out = (self.alpha / self.rank) * (x @ self.A @ self.B)
    return base_out + lora_out
