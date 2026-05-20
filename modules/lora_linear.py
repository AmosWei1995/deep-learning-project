import torch
import torch.nn as nn
from typing import Callable, Optional


class LoRALinear(nn.Module):
  def __init__(
    self,
    original_linear: nn.Linear,
    rank: int = 8,
    alpha: float = 16.0,
    init_fn: Optional[Callable] = None,
  ):
    super().__init__()
    self.original_linear = original_linear
    for p in self.original_linear.parameters():
      p.requires_grad = False

    # Use actual weight shape rather than .in_features/.out_features attributes:
    # the project loads pretrained weights via weight.data assignment, which can
    # silently produce a shape mismatch between the attribute and the real tensor
    # (e.g. interm_dense is constructed with intermediate_size=2304 from config
    # but the loaded HuggingFace weight has shape (3072, 768)).
    out_features = original_linear.weight.shape[0]
    in_features  = original_linear.weight.shape[1]
    self.rank = rank
    self.alpha = alpha

    self.A = nn.Parameter(torch.empty(in_features, rank))
    self.B = nn.Parameter(torch.empty(rank, out_features))

    fn = init_fn if init_fn is not None else init_lora
    fn(self.A, self.B, original_linear.weight, rank=rank, alpha=alpha)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.original_linear(x) + (self.alpha / self.rank) * (x @ self.A @ self.B)


def init_lora(A: torch.Tensor, B: torch.Tensor, weight=None, rank=None, **kwargs) -> None:
    nn.init.normal_(A)
    nn.init.zeros_(B)


def apply_lora(
  model: nn.Module,
  rank: int = 8,
  alpha: float = 16.0,
  init_fn: Optional[Callable] = None,
  target_modules: list = ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
) -> nn.Module:
  for name, module in model.named_modules():
    for attr_name in target_modules:
      if hasattr(module, attr_name):
        child = getattr(module, attr_name)
        if isinstance(child, nn.Linear):
          setattr(module, attr_name, LoRALinear(child, rank, alpha, init_fn))
  return model
