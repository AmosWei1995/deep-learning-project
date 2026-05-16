import torch
import torch.nn as nn


class LoRALinear(nn.Module):
  def __init__(
    self,
    original_linear: nn.Linear,
    rank: int = 8,
    alpha: float = 16.0,
    init_method: str = 'lora',
  ):
    super().__init__()
    self.original_linear = original_linear
    for p in self.original_linear.parameters():
      p.requires_grad = False

    in_features = original_linear.in_features
    out_features = original_linear.out_features
    self.rank = rank
    self.alpha = alpha

    self.A = nn.Parameter(torch.empty(in_features, rank))
    self.B = nn.Parameter(torch.empty(rank, out_features))

    if init_method == 'lora':
      init_lora(self.A, self.B)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.original_linear(x) + (self.alpha / self.rank) * (x @ self.A @ self.B)


def init_lora(A: torch.Tensor, B: torch.Tensor, **kwargs) -> None:
    nn.init.normal_(A)
    nn.init.zeros_(B)


def apply_lora(
  model: nn.Module,
  rank: int = 8,
  alpha: float = 16.0,
  init_method: str = 'lora',
  target_modules: list = ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
) -> nn.Module:
  for name, module in model.named_modules():
    for attr_name in target_modules:
      if hasattr(module, attr_name):
        child = getattr(module, attr_name)
        if isinstance(child, nn.Linear):
          setattr(module, attr_name, LoRALinear(child, rank, alpha, init_method))
  return model
