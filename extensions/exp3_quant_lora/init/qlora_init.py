import torch
import torch.nn as nn


def init_qlora(
  A: torch.Tensor,
  B: torch.Tensor,
  weight: torch.Tensor = None,
  rank: int = None,
  **kwargs,
) -> None:
  # Same initializer behavior as standard LoRA.
  nn.init.normal_(A)
  nn.init.zeros_(B)
