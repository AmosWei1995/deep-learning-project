from typing import Callable, List, Optional

import torch.nn as nn

from adapters.quant_lora_linear import QuantLoRALinear


def apply_quant_lora(
  model: nn.Module,
  rank: int = 8,
  alpha: float = 16.0,
  num_bits: int = 4,
  method: str = "qlora",
  init_fn: Optional[Callable] = None,
  target_modules: List[str] = None,
) -> nn.Module:
  if target_modules is None:
    target_modules = ["query", "key", "value"]

  for _, module in model.named_modules():
    for attr_name in target_modules:
      if not hasattr(module, attr_name):
        continue
      child = getattr(module, attr_name)
      if isinstance(child, nn.Linear):
        setattr(
          module,
          attr_name,
          QuantLoRALinear(
            child,
            rank=rank,
            alpha=alpha,
            num_bits=num_bits,
            method=method,
            init_fn=init_fn,
          ),
        )
  return model
