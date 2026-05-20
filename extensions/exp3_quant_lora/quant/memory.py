from typing import Optional

import torch


def reset_peak_memory(device: torch.device) -> None:
  if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)


def peak_memory_gb(device: torch.device) -> Optional[float]:
  if device.type != "cuda":
    return None
  peak = torch.cuda.max_memory_allocated(device=device)
  return peak / (1024 ** 3)
