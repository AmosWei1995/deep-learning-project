import torch

from quant.symmetric import dequantize_weight, quantize_weight


def init_loftq(
  A: torch.Tensor,
  B: torch.Tensor,
  weight: torch.Tensor,
  rank: int,
  num_bits: int = 4,
  **kwargs,
) -> None:
  """
  LoftQ-style initialization:
    1) Quantize full-precision base weight.
    2) Compute residual in LoRA path space: (W - W_q)^T with shape [in, out].
    3) SVD residual and place factors into A, B.
  """
  if weight is None:
    raise ValueError("init_loftq requires `weight` tensor from the original linear layer.")

  with torch.no_grad():
    qweight, scale = quantize_weight(weight, num_bits=num_bits)
    wq = dequantize_weight(qweight, scale, num_bits=num_bits)
    residual = (weight - wq).transpose(0, 1).to(torch.float32)  # [in, out]

    A.zero_()
    B.zero_()
    if residual.numel() == 0:
      return

    U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
    r = min(rank, U.size(1), Vh.size(0))
    if r <= 0:
      return

    sqrt_s = torch.sqrt(S[:r])
    A[:, :r] = (U[:, :r] * sqrt_s).to(dtype=A.dtype, device=A.device)
    B[:r, :] = (sqrt_s.unsqueeze(1) * Vh[:r, :]).to(dtype=B.dtype, device=B.device)
