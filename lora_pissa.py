import torch


def init_pissa(
  A: torch.Tensor,
  B: torch.Tensor,
  weight: torch.Tensor,
  rank: int,
  alpha: float = 16.0,
  **kwargs,
) -> None:
  """
  Initialize LoRA factors with PiSSA.

  PiSSA decomposes the pretrained linear weight into a frozen residual plus a
  trainable low-rank principal component. This project stores LoRA factors as
  A: (in_features, rank), B: (rank, out_features), and applies
  (alpha / rank) * x @ A @ B in the forward pass.
  """
  if weight is None:
    raise ValueError("PiSSA initialization requires the original linear weight.")
  if rank is None or rank <= 0:
    raise ValueError(f"rank must be a positive integer, got {rank}.")
  if alpha == 0:
    raise ValueError("alpha must be non-zero for PiSSA initialization.")

  max_rank = min(weight.shape)
  if rank > max_rank:
    raise ValueError(f"rank={rank} exceeds max feasible rank={max_rank} for weight shape {tuple(weight.shape)}.")

  dtype = weight.dtype
  device = weight.device
  scale = alpha / rank

  # nn.Linear stores weight as (out_features, in_features), while this LoRA
  # implementation adds x @ A @ B, i.e. it operates on weight.T.
  weight_t = weight.detach().to(device=device, dtype=torch.float32).T
  U, S, Vh = torch.linalg.svd(weight_t, full_matrices=False)

  U_r = U[:, :rank]
  S_r = S[:rank] / scale
  Vh_r = Vh[:rank, :]
  sqrt_s = torch.sqrt(S_r)

  A_init = U_r * sqrt_s.unsqueeze(0)
  B_init = sqrt_s.unsqueeze(1) * Vh_r

  with torch.no_grad():
    A.copy_(A_init.to(dtype=A.dtype, device=A.device))
    B.copy_(B_init.to(dtype=B.dtype, device=B.device))

    principal = scale * (A_init @ B_init)
    residual_t = weight_t - principal
    weight.copy_(residual_t.T.to(dtype=dtype, device=device))
