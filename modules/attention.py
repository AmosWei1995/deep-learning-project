import torch

from einops import rearrange
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  # def attention(self, key, query, value, attention_mask):

  #   ### YOUR CODE HERE
  #   raise NotImplementedError
  def attention(self, query, key, value, attention_mask):

    B, H, T, d_head = query.shape
    # scaled dot-product attention
    scores = (query @ key.transpose(-1, -2)) / (d_head ** 0.5) # (B, H, T, T)
    # causal mask
    causal_mask = torch.triu(
      torch.ones(T, T, device=query.device, dtype=torch.bool),
      diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float('-inf'))
    # Support both standard mask (B, T) and extended/additive mask (B, 1, 1, T).
    if attention_mask.dim() == 2:
      pad_mask = attention_mask[:, None, None, :].bool()
      scores = scores.masked_fill(~pad_mask, float('-inf'))
    elif attention_mask.dim() == 4:
      scores = scores + attention_mask.to(dtype=scores.dtype, device=scores.device)
    else:
      raise ValueError(f"attention_mask must have 2 or 4 dims, got {attention_mask.dim()}")
    # softmax -> weights
    weights = F.softmax(scores, dim=-1) # (B, H, T, T)
    weights = self.dropout(weights)
    output = weights @ value # (B, H, T, d_head)
    return output

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, seq_len] or [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(query_layer, key_layer, value_layer, attention_mask)
    # Merge heads back to (B, T, hidden_size) for attention_dense + residual in GPT2Layer.
    attn_value = rearrange(attn_value, 'b h t d -> b t (h d)')
    return attn_value
