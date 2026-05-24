'''
Sonnet generation: train SonnetGPT.

  python sonnet_generation.py --use_gpu

Each epoch logs train/dev loss. Optional flags:
  --compute_chrf              train/dev chrF++ (needs dev full + prompt paths)
  --save_every_epoch          checkpoint per epoch ({epoch}_{epochs}-{lr}-sonnet.pt)
  --print_generated_sonnets   print continuations each epoch (off by default)

After training: saves `{epochs}-{lr}-sonnet.pt`, dev continuations, and held-out predictions.


Outputs
-------
{epochs}-{lr}-sonnet.pt                 final weights
predictions/generated_sonnets.txt       held-out (submission)
predictions/generated_sonnets_dev.txt   dev set (evaluation)
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange
from sacrebleu.metrics import CHRF

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW
from utils import get_device

TQDM_DISABLE = False


def _prefix_first_k_non_empty_lines(full_sonnet: str, k: int = 3) -> str:
  """Extract the first k non-empty lines as a prefix, matching the held-out prompt format."""
  lines = [ln for ln in full_sonnet.splitlines() if ln.strip()]
  if len(lines) <= k:
    return full_sonnet.strip()
  return '\n'.join(lines[:k]).strip()


@torch.no_grad()
def compute_average_lm_loss(model, dataloader, device, desc: str = 'eval'):
  """Compute mean next-token cross-entropy over a dataloader (no backprop)."""
  model.eval()
  total_loss = 0.0
  num_batches = 0
  for batch in tqdm(dataloader, desc=desc, disable=TQDM_DISABLE):
    b_ids, b_mask = batch['token_ids'], batch['attention_mask']
    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    logits = model(b_ids, b_mask)
    logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
    labels = b_ids[:, 1:].contiguous().flatten()
    loss = F.cross_entropy(logits, labels, ignore_index=model.tokenizer.pad_token_id, reduction='mean')
    total_loss += loss.item()
    num_batches += 1
  return total_loss / num_batches if num_batches else float('nan')


@torch.no_grad()
def compute_corpus_chrf_with_prefixes(model, prefixes, references, device, args, desc: str = 'chrf'):

  assert len(prefixes) == len(references), 'prefixes and references must have the same length'
  model.eval()
  chrf = CHRF()
  hypotheses = []
  for prefix in tqdm(prefixes, desc=desc, disable=TQDM_DISABLE):
    encoding = model.tokenizer(prefix, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'].to(device)
    _, gen_suffix = model.generate(input_ids, temperature=args.temperature, top_p=args.top_p)
    hypotheses.append(gen_suffix)
  return float(chrf.corpus_score(hypotheses, [references]).score)


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
  if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # By default, fine-tune the full model. TODO: this is maybe not idea.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    output = self.gpt(input_ids, attention_mask)
    hidden = output['last_hidden_state']
    logits = self.gpt.hidden_state_to_token(hidden)
    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())


    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = get_device(args.use_gpu)
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  dev_full_dataset = SonnetsDataset(args.dev_full_sonnet_path)
  dev_dataloader = DataLoader(
    dev_full_dataset,
    shuffle=False,
    batch_size=args.batch_size,
    collate_fn=dev_full_dataset.collate_fn,
  )

  dev_prompt_dataset = SonnetsDataset(args.dev_prompt_sonnet_path)
  assert len(dev_prompt_dataset) == len(dev_full_dataset), (
    'dev prompt and reference sets must have the same length'
  )

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, ignore_index=model.tokenizer.pad_token_id, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_loss = compute_average_lm_loss(model, dev_dataloader, device, desc=f'dev-loss-{epoch}')

    if args.compute_chrf:
      train_prefixes = [_prefix_first_k_non_empty_lines(sonnet_dataset[i][1], k=3) for i in range(len(sonnet_dataset))]
      train_refs = [sonnet_dataset[i][1] for i in range(len(sonnet_dataset))]
      train_chrf = compute_corpus_chrf_with_prefixes(
        model, train_prefixes, train_refs, device, args, desc=f'train-chrf-{epoch}'
      )
      dev_prefixes = [dev_prompt_dataset[i][1] for i in range(len(dev_prompt_dataset))]
      dev_refs = [dev_full_dataset[i][1] for i in range(len(dev_full_dataset))]
      dev_chrf = compute_corpus_chrf_with_prefixes(
        model, dev_prefixes, dev_refs, device, args, desc=f'dev-chrf-{epoch}'
      )

    # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    if args.compute_chrf:
      print(
        f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev loss :: {dev_loss :.3f}, "
        f"train chrF++ :: {train_chrf :.3f}, dev chrF++ :: {dev_chrf :.3f}."
      )
    else:
      print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev loss :: {dev_loss :.3f}.")

    # print('Generating several output sonnets...')
    # model.eval()
    # for batch in held_out_sonnet_dataset:
    #   encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
    #   output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
    #   print(f'{batch[1]}{output[1]}\n\n')
    if args.print_generated_sonnets:
      print('Generating several output sonnets...')
      model.eval()
      for batch in held_out_sonnet_dataset:
        encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
        output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
        print(f'{batch[1]}{output[1]}\n\n')

    # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets.
    if args.save_every_epoch:
      save_model(model, optimizer, args, f'{epoch}_{args.filepath}')

  model.eval()
  save_model(model, optimizer, args, args.filepath)
  write_dev_generated_sonnets(model, args, device, dev_prompt_dataset)


@torch.no_grad()
def write_dev_generated_sonnets(model, args, device, dev_prompt_dataset):
  """Generate continuations for dev prompts and save to disk."""
  model.eval()
  lines = []
  for batch in tqdm(dev_prompt_dataset, desc='save-dev-gen', disable=TQDM_DISABLE):
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    lines.append((sonnet_id, full_sonnet))
    if args.print_generated_sonnets:
      print(f'{decoded_output}\n\n')

  with open(args.dev_sonnet_out, 'w+', encoding='utf-8') as f:
    f.write('--Generated Sonnets (dev)-- \n\n')
    for sonnet_id, text in lines:
      f.write(f'\n{sonnet_id}\n')
      f.write(text)
  print(f"save dev generations to {args.dev_sonnet_out}")


@torch.no_grad()
def generate_submission_sonnets(args):
  device = get_device(args.use_gpu)
  saved = torch.load(args.filepath, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    # print(f'{decoded_output}\n\n')
    if args.print_generated_sonnets:
      print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


@torch.no_grad()
def eval_only(args):
  """Load a checkpoint and compute dev loss and dev chrF++ without retraining."""
  device = get_device(args.use_gpu)
  saved = torch.load(args.filepath, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded checkpoint: {args.filepath}")

  dev_full_dataset = SonnetsDataset(args.dev_full_sonnet_path)
  dev_prompt_dataset = SonnetsDataset(args.dev_prompt_sonnet_path)
  assert len(dev_prompt_dataset) == len(dev_full_dataset), \
    'dev prompt and reference sets must have the same length'

  dev_dataloader = DataLoader(
    dev_full_dataset,
    shuffle=False,
    batch_size=args.batch_size,
    collate_fn=dev_full_dataset.collate_fn,
  )

  dev_loss = compute_average_lm_loss(model, dev_dataloader, device, desc='dev-loss')

  dev_prefixes = [dev_prompt_dataset[i][1] for i in range(len(dev_prompt_dataset))]
  dev_refs = [dev_full_dataset[i][1] for i in range(len(dev_full_dataset))]
  dev_chrf = compute_corpus_chrf_with_prefixes(
    model, dev_prefixes, dev_refs, device, args, desc='dev-chrf'
  )

  print(f"\n=== Eval results for {args.filepath} ===")
  print(f"  dev loss  : {dev_loss:.4f}")
  print(f"  dev chrF++: {dev_chrf:.4f}")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
  parser.add_argument(
    "--dev_sonnet_out",
    type=str,
    default="predictions/generated_sonnets_dev.txt",
    help="Output path for dev continuations.",
  )

  parser.add_argument(
    "--dev_full_sonnet_path",
    type=str,
    default="data/TRUE_sonnets_held_out_dev.txt",
    help="Full dev references (for dev loss and chrF).",
  )
  parser.add_argument(
    "--dev_prompt_sonnet_path",
    type=str,
    default="data/sonnets_held_out_dev.txt",
    help="Dev prompts (first few lines); used for dev chrF and dev generation output.",
  )

  parser.add_argument(
    "--compute_chrf",
    action="store_true",
    help="Compute train/dev chrF++ each epoch (slow).",
  )

  parser.add_argument(
    "--print_generated_sonnets",
    action="store_true",
    help="Print generated sonnet continuations to stdout.",
  )

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument(
    "--save_every_epoch",
    action="store_true",
    help="Save a checkpoint after each epoch (prefixed with epoch index).",
  )

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  parser.add_argument(
    "--eval_only",
    action="store_true",
    help="Skip training; load checkpoint and evaluate dev loss and chrF++.",
  )
  parser.add_argument(
    "--filepath",
    type=str,
    default=None,
    help="Checkpoint path (auto-derived from epochs/lr if not set).",
  )

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  if args.filepath is None:
    args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'
  seed_everything(args.seed)
  if args.eval_only:
    eval_only(args)
  else:
    train(args)
    generate_submission_sonnets(args)