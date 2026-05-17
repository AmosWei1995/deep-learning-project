'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

可选：若仍希望每个 epoch 在终端打印 held-out 续写示例（或提交阶段打印解码全文），追加
`--print_generated_sonnets`；默认关闭以减轻刷屏。

每个 epoch 会打印 train / dev 的 loss。若追加 `--compute_chrf`，再计算并打印 train / dev 的 chrF++
（此时依赖 `--dev_full_sonnet_path` 与 `--dev_prompt_sonnet_path`）。

训练结束后保存最终权重到 `{epochs}-{lr}-sonnet.pt`，并写入 dev 续写（`--dev_sonnet_out`）与提交用 held-out 文件。
若需要每个 epoch 各存一份 checkpoint，追加 `--save_every_epoch`（文件名为 `{epoch}_{epochs}-{lr}-sonnet.pt`）。

trains your SonnetGPT model and writes the required submission files.

常用命令
--------
# 正常训练，跑完自动写预测文件
python sonnet_generation.py --use_gpu

# 不重新训练，只用已有 checkpoint 算 dev loss 和 dev chrF++
python sonnet_generation.py --eval_only --filepath 10-1e-05-sonnet.pt --use_gpu

# 训练时同步算 chrF（慢，每个 epoch 要对全量数据生成一遍）
python sonnet_generation.py --use_gpu --compute_chrf

# 每个 epoch 单独存一份 checkpoint（默认只存最后一轮）
python sonnet_generation.py --use_gpu --save_every_epoch

# 在终端打印续写结果（默认关闭）
python sonnet_generation.py --use_gpu --print_generated_sonnets

输出文件
--------
{epochs}-{lr}-sonnet.pt              训练完的模型权重
predictions/generated_sonnets.txt    held-out 集的续写结果（提交用）
predictions/generated_sonnets_dev.txt dev 集的续写结果（自测用）
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
  """
  改进目的：为「训练集 chrF」构造与 held-out 风格一致的前缀（前 k 个非空文本行），
  使生成评测与作业提供的 dev 前缀格式对齐，便于同一套 chrF 逻辑复用。
  """
  lines = [ln for ln in full_sonnet.splitlines() if ln.strip()]
  if len(lines) <= k:
    return full_sonnet.strip()
  return '\n'.join(lines[:k]).strip()


@torch.no_grad()
def compute_average_lm_loss(model, dataloader, device, desc: str = 'eval'):
  """
  改进目的：在任意 DataLoader 上计算与训练相同的 next-token 交叉熵均值（不反传），
  用于 dev / 校验集上的 loss，弥补原先只有 train loss 的监控盲区。
  """
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
  """
  改进目的：在「前缀 + 模型续写」与「参考全文」之间计算语料级 chrF（与 evaluation.test_sonnet 同用 sacrebleu），
  分别用于训练监控与 dev 监控，避免只有 loss 而看不到生成质量。
  """
  assert len(prefixes) == len(references), 'prefixes 与 references 必须一一对应、长度一致'
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

  # 改进目的：在「带完整参考答案」的 dev 句子上计算 dev loss（与训练相同的 next-token CE），
  # 与 train loss 并列打印，便于观察过拟合与泛化。
  dev_full_dataset = SonnetsDataset(args.dev_full_sonnet_path)
  dev_dataloader = DataLoader(
    dev_full_dataset,
    shuffle=False,
    batch_size=args.batch_size,
    collate_fn=dev_full_dataset.collate_fn,
  )

  # 改进目的：dev 前缀与 dev 金标条数一致时才可做 dev loss / chrF / 保存 dev 生成。
  dev_prompt_dataset = SonnetsDataset(args.dev_prompt_sonnet_path)
  assert len(dev_prompt_dataset) == len(dev_full_dataset), (
    'dev 前缀集与 dev 金标条数不一致，请检查 dev_prompt_sonnet_path 与 dev_full_sonnet_path。'
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

    # 改进目的：每个 epoch 结束除 train loss 外，同时汇报 dev loss 与 train/dev 的语料级 chrF，
    # 形成与「仅看 train loss + 终端刷生成句」相比更完整的训练曲线与质量信号。
    dev_loss = compute_average_lm_loss(model, dev_dataloader, device, desc=f'dev-loss-{epoch}')

    # 改进目的：chrF 需逐条生成，默认关闭以加快训练；需要监控生成质量时再加 --compute_chrf。
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

    # 以下为原始「仅 train loss」的终端输出；已由上一行汇总指标替代，故整段注释保留，便于对照与回滚实验。
    # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    if args.compute_chrf:
      print(
        f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev loss :: {dev_loss :.3f}, "
        f"train chrF++ :: {train_chrf :.3f}, dev chrF++ :: {dev_chrf :.3f}."
      )
    else:
      print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev loss :: {dev_loss :.3f}.")

    # 改进目的：原先每个 epoch 都在终端打印大量续写结果，信息噪声大；默认关闭，仅当用户传入
    # --print_generated_sonnets 时才打印，需要肉眼检查生成时再打开。
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
    # 默认不在每轮写盘；若传入 --save_every_epoch，则每轮额外保存 f'{epoch}_{args.filepath}'（与最终 args.filepath 并存）。
    if args.save_every_epoch:
      save_model(model, optimizer, args, f'{epoch}_{args.filepath}')

  model.eval()
  save_model(model, optimizer, args, args.filepath)
  write_dev_generated_sonnets(model, args, device, dev_prompt_dataset)


@torch.no_grad()
def write_dev_generated_sonnets(model, args, device, dev_prompt_dataset):
  """
  在 dev 前缀集上续写并写入磁盘，格式与 generate_submission_sonnets 一致，便于与 TRUE_sonnets_held_out_dev 对照。
  """
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

    # 改进目的：与训练阶段一致，提交脚本默认不在终端刷屏；需要查看时再传 --print_generated_sonnets。
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
  """
  加载已有 checkpoint，只计算 dev loss 和 dev chrF++，不重新训练。
  用法：python sonnet_generation.py --eval_only [--filepath 10-1e-05-sonnet.pt] --use_gpu
  """
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
    'dev 前缀集与 dev 金标条数不一致'

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
    help="训练结束后将 dev 前缀续写写入该路径（与 sonnet_out 并列）。",
  )

  # 改进目的：dev loss / dev chrF 使用的数据路径与作业仓库中的 dev 金标、dev 前缀文件对应。
  parser.add_argument(
    "--dev_full_sonnet_path",
    type=str,
    default="data/TRUE_sonnets_held_out_dev.txt",
    help="完整 dev 金标（用于 dev loss；若使用 --compute_chrf 则亦用于 dev chrF 的 reference）。",
  )
  parser.add_argument(
    "--dev_prompt_sonnet_path",
    type=str,
    default="data/sonnets_held_out_dev.txt",
    help="dev 前缀集（仅前几行；用于 dev chrF、训练结束时的 dev 生成文件；与 dev_full 按顺序对齐）。",
  )

  # 改进目的：chrF 依赖大量生成，默认不算；需要 train/dev chrF 监控时显式打开。
  parser.add_argument(
    "--compute_chrf",
    action="store_true",
    help="每个 epoch 计算并打印 train / dev 的 chrF++（会显著增加耗时）。",
  )

  # 改进目的：默认不在终端打印每个 epoch / 提交阶段的生成全文；显式传入本开关时再打印，减轻视觉干扰。
  parser.add_argument(
    "--print_generated_sonnets",
    action="store_true",
    help="在终端打印训练每个 epoch 的 held-out 续写示例，以及 generate_submission_sonnets 时的解码结果。",
  )

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # 默认仅训练结束保存最终 pt；打开后每个 epoch 额外保存 `{epoch}_{filepath}`，便于挑轮次或断点续训。
  parser.add_argument(
    "--save_every_epoch",
    action="store_true",
    help="每个 epoch 结束后额外保存 checkpoint（文件名前缀为 epoch 序号）；最终仍会保存到主 filepath。",
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
    help="跳过训练，直接加载 --filepath 指定的 checkpoint 计算 dev loss 和 dev chrF++。",
  )
  parser.add_argument(
    "--filepath",
    type=str,
    default=None,
    help="--eval_only 时指定 checkpoint 路径；训练模式下自动从 epochs/lr 推导。",
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