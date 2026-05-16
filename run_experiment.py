#!/usr/bin/env python3

'''
跑 LoRA 对比实验（lora / pissa / loftq × rank 4/8/16 = 9组）。
实验方案见 docs/experiment_lora_sentiment.md。

常用命令
--------
# 先在本地跑一下确认代码没问题（只用少量数据，结果写 results_smoke.json）
python3 run_experiment.py --task sentiment --smoke

# GPU 上跑全部 9 组
python3 run_experiment.py --task sentiment --use_gpu

# 只跑某一组
python3 run_experiment.py --task sentiment --use_gpu --init pissa --rank 8

# 重跑（旧 results.json 自动备份成 results_backup.json）
python3 run_experiment.py --task sentiment --use_gpu --rerun

输出文件
--------
predictions/results.json                          汇总结果（每组一条）
predictions/results_smoke.json                    smoke 测试结果，不影响正式结果
predictions/lora/sentiment_{init}_rank{rank}_dev.csv  每组的 dev 预测
checkpoints/lora/sentiment_{init}_rank{rank}.pt   每组最佳 epoch 的权重

待实现（见 schedule.md）
--------
python3 run_experiment.py --task ablation   --use_gpu  
python3 run_experiment.py --task sonnet     --use_gpu  
python3 run_experiment.py --task paraphrase --use_gpu  

'''

import os, json, random, argparse, time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from types import SimpleNamespace
from tqdm import tqdm, trange
from sklearn.metrics import f1_score

from classifier import (
  GPT2SentimentClassifier, SentimentDataset,
  load_data, model_eval,
)
from modules.lora_linear import apply_lora
from optimizer import AdamW
from utils import get_device

RESULTS_PATH = 'predictions/results.json'

FIXED_CONFIG = {
  'seed':       11711,
  'epochs':     10,
  'lr':         1e-5,
  'batch_size': 8,
  'model_size': 'gpt2',
}

LORA_GRID = {
  'init_method': ['lora', 'pissa', 'loftq'],
  'rank':        [4, 8, 16],
  'alpha':       16.0,
}


def seed_everything(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
  if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)


def is_done(results, task, dataset, init_method, rank):
  return any(
    r['task'] == task and r['dataset'] == dataset
    and r['init_method'] == init_method and r['rank'] == rank
    for r in results
  )


def compute_dev_loss(model, dataloader, device, batch_size):
  model.eval()
  total_loss, n = 0, 0
  with torch.no_grad():
    for batch in dataloader:
      b_ids    = batch['token_ids'].to(device)
      b_mask   = batch['attention_mask'].to(device)
      b_labels = batch['labels'].to(device)
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / batch_size
      total_loss += loss.item()
      n += 1
  return total_loss / n


def load_results():
  if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
      return json.load(f)
  return []


def save_results(results):
  os.makedirs('predictions', exist_ok=True)
  with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)


def train_one_epoch(model, dataloader, optimizer, device, batch_size):
  model.train()
  total_loss, correct, total, n = 0, 0, 0, 0
  pbar = tqdm(dataloader, desc='train')
  for batch in pbar:
    b_ids    = batch['token_ids'].to(device)
    b_mask   = batch['attention_mask'].to(device)
    b_labels = batch['labels'].to(device)

    optimizer.zero_grad()
    logits = model(b_ids, b_mask)
    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / batch_size
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    correct += (logits.argmax(dim=-1) == b_labels.view(-1)).sum().item()
    total += b_labels.size(0)
    n += 1

  return total_loss / n, correct / total


def run_sentiment(args):
  device = get_device(args.use_gpu)
  if args.rerun and os.path.exists(RESULTS_PATH):
    backup = RESULTS_PATH.replace('.json', '_backup.json')
    os.rename(RESULTS_PATH, backup)
    print(f'旧结果已备份至 {backup}')
  results = load_results()

  train_data, num_labels = load_data('data/ids-sst-train.csv', 'train')
  dev_data = load_data('data/ids-sst-dev.csv', 'valid')

  if args.smoke:
    train_data = train_data[:64]
    dev_data   = dev_data[:32]

  train_dataset = SentimentDataset(train_data, args)
  dev_dataset   = SentimentDataset(dev_data,   args)
  train_loader  = DataLoader(train_dataset, shuffle=True,  batch_size=FIXED_CONFIG['batch_size'],
                             collate_fn=train_dataset.collate_fn)
  dev_loader    = DataLoader(dev_dataset,   shuffle=False, batch_size=FIXED_CONFIG['batch_size'],
                             collate_fn=dev_dataset.collate_fn)

  init_methods = ['lora'] if args.smoke else ([args.init] if args.init else LORA_GRID['init_method'])
  ranks        = [4]     if args.smoke else ([args.rank] if args.rank else LORA_GRID['rank'])
  epochs       = 10      if args.smoke else FIXED_CONFIG['epochs']

  for init_method in init_methods:
    for rank in ranks:
      if is_done(results, 'sentiment', 'sst', init_method, rank) and not args.rerun:
        print(f'\n=== sentiment | init={init_method} | rank={rank} — 已完成，跳过 ===')
        continue

      seed_everything(FIXED_CONFIG['seed'])
      t_start = time.time()
      print(f'\n=== sentiment | init={init_method} | rank={rank} ===')

      config = SimpleNamespace(
        hidden_dropout_prob=0.1,
        num_labels=num_labels,
        hidden_size=768,
        data_dir='.',
        fine_tune_mode='full-model',
      )
      model = GPT2SentimentClassifier(config)
      # 冻结整个 GPT，只保留分类头和后续 LoRA 的 A、B 可训练
      for param in model.gpt.parameters():
        param.requires_grad = False
      model = apply_lora(
        model,
        rank=rank,
        alpha=LORA_GRID['alpha'],
        init_method=init_method,
        target_modules=['query', 'key', 'value'],
      )
      model = model.to(device)

      trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
      total_params = sum(p.numel() for p in model.parameters())
      print(f'  可训练参数: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)')

      optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FIXED_CONFIG['lr'],
      )

      os.makedirs('checkpoints/lora', exist_ok=True)
      ckpt_path = f'checkpoints/lora/sentiment_{init_method}_rank{rank}.pt'

      best_acc, best_loss, best_preds, best_true, best_sent_ids = 0.0, float('inf'), [], [], []
      curve = {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}
      for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, FIXED_CONFIG['batch_size'])
        dev_acc, _, preds, true, _, sent_ids = model_eval(dev_loader, model, device)
        dev_loss = compute_dev_loss(model, dev_loader, device, FIXED_CONFIG['batch_size'])
        curve['train_loss'].append(round(train_loss, 4))
        curve['train_acc'].append(round(train_acc, 4))
        curve['dev_loss'].append(round(dev_loss, 4))
        curve['dev_acc'].append(round(dev_acc, 4))
        if dev_acc > best_acc:
          best_acc, best_loss = dev_acc, train_loss
          best_preds, best_true, best_sent_ids = preds, true, sent_ids
          if not args.smoke:
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'dev_acc': dev_acc, 'init_method': init_method, 'rank': rank}, ckpt_path)
        print(f'  epoch {epoch}: train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  dev_loss={dev_loss:.4f}  dev_acc={dev_acc:.4f}')

      pred_path = f'predictions/lora/sentiment_{init_method}_rank{rank}_dev.csv'
      os.makedirs('predictions/lora', exist_ok=True)
      with open(pred_path, 'w') as f:
        f.write('id\tpredicted\ttrue\n')
        for sid, p, t in zip(best_sent_ids, best_preds, best_true):
          f.write(f'{sid}\t{p}\t{t}\n')

      elapsed = time.time() - t_start
      if device.type == 'cuda':
        device_name = torch.cuda.get_device_name(0)
      elif device.type == 'mps':
        import subprocess
        try:
          chip = subprocess.check_output(
            ['sysctl', '-n', 'machdep.cpu.brand_string'], stderr=subprocess.DEVNULL
          ).decode().strip()
          if not chip:
            chip = subprocess.check_output(
              ['sysctl', '-n', 'hw.model'], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
          chip = 'Apple Silicon'
        device_name = f'Apple MPS ({chip})'
      else:
        import platform, subprocess
        cpu = ''
        for sysctl_key in ['machdep.cpu.brand_string', 'hw.model']:
          try:
            cpu = subprocess.check_output(
              ['sysctl', '-n', sysctl_key], stderr=subprocess.DEVNULL
            ).decode().strip()
            if cpu:
              break
          except Exception:
            pass
        if not cpu:
          cpu = platform.processor() or platform.machine()
        device_name = f'CPU ({cpu})'

      best_f1 = round(f1_score(best_true, best_preds, average='macro'), 4)

      entry = {
        'task':             'sentiment',
        'dataset':          'sst',
        'train_size':       len(train_data),
        'dev_size':         len(dev_data),
        'init_method':      init_method,
        'rank':             rank,
        'dev_acc':          round(best_acc,  4),
        'dev_f1':           best_f1,
        'train_loss':       round(best_loss, 4),
        'trainable_params': trainable,
        'total_params':     total_params,
        'trainable_pct':    round(100 * trainable / total_params, 2),
        'elapsed_min':      round(elapsed / 60, 1),
        'device':           device_name,
        'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M'),
        'pred_file':        pred_path,
        'curve':            curve,
      }
      if args.smoke:
        smoke_path = RESULTS_PATH.replace('.json', '_smoke.json')
        smoke_results = []
        if os.path.exists(smoke_path):
          with open(smoke_path) as f:
            content = f.read().strip()
            if content:
              smoke_results = json.loads(content)
        smoke_results.append(entry)
        with open(smoke_path, 'w') as f:
          json.dump(smoke_results, f, indent=2)
        print(f'  smoke 结果已写入 {smoke_path}')
      else:
        results.append(entry)
        save_results(results)
      print(f'  => best dev acc: {best_acc:.4f}  predictions -> {pred_path}')


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', choices=['sentiment', 'ablation', 'sonnet', 'paraphrase'],
                      default='sentiment')
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--smoke', action='store_true',
                      help='快速冒烟测试：只用少量数据跑 10 epoch，验证代码流程')
  parser.add_argument('--rerun', action='store_true',
                      help='忽略已有结果，强制重跑所有组')
  parser.add_argument('--init', choices=['lora', 'pissa', 'loftq'], default=None,
                      help='只跑指定的 init_method')
  parser.add_argument('--rank', type=int, choices=[4, 8, 16], default=None,
                      help='只跑指定的 rank')
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  if args.task == 'sentiment':
    run_sentiment(args)
  elif args.task in ('ablation', 'sonnet', 'paraphrase'):
    print(f'Task {args.task} not yet implemented — see schedule.md for owners.')
  
