#!/usr/bin/env python3

'''
LoRA comparison experiments (lora / pissa / loftq x rank 4/8/16 = 9 runs).
See docs/experiment_lora_sentiment.md for the full experiment plan.

Usage
-----
# Smoke test locally with a small subset of data (results written to results_smoke.json)
python3 run_experiment.py --task sentiment --smoke

# Run all configurations on GPU (SST + CFIMDB)
python3 run_experiment.py --task sentiment --use_gpu

# Run only one dataset
python3 run_experiment.py --task sentiment --use_gpu --dataset sst
python3 run_experiment.py --task sentiment --use_gpu --dataset cfimdb

# Run a single configuration
python3 run_experiment.py --task sentiment --use_gpu --dataset sst --init lora --rank 8

# Force rerun (old results.json is backed up as results_backup.json)
python3 run_experiment.py --task sentiment --use_gpu --rerun

Output files
------------
predictions/results.json                                    aggregated results (one entry per run)
predictions/results_smoke.json                              smoke-test results, does not affect main results
predictions/lora/{dataset}_{init}_rank{rank}_dev.csv        per-run dev predictions
predictions/lora/{dataset}_{init}_rank{rank}_test.csv       per-run test predictions (if test set exists)
checkpoints/lora/{dataset}_{init}_rank{rank}.pt             best-epoch weights per run

TODO (see schedule.md)
----------------------
python3 run_experiment.py --task ablation   --use_gpu
python3 run_experiment.py --task sonnet     --use_gpu
python3 run_experiment.py --task paraphrase --use_gpu

'''

import os, json, random, argparse, time, subprocess, sys
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
from modules.lora_linear import apply_lora, init_lora
from paraphrase_detection import ParaphraseGPT, add_arguments as para_add_arguments
from datasets import ParaphraseDetectionDataset, load_paraphrase_data
from evaluation import model_eval_paraphrase

def _try_import(module_name, fn_name):
  try:
    import importlib
    return getattr(importlib.import_module(module_name), fn_name)
  except (ImportError, AttributeError):
    return None

_init_pissa = _try_import('lora_pissa', 'init_pissa')
_init_loftq = _try_import('lora_loftq', 'init_loftq')

INIT_REGISTRY = {'lora': init_lora}
if _init_pissa is not None:
  INIT_REGISTRY['pissa'] = _init_pissa
if _init_loftq is not None:
  INIT_REGISTRY['loftq'] = _init_loftq
from optimizer import AdamW
from utils import get_device

RESULTS_PATH          = 'predictions/results.json'
ABLATION_RESULTS_PATH = 'predictions/results_ablation.json'

ABLATION_GRID = {
  'rank':   [4, 8, 16, 32],
  'target': {
    'QV':  ['query', 'value'],
    'QKV': ['query', 'key', 'value'],
    'ALL': ['query', 'key', 'value', 'attention_dense', 'interm_dense', 'out_dense'],
  },
  'alpha': 16.0,
}

QUORA_CONFIG = {
  'train': 'data/quora-train.csv',
  'dev':   'data/quora-dev.csv',
  'test':  'data/quora-test-student.csv',
}

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

SENTIMENT_DATASETS = {
  'sst': {
    'train': 'data/ids-sst-train.csv',
    'dev':   'data/ids-sst-dev.csv',
    'test':  'data/ids-sst-test.csv',
  },
  'cfimdb': {
    'train': 'data/ids-cfimdb-train.csv',
    'dev':   'data/ids-cfimdb-dev.csv',
    'test':  'data/ids-cfimdb-test.csv',
  },
}

SENTIMENT_TARGET_DEV_ACC = {
  'sst': 0.516,
  'cfimdb': 0.976,
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
      content = f.read().strip()
      if content:
        return json.loads(content)
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


def get_device_name(device):
  if device.type == 'cuda':
    return torch.cuda.get_device_name(0)
  if device.type == 'mps':
    import subprocess
    for key in ['machdep.cpu.brand_string', 'hw.model']:
      try:
        name = subprocess.check_output(['sysctl', '-n', key],
                                       stderr=subprocess.DEVNULL).decode().strip()
        if name:
          return f'Apple MPS ({name})'
      except Exception:
        pass
    return 'Apple MPS'
  import platform, subprocess
  for key in ['machdep.cpu.brand_string', 'hw.model']:
    try:
      name = subprocess.check_output(['sysctl', '-n', key],
                                     stderr=subprocess.DEVNULL).decode().strip()
      if name:
        return f'CPU ({name})'
    except Exception:
      pass
  return f'CPU ({platform.processor() or platform.machine()})'


def save_preds_csv(path, sent_ids, preds, true_labels):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as f:
    f.write('id\tpredicted\ttrue\n')
    for sid, p, t in zip(sent_ids, preds, true_labels):
      f.write(f'{sid}\t{p}\t{t}\n')


def run_sentiment(args):
  global RESULTS_PATH
  if args.smoke:
    RESULTS_PATH = RESULTS_PATH.replace('.json', '_smoke.json')
  device  = get_device(args.use_gpu)
  if args.rerun and os.path.exists(RESULTS_PATH):
    backup = RESULTS_PATH.replace('.json', '_backup.json')
    os.rename(RESULTS_PATH, backup)
    print(f'Old results backed up to {backup}')
  results = load_results()

  batch_size = args.batch_size or FIXED_CONFIG['batch_size']
  init_methods = ([args.init] if args.init else ['lora']) if args.smoke else ([args.init] if args.init else LORA_GRID['init_method'])
  skipped = [m for m in init_methods if m not in INIT_REGISTRY]
  init_methods = [m for m in init_methods if m in INIT_REGISTRY]
  if skipped:
    print(f'[skip] init_method not yet implemented, skipping: {skipped}')
  ranks  = ([args.rank] if args.rank else [4]) if args.smoke else ([args.rank] if args.rank else LORA_GRID['rank'])
  epochs = 10  if args.smoke else (args.epochs or FIXED_CONFIG['epochs'])

  datasets_to_run = [args.dataset] if args.dataset else list(SENTIMENT_DATASETS.keys())

  for dataset_name in datasets_to_run:
    ds_cfg = SENTIMENT_DATASETS[dataset_name]

    train_data, num_labels = load_data(ds_cfg['train'], 'train')
    dev_data               = load_data(ds_cfg['dev'],   'valid')
    test_path = ds_cfg.get('test', '')
    has_test  = bool(test_path) and os.path.exists(test_path) and not args.smoke
    test_data = load_data(test_path, 'valid') if has_test else None

    if args.smoke:
      train_data = train_data[:64]
      dev_data   = dev_data[:32]

    train_dataset = SentimentDataset(train_data, args)
    dev_dataset   = SentimentDataset(dev_data,   args)
    train_loader  = DataLoader(train_dataset, shuffle=True,  batch_size=batch_size,
                               collate_fn=train_dataset.collate_fn)
    dev_loader    = DataLoader(dev_dataset,   shuffle=False, batch_size=batch_size,
                               collate_fn=dev_dataset.collate_fn)
    if has_test:
      test_dataset = SentimentDataset(test_data, args)
      test_loader  = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                                collate_fn=test_dataset.collate_fn)

    for init_method in init_methods:
      for rank in ranks:
        if is_done(results, 'sentiment', dataset_name, init_method, rank) and not args.rerun:
          print(f'\n=== sentiment | {dataset_name} | init={init_method} | rank={rank} — already done, skipping ===')
          continue

        seed_everything(FIXED_CONFIG['seed'])
        t_start = time.time()
        print(f'\n=== sentiment | {dataset_name} | init={init_method} | rank={rank} ===')
        if device.type == 'cuda':
          torch.cuda.reset_peak_memory_stats(device)

        config = SimpleNamespace(
          hidden_dropout_prob=0.1,
          num_labels=num_labels,
          hidden_size=768,
          data_dir='.',
          fine_tune_mode='full-model',
        )
        model = GPT2SentimentClassifier(config)
        # freeze the entire GPT backbone; only the classifier head and LoRA A/B are trainable
        for param in model.gpt.parameters():
          param.requires_grad = False
        model = apply_lora(
          model,
          rank=rank,
          alpha=LORA_GRID['alpha'],
          init_fn=INIT_REGISTRY[init_method],
          target_modules=['query', 'key', 'value'],
        )
        model = model.to(device)

        trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f'  trainable params: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)')

        optimizer = AdamW(
          filter(lambda p: p.requires_grad, model.parameters()),
          lr=FIXED_CONFIG['lr'],
        )

        os.makedirs('checkpoints/lora', exist_ok=True)
        ckpt_path = f'checkpoints/lora/sentiment_{dataset_name}_{init_method}_rank{rank}.pt'

        best_acc, best_loss, best_preds, best_true, best_sent_ids = 0.0, float('inf'), [], [], []
        best_epoch = 0
        steps_to_target = None
        target_dev_acc = SENTIMENT_TARGET_DEV_ACC.get(dataset_name)
        curve = {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}
        for epoch in range(epochs):
          train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, batch_size)
          dev_acc, _, preds, true, _, sent_ids = model_eval(dev_loader, model, device)
          dev_loss = compute_dev_loss(model, dev_loader, device, batch_size)
          curve['train_loss'].append(round(train_loss, 4))
          curve['train_acc'].append(round(train_acc, 4))
          curve['dev_loss'].append(round(dev_loss, 4))
          curve['dev_acc'].append(round(dev_acc, 4))
          if target_dev_acc is not None and steps_to_target is None and dev_acc >= target_dev_acc:
            steps_to_target = (epoch + 1) * len(train_loader)
          if dev_acc > best_acc:
            best_acc, best_loss, best_epoch = dev_acc, train_loss, epoch
            best_preds, best_true, best_sent_ids = preds, true, sent_ids
            if not args.smoke:
              torch.save({'model': model.state_dict(), 'epoch': epoch,
                          'dev_acc': dev_acc, 'init_method': init_method, 'rank': rank}, ckpt_path)
          print(f'  epoch {epoch}: train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  dev_loss={dev_loss:.4f}  dev_acc={dev_acc:.4f}')

        os.makedirs('predictions/lora', exist_ok=True)
        dev_pred_path = f'predictions/lora/sentiment_{dataset_name}_{init_method}_rank{rank}_dev.csv'
        save_preds_csv(dev_pred_path, best_sent_ids, best_preds, best_true)

        # --- test evaluation ---
        test_acc = test_f1 = test_pred_path = None
        if has_test and os.path.exists(ckpt_path):
          ckpt = torch.load(ckpt_path, map_location=device)
          model.load_state_dict(ckpt['model'])
          test_acc_val, _, test_preds, test_true, _, test_sent_ids = model_eval(test_loader, model, device)
          test_pred_path = f'predictions/lora/sentiment_{dataset_name}_{init_method}_rank{rank}_test.csv'
          save_preds_csv(test_pred_path, test_sent_ids, test_preds, test_true)
          test_acc = round(test_acc_val, 4)
          test_f1  = round(f1_score(test_true, test_preds, average='macro'), 4)
          print(f'  test  acc={test_acc:.4f}  f1={test_f1:.4f}  predictions -> {test_pred_path}')

        elapsed    = time.time() - t_start
        best_f1    = round(f1_score(best_true, best_preds, average='macro'), 4)
        device_name = get_device_name(device)
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1) if device.type == 'cuda' else None

        entry = {
          'task':             'sentiment',
          'dataset':          dataset_name,
          'train_size':       len(train_data),
          'dev_size':         len(dev_data),
          'test_size':        len(test_data) if test_data is not None else None,
          'hparams': {
            'init_method':  init_method,
            'rank':         rank,
            'alpha':        LORA_GRID['alpha'],
            'lr':           FIXED_CONFIG['lr'],
            'batch_size':   batch_size,
            'epochs':       epochs,
            'seed':         FIXED_CONFIG['seed'],
            'model':        FIXED_CONFIG['model_size'],
          },
          'init_method':      init_method,
          'rank':             rank,
          'best_epoch':       best_epoch,
          'dev_acc':          round(best_acc,  4),
          'dev_f1':           best_f1,
          'test_acc':         test_acc,
          'test_f1':          test_f1,
          'train_loss':       round(best_loss, 4),
          'trainable_params': trainable,
          'total_params':     total_params,
          'trainable_pct':    round(100 * trainable / total_params, 2),
          'target_dev_acc':   target_dev_acc,
          'steps_to_target_acc': steps_to_target,
          'peak_vram_mb':     peak_vram_mb,
          'elapsed_min':      round(elapsed / 60, 1),
          'device':           device_name,
          'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M'),
          'dev_pred_file':    dev_pred_path,
          'test_pred_file':   test_pred_path,
          'curve':            curve,
        }
        if args.smoke:
          smoke_path = RESULTS_PATH
          smoke_results = []
          if os.path.exists(smoke_path):
            with open(smoke_path) as f:
              content = f.read().strip()
              if content:
                smoke_results = json.loads(content)
          smoke_results.append(entry)
          with open(smoke_path, 'w') as f:
            json.dump(smoke_results, f, indent=2)
          print(f'  smoke results written to {smoke_path}')
        else:
          results.append(entry)
          save_results(results)
        print(f'  => best dev acc: {best_acc:.4f}  predictions -> {dev_pred_path}')


def compute_dev_loss_paraphrase(model, dataloader, device):
  model.eval()
  total_loss, n = 0, 0
  with torch.no_grad():
    for batch in dataloader:
      b_ids      = batch['token_ids'].to(device)
      b_mask     = batch['attention_mask'].to(device)
      answer_ids = batch['answer_token_ids'].to(device)
      logits     = model(b_ids, b_mask)
      total_loss += F.cross_entropy(logits, answer_ids, reduction='mean').item()
      n += 1
  return total_loss / max(n, 1)


def train_one_epoch_paraphrase(model, dataloader, optimizer, device):
  model.train()
  total_loss, correct, total, n = 0, 0, 0, 0
  for batch in tqdm(dataloader, desc='train'):
    b_ids      = batch['token_ids'].to(device)
    b_mask     = batch['attention_mask'].to(device)
    answer_ids = batch['answer_token_ids'].to(device)
    optimizer.zero_grad()
    logits = model(b_ids, b_mask)
    loss   = F.cross_entropy(logits, answer_ids, reduction='mean')
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    correct    += (logits.argmax(dim=-1) == answer_ids).sum().item()
    total      += answer_ids.size(0)
    n          += 1
  return total_loss / max(n, 1), correct / max(total, 1)


def run_ablation(args):
  results_path = (ABLATION_RESULTS_PATH.replace('.json', '_smoke.json')
                  if args.smoke else ABLATION_RESULTS_PATH)

  if args.rerun and os.path.exists(results_path):
    backup = results_path.replace('.json', '_backup.json')
    os.rename(results_path, backup)
    print(f'Old results backed up to {backup}')

  results = []
  if os.path.exists(results_path):
    with open(results_path) as f:
      content = f.read().strip()
      if content:
        results = json.loads(content)

  device     = get_device(args.use_gpu)
  batch_size = args.batch_size or FIXED_CONFIG['batch_size']
  epochs     = 1 if args.smoke else (args.epochs or FIXED_CONFIG['epochs'])

  model_args = SimpleNamespace(model_size=FIXED_CONFIG['model_size'])
  model_args = para_add_arguments(model_args)

  train_data = load_paraphrase_data(QUORA_CONFIG['train'])
  dev_data   = load_paraphrase_data(QUORA_CONFIG['dev'])
  if args.smoke:
    train_data = train_data[:64]
    dev_data   = dev_data[:32]

  train_ds     = ParaphraseDetectionDataset(train_data, model_args)
  dev_ds       = ParaphraseDetectionDataset(dev_data,   model_args)
  train_loader = DataLoader(train_ds, shuffle=True,  batch_size=batch_size, collate_fn=train_ds.collate_fn)
  dev_loader   = DataLoader(dev_ds,   shuffle=False, batch_size=batch_size, collate_fn=dev_ds.collate_fn)

  ranks   = [4] if args.smoke else ABLATION_GRID['rank']
  targets = {'QV': ABLATION_GRID['target']['QV']} if args.smoke else ABLATION_GRID['target']

  for target_key, target_modules in targets.items():
    for rank in ranks:
      already = any(r.get('target') == target_key and r.get('rank') == rank for r in results)
      if already and not args.rerun:
        print(f'\n=== ablation | target={target_key} | rank={rank} — already done, skipping ===')
        continue

      print(f'\n=== ablation | target={target_key} | rank={rank} ===')
      seed_everything(FIXED_CONFIG['seed'])
      if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
      t_start = time.time()

      model = ParaphraseGPT(model_args)
      for param in model.gpt.parameters():
        param.requires_grad = False
      model = apply_lora(model, rank=rank, alpha=ABLATION_GRID['alpha'],
                         init_fn=init_lora, target_modules=target_modules)
      model = model.to(device)

      trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
      total_params = sum(p.numel() for p in model.parameters())
      print(f'  trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)')

      optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=FIXED_CONFIG['lr'])

      best_acc, best_epoch = 0.0, 0
      best_preds, best_true, best_sent_ids = [], [], []
      curve = {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}

      for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_paraphrase(model, train_loader, optimizer, device)
        dev_acc, _, preds, true, sent_ids = model_eval_paraphrase(dev_loader, model, device)
        dev_loss = compute_dev_loss_paraphrase(model, dev_loader, device)
        curve['train_loss'].append(round(train_loss, 4))
        curve['train_acc'].append(round(train_acc,   4))
        curve['dev_loss'].append(round(dev_loss,     4))
        curve['dev_acc'].append(round(dev_acc,       4))
        if dev_acc > best_acc:
          best_acc, best_epoch = dev_acc, epoch
          best_preds, best_true, best_sent_ids = preds, true, sent_ids
        print(f'  epoch {epoch}: train_loss={train_loss:.4f}  train_acc={train_acc:.4f}'
              f'  dev_loss={dev_loss:.4f}  dev_acc={dev_acc:.4f}')

      if args.smoke:
        assert not any(v != v for v in curve['train_loss']), 'NaN in train_loss'
        assert 0.0 <= best_acc <= 1.0, f'dev_acc out of range: {best_acc}'
        print('  [smoke] sanity assertions passed ✓')

      elapsed   = time.time() - t_start
      peak_vram = (round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)
                   if device.type == 'cuda' else None)

      pred_dir = 'predictions/ablation'
      os.makedirs(pred_dir, exist_ok=True)
      tag          = f'quora_lora_{target_key}_rank{rank}'
      dev_pred_path = f'{pred_dir}/{tag}_dev.csv'
      save_preds_csv(dev_pred_path, best_sent_ids, best_preds, best_true)

      entry = {
        'task':           'ablation',
        'dataset':        'quora',
        'target':         target_key,
        'target_modules': target_modules,
        'rank':           rank,
        'alpha':          ABLATION_GRID['alpha'],
        'init_method':    'lora',
        'hparams': {
          'seed': FIXED_CONFIG['seed'], 'epochs': epochs, 'lr': FIXED_CONFIG['lr'],
          'batch_size': batch_size, 'model': FIXED_CONFIG['model_size'],
        },
        'train_size':       len(train_data),
        'dev_size':         len(dev_data),
        'best_epoch':       best_epoch,
        'dev_acc':          round(best_acc, 4),
        'dev_f1':           round(f1_score(best_true, best_preds, average='macro'), 4),
        'train_loss':       round(curve['train_loss'][best_epoch], 4),
        'trainable_params': trainable,
        'total_params':     total_params,
        'trainable_pct':    round(100 * trainable / total_params, 2),
        'peak_vram_mb':     peak_vram,
        'elapsed_min':      round(elapsed / 60, 1),
        'device':           get_device_name(device),
        'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M'),
        'dev_pred_file':    dev_pred_path,
        'curve':            curve,
      }
      results.append(entry)
      os.makedirs(os.path.dirname(results_path), exist_ok=True)
      with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
      print(f'  => best dev acc: {best_acc:.4f}  → {dev_pred_path}')

  print('\n' + '=' * 60)
  print('ABLATION SUMMARY (Quora paraphrase)')
  for e in sorted(results, key=lambda x: -x['dev_acc']):
    print(f'  target={e["target"]:<5} rank={e["rank"]:<4} '
          f'dev_acc={e["dev_acc"]:.4f}  trainable={e["trainable_pct"]}%')
  if results:
    best_val  = max(e['dev_acc'] for e in results)
    threshold = best_val * 0.95
    cands     = [e for e in results if e['dev_acc'] >= threshold]
    opt       = min(cands, key=lambda x: x['trainable_pct'])
    print(f'\nConfig_opt: rank={opt["rank"]}  target={opt["target"]}  '
          f'dev_acc={opt["dev_acc"]:.4f}  trainable={opt["trainable_pct"]}%')
  print('=' * 60)


def run_exp3_quant(args):
  ext_script = os.path.join('extensions', 'exp3_quant_lora', 'run_exp3.py')
  if not os.path.exists(ext_script):
    raise FileNotFoundError(
      f'Exp3 extension script not found: {ext_script}. '
      'Please ensure extensions/exp3_quant_lora is integrated into this project.'
    )

  if args.dataset is not None and args.dataset != 'sst':
    raise ValueError('exp3_quant currently supports dataset=sst only.')

  cmd = [sys.executable, ext_script]
  if args.use_gpu:
    cmd.append('--use_gpu')
  if args.smoke:
    cmd.append('--smoke')
  if args.rerun:
    cmd.append('--rerun')
  if args.dataset is not None:
    cmd.extend(['--dataset', args.dataset])
  if args.rank is not None:
    cmd.extend(['--rank', str(args.rank)])
  if args.epochs is not None:
    cmd.extend(['--epochs', str(args.epochs)])
  if args.batch_size is not None:
    cmd.extend(['--batch_size', str(args.batch_size)])
  if args.exp3_method is not None:
    cmd.extend(['--method', args.exp3_method])
  if args.exp3_bits is not None:
    cmd.extend(['--bits', str(args.exp3_bits)])
  if args.exp3_alpha is not None:
    cmd.extend(['--alpha', str(args.exp3_alpha)])
  if args.exp3_baseline_dev_acc is not None:
    cmd.extend(['--baseline-dev-acc', str(args.exp3_baseline_dev_acc)])
  if args.dlp_root is not None:
    cmd.extend(['--dlp-root', args.dlp_root])

  print(f'Running Exp3 quantization via extension: {" ".join(cmd)}')
  subprocess.run(cmd, check=True)


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', choices=['sentiment', 'ablation', 'sonnet', 'paraphrase', 'exp3_quant'],
                      default='sentiment')
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--smoke', action='store_true',
                      help='smoke test: run with a small data subset to verify the pipeline')
  parser.add_argument('--rerun', action='store_true',
                      help='ignore existing results and rerun all configurations')
  parser.add_argument('--dataset', choices=list(SENTIMENT_DATASETS.keys()), default=None,
                      help='run only the specified dataset (default: all)')
  parser.add_argument('--init', choices=['lora', 'pissa', 'loftq'], default=None,
                      help='run only the specified init_method')
  parser.add_argument('--rank', type=int, choices=[4, 8, 16, 32], default=None,
                      help='run only the specified rank')
  parser.add_argument('--epochs', type=int, default=None,
                      help=f'number of training epochs (default: {FIXED_CONFIG["epochs"]}; ignored in --smoke mode)')
  parser.add_argument('--batch_size', type=int, default=None,
                      help=f'training batch size (default: {FIXED_CONFIG["batch_size"]})')
  parser.add_argument('--exp3_method', choices=['qlora', 'loftq'], default=None,
                      help='exp3_quant only: run only the specified quantization method')
  parser.add_argument('--exp3_bits', type=int, choices=[2, 4], default=None,
                      help='exp3_quant only: run only the specified quantization bit width')
  parser.add_argument('--exp3_alpha', type=float, default=None,
                      help='exp3_quant only: override LoRA alpha')
  parser.add_argument('--exp3_baseline_dev_acc', type=float, default=None,
                      help='exp3_quant only: override baseline dev accuracy for degradation')
  parser.add_argument('--dlp_root', type=str, default=None,
                      help='exp3_quant only: override project root passed to extension')
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  if args.task == 'sentiment':
    run_sentiment(args)
  elif args.task == 'exp3_quant':
    run_exp3_quant(args)
  elif args.task == 'ablation':
    run_ablation(args)
  elif args.task in ('sonnet', 'paraphrase'):
    print(f'Task {args.task} not yet implemented — see schedule.md for owners.')
  
