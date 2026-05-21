#!/usr/bin/env python3
"""
run_exp1_exp2.py  —  Experiment 1 (LoRA ablation) and Experiment 2 (SVD warm-start)

Baselines
    Modes: last-linear-layer (linear probe, lower bound) + full-model (FFT, upper bound)
    Datasets: SST + CFIMDB    Epochs: 10
    Output: experiments/baselines/results_baselines.json

Experiment 1
    Grid: rank ∈ {4, 8, 16, 32}  ×  target_modules ∈ {QV, QKV, ALL}  ×  init=lora
    Datasets: SST + CFIMDB    Epochs: 10  (LoRA converges within 10 epochs)
    Goal: find Config_opt = highest dev_acc with fewest trainable params (Pareto rule)
    Output: experiments/exp1/results_exp1.json
            experiments/exp1/predictions/

Experiment 2
    Grid: init_method ∈ {lora, pissa}  @  Config_opt
    Datasets: SST + CFIMDB    Epochs: 20  (PiSSA needs more epochs to show warm-start)
    New metrics: loss_after_init, steps_to_80pct_fft, steps_to_95pct_fft
    Output: experiments/exp2/results_exp2.json
            experiments/exp2/predictions/

Usage
-----
# Baselines: linear probing + FFT upper bound (both datasets by default)
python3 run_exp1_exp2.py --task baselines --smoke
python3 run_exp1_exp2.py --task baselines --use_gpu
python3 run_exp1_exp2.py --task baselines --use_gpu --dataset sst    # single dataset

# Smoke tests (fast, small data, validates pipeline)
python3 run_exp1_exp2.py --task exp1 --smoke
python3 run_exp1_exp2.py --task exp2 --smoke

# Full GPU runs (both SST and CFIMDB by default)
python3 run_exp1_exp2.py --task baselines --use_gpu
python3 run_exp1_exp2.py --task exp1 --use_gpu
python3 run_exp1_exp2.py --task exp2 --use_gpu   # auto-loads Config_opt per dataset

# Single-dataset runs
python3 run_exp1_exp2.py --task exp1 --use_gpu --dataset cfimdb
python3 run_exp1_exp2.py --task exp2 --use_gpu --dataset sst

# Override Config_opt for exp2 manually (applies to all datasets in run)
python3 run_exp1_exp2.py --task exp2 --use_gpu --configopt-rank 8 --configopt-target QV

Note: dataset defaults to SST (sentiment).  Swap DATASET_CFG and model/eval imports
      for the paraphrase task once that pipeline is integrated.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifier import GPT2SentimentClassifier, SentimentDataset, load_data, model_eval
from modules.lora_linear import LoRALinear, apply_lora, init_lora
from optimizer import AdamW
from utils import get_device

# ---------------------------------------------------------------------------
# Try importing PiSSA; skip gracefully if not present.
# ---------------------------------------------------------------------------
try:
    from lora_pissa import init_pissa
    _INIT_REGISTRY = {'lora': init_lora, 'pissa': init_pissa}
except ImportError:
    _INIT_REGISTRY = {'lora': init_lora}
    print('[warn] lora_pissa.py not found — pissa will be skipped in exp2')

# ---------------------------------------------------------------------------
# Target-module presets (attr names used by apply_lora / named_modules scan)
# ---------------------------------------------------------------------------
TARGET_CONFIGS: Dict[str, List[str]] = {
    # Attention Q+V only
    'QV':  ['query', 'value'],
    # Attention Q+K+V
    'QKV': ['query', 'key', 'value'],
    # All linear layers: attn Q/K/V + output proj + FFN up/down
    'ALL': ['query', 'key', 'value', 'attention_dense', 'interm_dense', 'out_dense'],
}

# ---------------------------------------------------------------------------
# Fixed hyper-parameters (must not change between exp1 and exp2)
# ---------------------------------------------------------------------------
FIXED = {
    'seed':            11711,
    'epochs_baseline': 10,   # linear-probe + FFT bounds
    'epochs_exp1':     10,   # LoRA ablation — converges within 10 epochs
    'epochs_exp2':     20,   # warm-start comparison — PiSSA needs more epochs to converge
    'lr':              1e-5,
    'batch_size':      8,
    'model_size':      'gpt2',
    'alpha':           16.0,
}

DATASET_CONFIGS = {
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

# Full-model dev_acc per dataset (Section 6.3) — used as FFT upper bound fallback
DEFAULT_FFT_DEV_ACC: Dict[str, float] = {
    'sst':    0.516,
    'cfimdb': 0.976,
}

EXP1_DIR          = 'experiments/exp1'
EXP2_DIR          = 'experiments/exp2'
BASELINES_DIR     = 'experiments/baselines'
EXP1_RESULTS_PATH        = f'{EXP1_DIR}/results_exp1.json'
EXP2_RESULTS_PATH        = f'{EXP2_DIR}/results_exp2.json'
EXP1_SMOKE_RESULTS_PATH  = f'{EXP1_DIR}/smoke/results_exp1_smoke.json'
EXP2_SMOKE_RESULTS_PATH  = f'{EXP2_DIR}/smoke/results_exp2_smoke.json'
BASELINES_RESULTS_PATH       = f'{BASELINES_DIR}/results_baselines.json'
BASELINES_SMOKE_RESULTS_PATH = f'{BASELINES_DIR}/smoke/results_baselines_smoke.json'

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device_name(device: torch.device) -> str:
    if device.type == 'cuda':
        return torch.cuda.get_device_name(0)
    if device.type == 'mps':
        return 'Apple MPS'
    import platform
    return f'CPU ({platform.processor() or platform.machine()})'


def load_json(path: str, default):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            content = f.read().strip()
        if content:
            return json.loads(content)
    return default


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def save_preds_csv(path: str, sent_ids, preds, true_labels) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('id\tpredicted\ttrue\n')
        for sid, p, t in zip(sent_ids, preds, true_labels):
            f.write(f'{sid}\t{p}\t{t}\n')


def already_done(results: list, key_fields: dict) -> bool:
    return any(all(r.get(k) == v for k, v in key_fields.items()) for r in results)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_dev_loss(model, dataloader, device, batch_size: int) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch['token_ids'].to(device), batch['attention_mask'].to(device))
            loss = F.cross_entropy(logits, batch['labels'].to(device).view(-1),
                                   reduction='sum') / batch_size
            total += loss.item()
            n += 1
    return total / max(n, 1)


def train_one_epoch(model, dataloader, optimizer, device, batch_size: int) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total, n = 0.0, 0, 0, 0
    for batch in tqdm(dataloader, desc='train', leave=False):
        b_ids   = batch['token_ids'].to(device)
        b_mask  = batch['attention_mask'].to(device)
        b_lbls  = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits, b_lbls.view(-1), reduction='sum') / batch_size
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == b_lbls.view(-1)).sum().item()
        total   += b_lbls.size(0)
        n += 1
    return total_loss / max(n, 1), correct / max(total, 1)


def build_model(num_labels: int, rank: int, target_key: str,
                init_method: str, device: torch.device):
    config = SimpleNamespace(
        hidden_dropout_prob=0.1,
        num_labels=num_labels,
        hidden_size=768,
        data_dir='.',
        fine_tune_mode='full-model',
    )
    model = GPT2SentimentClassifier(config)
    for p in model.gpt.parameters():
        p.requires_grad = False
    model = apply_lora(
        model,
        rank=rank,
        alpha=FIXED['alpha'],
        init_fn=_INIT_REGISTRY[init_method],
        target_modules=TARGET_CONFIGS[target_key],
    )
    return model.to(device)


def count_params(model) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_dataloaders(smoke: bool, batch_size: int, dataset: str = 'sst'):
    cfg = DATASET_CONFIGS[dataset]
    train_data, num_labels = load_data(cfg['train'], 'train')
    dev_data               = load_data(cfg['dev'],   'valid')
    test_path = cfg['test']
    has_test  = os.path.exists(test_path) and not smoke
    test_data = load_data(test_path, 'valid') if has_test else None

    if smoke:
        train_data = train_data[:64]
        dev_data   = dev_data[:32]

    ns = SimpleNamespace(hidden_size=768)  # only needed for dataset tokenisation
    train_ds = SentimentDataset(train_data, ns)
    dev_ds   = SentimentDataset(dev_data,   ns)
    kw = dict(batch_size=batch_size)
    train_loader = DataLoader(train_ds, shuffle=True,  collate_fn=train_ds.collate_fn, **kw)
    dev_loader   = DataLoader(dev_ds,   shuffle=False, collate_fn=dev_ds.collate_fn,   **kw)
    test_loader  = None
    if has_test:
        test_ds     = SentimentDataset(test_data, ns)
        test_loader = DataLoader(test_ds, shuffle=False, collate_fn=test_ds.collate_fn, **kw)
    return train_data, dev_data, test_data, num_labels, train_loader, dev_loader, test_loader


# ---------------------------------------------------------------------------
# Core training run (shared by Exp1 and Exp2)
# ---------------------------------------------------------------------------

def run_one(
    rank: int,
    target_key: str,
    init_method: str,
    train_loader, dev_loader, test_loader,
    train_data, dev_data, test_data,
    num_labels: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    smoke: bool,
    dataset: str = 'sst',
    # Exp2-only extras (pass None to skip)
    fft_dev_acc: Optional[float] = None,
) -> dict:
    seed_everything(FIXED['seed'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()

    model = build_model(num_labels, rank, target_key, init_method, device)
    trainable, total_params = count_params(model)
    print(f'  trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)')

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=FIXED['lr'])

    # ---- Exp2: record loss BEFORE any gradient update ----
    loss_after_init: Optional[float] = None
    if fft_dev_acc is not None:
        loss_after_init = round(compute_dev_loss(model, dev_loader, device, batch_size), 4)
        print(f'  loss_after_init = {loss_after_init:.4f}')

    # ---- Training loop ----
    best_acc, best_loss, best_epoch = 0.0, float('inf'), 0
    best_preds, best_true, best_sent_ids = [], [], []
    curve = {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}

    steps_per_epoch  = len(train_loader)
    steps_to_80: Optional[int] = None
    steps_to_95: Optional[int] = None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, batch_size)
        dev_acc, _, preds, true, _, sent_ids = model_eval(dev_loader, model, device)
        dev_loss = compute_dev_loss(model, dev_loader, device, batch_size)

        curve['train_loss'].append(round(train_loss, 4))
        curve['train_acc'].append(round(train_acc,  4))
        curve['dev_loss'].append(round(dev_loss,    4))
        curve['dev_acc'].append(round(dev_acc,      4))

        # Exp2: steps_to_X%_fft — evaluated at end of each epoch (epoch-level granularity)
        if fft_dev_acc is not None:
            completed_steps = (epoch + 1) * steps_per_epoch
            if steps_to_80 is None and dev_acc >= fft_dev_acc * 0.80:
                steps_to_80 = completed_steps
            if steps_to_95 is None and dev_acc >= fft_dev_acc * 0.95:
                steps_to_95 = completed_steps

        if dev_acc > best_acc:
            best_acc, best_loss, best_epoch = dev_acc, train_loss, epoch
            best_preds, best_true, best_sent_ids = preds, true, sent_ids

        print(f'  epoch {epoch}: train_loss={train_loss:.4f}  train_acc={train_acc:.4f}'
              f'  dev_loss={dev_loss:.4f}  dev_acc={dev_acc:.4f}')

    # ---- Smoke sanity assertions ----
    if smoke:
        nan_epochs = [i for i, v in enumerate(curve['train_loss']) if v != v]
        assert not nan_epochs, f'NaN in train_loss at epochs {nan_epochs}'
        assert 0.0 <= best_acc <= 1.0, f'dev_acc out of range: {best_acc}'
        if loss_after_init is not None:
            assert loss_after_init > 0, f'loss_after_init <= 0: {loss_after_init}'
        print('  [smoke] sanity assertions passed ✓')

    # ---- Predictions ----
    tag = f'{dataset}_{init_method}_{target_key}_rank{rank}'
    exp_dir = EXP2_DIR if fft_dev_acc is not None else EXP1_DIR
    pred_dir = f'{exp_dir}/smoke/predictions' if smoke else f'{exp_dir}/predictions'
    dev_pred_path = f'{pred_dir}/{tag}_dev.csv'
    save_preds_csv(dev_pred_path, best_sent_ids, best_preds, best_true)

    test_acc = test_f1 = test_pred_path = None
    if test_loader is not None:
        test_acc_val, _, tp, tt, _, ts = model_eval(test_loader, model, device)
        test_pred_path = f'{pred_dir}/{tag}_test.csv'
        save_preds_csv(test_pred_path, ts, tp, tt)
        test_acc = round(test_acc_val, 4)
        test_f1  = round(f1_score(tt, tp, average='macro'), 4)

    elapsed = time.time() - t_start
    peak_vram = (round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)
                 if device.type == 'cuda' else None)

    entry = {
        'task':           dataset,
        'init_method':    init_method,
        'target':         target_key,
        'target_modules': TARGET_CONFIGS[target_key],
        'rank':           rank,
        'alpha':          FIXED['alpha'],
        'hparams': {
            'seed': FIXED['seed'], 'epochs': epochs, 'lr': FIXED['lr'],
            'batch_size': batch_size, 'model': FIXED['model_size'],
        },
        'train_size':       len(train_data),
        'dev_size':         len(dev_data),
        'test_size':        len(test_data) if test_data else None,
        'best_epoch':       best_epoch,
        'dev_acc':          round(best_acc, 4),
        'dev_f1':           round(f1_score(best_true, best_preds, average='macro'), 4),
        'test_acc':         test_acc,
        'test_f1':          test_f1,
        'train_loss':       round(best_loss, 4),
        'trainable_params': trainable,
        'total_params':     total_params,
        'trainable_pct':    round(100 * trainable / total_params, 2),
        'peak_vram_mb':     peak_vram,
        'elapsed_min':      round(elapsed / 60, 1),
        'device':           get_device_name(device),
        'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M'),
        'dev_pred_file':    dev_pred_path,
        'test_pred_file':   test_pred_path,
        'curve':            curve,
        # Exp2 extras — None when running exp1
        'loss_after_init':    loss_after_init,
        'fft_dev_acc':        fft_dev_acc,
        'steps_to_80pct_fft': steps_to_80,
        'steps_to_95pct_fft': steps_to_95,
        'steps_per_epoch':    steps_per_epoch,
    }
    return entry


# ---------------------------------------------------------------------------
# Baselines — linear probing (last-linear-layer) and full fine-tuning (full-model)
# ---------------------------------------------------------------------------

BASELINE_MODES = ['last-linear-layer', 'full-model']


def run_one_baseline(
    fine_tune_mode: str,
    train_loader, dev_loader, test_loader,
    train_data, dev_data, test_data,
    num_labels: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    smoke: bool,
    dataset: str = 'sst',
) -> dict:
    seed_everything(FIXED['seed'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()

    config = SimpleNamespace(
        hidden_dropout_prob=0.1,
        num_labels=num_labels,
        hidden_size=768,
        data_dir='.',
        fine_tune_mode=fine_tune_mode,
    )
    model = GPT2SentimentClassifier(config).to(device)
    trainable, total_params = count_params(model)
    print(f'  trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)')

    # last-linear-layer trains only the classifier head (~3845 params); use a
    # higher lr matching classifier.py defaults so the probe converges properly.
    baseline_lr = 1e-3 if fine_tune_mode == 'last-linear-layer' else FIXED['lr']
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=baseline_lr)

    best_acc, best_loss, best_epoch = 0.0, float('inf'), 0
    best_preds, best_true, best_sent_ids = [], [], []
    curve = {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, batch_size)
        dev_acc, _, preds, true, _, sent_ids = model_eval(dev_loader, model, device)
        dev_loss = compute_dev_loss(model, dev_loader, device, batch_size)

        curve['train_loss'].append(round(train_loss, 4))
        curve['train_acc'].append(round(train_acc,  4))
        curve['dev_loss'].append(round(dev_loss,    4))
        curve['dev_acc'].append(round(dev_acc,      4))

        if dev_acc > best_acc:
            best_acc, best_loss, best_epoch = dev_acc, train_loss, epoch
            best_preds, best_true, best_sent_ids = preds, true, sent_ids

        print(f'  epoch {epoch}: train_loss={train_loss:.4f}  train_acc={train_acc:.4f}'
              f'  dev_loss={dev_loss:.4f}  dev_acc={dev_acc:.4f}')

    if smoke:
        nan_epochs = [i for i, v in enumerate(curve['train_loss']) if v != v]
        assert not nan_epochs, f'NaN in train_loss at epochs {nan_epochs}'
        assert 0.0 <= best_acc <= 1.0, f'dev_acc out of range: {best_acc}'
        print('  [smoke] sanity assertions passed ✓')

    tag = f'{dataset}_{fine_tune_mode.replace("-", "_")}'
    pred_dir = (f'{BASELINES_DIR}/smoke/predictions' if smoke
                else f'{BASELINES_DIR}/predictions')
    dev_pred_path = f'{pred_dir}/{tag}_dev.csv'
    save_preds_csv(dev_pred_path, best_sent_ids, best_preds, best_true)

    test_acc = test_f1 = test_pred_path = None
    if test_loader is not None:
        test_acc_val, _, tp, tt, _, ts = model_eval(test_loader, model, device)
        test_pred_path = f'{pred_dir}/{tag}_test.csv'
        save_preds_csv(test_pred_path, ts, tp, tt)
        test_acc = round(test_acc_val, 4)
        test_f1  = round(f1_score(tt, tp, average='macro'), 4)

    elapsed  = time.time() - t_start
    peak_vram = (round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)
                 if device.type == 'cuda' else None)

    return {
        'task':           dataset,
        'fine_tune_mode': fine_tune_mode,
        'hparams': {
            'seed': FIXED['seed'], 'epochs': epochs, 'lr': baseline_lr,
            'batch_size': batch_size, 'model': FIXED['model_size'],
        },
        'train_size':     len(train_data),
        'dev_size':       len(dev_data),
        'test_size':      len(test_data) if test_data else None,
        'best_epoch':     best_epoch,
        'dev_acc':        round(best_acc, 4),
        'dev_f1':         round(f1_score(best_true, best_preds, average='macro'), 4),
        'test_acc':       test_acc,
        'test_f1':        test_f1,
        'train_loss':     round(best_loss, 4),
        'trainable_params': trainable,
        'total_params':   total_params,
        'trainable_pct':  round(100 * trainable / total_params, 2),
        'peak_vram_mb':   peak_vram,
        'elapsed_min':    round(elapsed / 60, 1),
        'device':         get_device_name(device),
        'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M'),
        'dev_pred_file':  dev_pred_path,
        'test_pred_file': test_pred_path,
        'curve':          curve,
    }


def run_baselines(args) -> None:
    device     = get_device(args.use_gpu)
    batch_size = args.batch_size or FIXED['batch_size']
    epochs     = 1 if args.smoke else (args.epochs or FIXED['epochs_baseline'])
    datasets   = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())

    results_path = BASELINES_SMOKE_RESULTS_PATH if args.smoke else BASELINES_RESULTS_PATH
    results: list = load_json(results_path, [])

    for dataset in datasets:
        train_data, dev_data, test_data, num_labels, train_loader, dev_loader, test_loader = \
            build_dataloaders(args.smoke, batch_size, dataset)

        for mode in BASELINE_MODES:
            if already_done(results, {'fine_tune_mode': mode, 'task': dataset}) and not args.rerun:
                print(f'\n=== [skip] baselines | dataset={dataset} | mode={mode} ===')
                continue
            print(f'\n=== baselines | dataset={dataset} | mode={mode} ===')
            entry = run_one_baseline(
                fine_tune_mode=mode,
                train_loader=train_loader, dev_loader=dev_loader, test_loader=test_loader,
                train_data=train_data, dev_data=dev_data, test_data=test_data,
                num_labels=num_labels, device=device, epochs=epochs,
                batch_size=batch_size, smoke=args.smoke, dataset=dataset,
            )
            results.append(entry)
            save_json(results_path, results)
            print(f'  => dev_acc={entry["dev_acc"]:.4f}  '
                  f'trainable={entry["trainable_pct"]:.2f}%  ({mode})')

    if not args.smoke:
        _print_baseline_summary(results)


def _print_baseline_summary(results: list) -> None:
    print('\n' + '='*60)
    print('BASELINES SUMMARY')
    for r in results:
        print(f'  {r["fine_tune_mode"]:25s}  dev_acc={r["dev_acc"]:.4f}'
              f'  trainable={r["trainable_pct"]:.2f}%')
    print('='*60)


# ---------------------------------------------------------------------------
# Experiment 1 — LoRA ablation
# ---------------------------------------------------------------------------

def run_exp1(args) -> None:
    # smoke: rank=4 only, but ALL three target configs (to test QV/QKV/ALL paths)
    ranks       = [4] if args.smoke else [4, 8, 16, 32]
    target_keys = list(TARGET_CONFIGS.keys())   # always run all three targets
    device      = get_device(args.use_gpu)
    batch_size  = args.batch_size or FIXED['batch_size']
    epochs      = 1 if args.smoke else (args.epochs or FIXED['epochs_exp1'])
    datasets    = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())

    results_path = EXP1_SMOKE_RESULTS_PATH if args.smoke else EXP1_RESULTS_PATH
    results: list = load_json(results_path, [])

    for dataset in datasets:
        train_data, dev_data, test_data, num_labels, train_loader, dev_loader, test_loader = \
            build_dataloaders(args.smoke, batch_size, dataset)

        for target_key in target_keys:
            for rank in ranks:
                key = {'init_method': 'lora', 'target': target_key, 'rank': rank, 'task': dataset}
                if already_done(results, key) and not args.rerun:
                    print(f'\n=== [skip] exp1 | dataset={dataset} | target={target_key} | rank={rank} ===')
                    continue
                print(f'\n=== exp1 | dataset={dataset} | target={target_key} | rank={rank} ===')
                entry = run_one(
                    rank=rank, target_key=target_key, init_method='lora',
                    train_loader=train_loader, dev_loader=dev_loader, test_loader=test_loader,
                    train_data=train_data, dev_data=dev_data, test_data=test_data,
                    num_labels=num_labels, device=device, epochs=epochs,
                    batch_size=batch_size, smoke=args.smoke, dataset=dataset,
                    fft_dev_acc=None,
                )
                results.append(entry)
                save_json(results_path, results)
                print(f'  => dev_acc={entry["dev_acc"]:.4f}  trainable={entry["trainable_pct"]:.2f}%')

    # Print Config_opt suggestion after all runs (per dataset)
    if not args.smoke and results:
        for dataset in datasets:
            ds_results = [r for r in results if r.get('task') == dataset]
            if ds_results:
                print(f'\n[{dataset.upper()}]')
                _print_configopt(ds_results)


def _print_configopt(results: list) -> None:
    """Pareto rule: dev_acc >= 95% of best dev_acc, then fewest trainable params."""
    best_dev = max(r['dev_acc'] for r in results)
    threshold = best_dev * 0.95
    candidates = [r for r in results if r['dev_acc'] >= threshold]
    if not candidates:
        return
    opt = min(candidates, key=lambda r: r['trainable_pct'])
    print('\n' + '='*60)
    print('CONFIG_OPT (Pareto: dev_acc >= 95% best, fewest params)')
    print(f'  rank    = {opt["rank"]}')
    print(f'  target  = {opt["target"]}  {opt["target_modules"]}')
    print(f'  dev_acc = {opt["dev_acc"]:.4f}  (best={best_dev:.4f}, threshold={threshold:.4f})')
    print(f'  trainable = {opt["trainable_pct"]:.2f}%')
    print('='*60)


def load_configopt(args, dataset: str = 'sst') -> Tuple[int, str]:
    """Load Config_opt for a given dataset: prefer CLI args, then infer from exp1 results.

    Search order:
      1. --configopt-rank / --configopt-target  (always wins, applies to all datasets)
      2. Formal exp1 results  (experiments/exp1/results_exp1.json)
      3. Smoke exp1 results   (experiments/exp1/smoke/results_exp1_smoke.json)
         — only used when --smoke is set, to allow smoke exp2 without formal exp1
    """
    if args.configopt_rank is not None and args.configopt_target is not None:
        return args.configopt_rank, args.configopt_target

    # Try formal results first, fall back to smoke results in smoke mode
    all_results: list = load_json(EXP1_RESULTS_PATH, [])
    source = 'formal'
    if not all_results and getattr(args, 'smoke', False):
        all_results = load_json(EXP1_SMOKE_RESULTS_PATH, [])
        source = 'smoke'

    results = [r for r in all_results if r.get('task') == dataset]

    if not results:
        raise RuntimeError(
            f'Exp1 results not found for dataset={dataset}. Options:\n'
            '  1. Run exp1 first:  python3 run_exp1_exp2.py --task exp1 [--smoke]\n'
            '  2. Specify manually: --configopt-rank 4 --configopt-target QKV'
        )

    best_dev = max(r['dev_acc'] for r in results)
    threshold = best_dev * 0.95
    candidates = [r for r in results if r['dev_acc'] >= threshold]
    opt = min(candidates, key=lambda r: r['trainable_pct'])
    print(f'[config_opt/{dataset}] auto-selected from {source} exp1 results: '
          f'rank={opt["rank"]}  target={opt["target"]}  '
          f'dev_acc={opt["dev_acc"]:.4f}  trainable={opt["trainable_pct"]:.2f}%')
    return opt['rank'], opt['target']


# ---------------------------------------------------------------------------
# Experiment 2 — SVD warm-start (LoRA vs PiSSA @ Config_opt)
# ---------------------------------------------------------------------------

def run_exp2(args) -> None:
    # smoke also runs both lora and pissa to test the SVD init path
    init_methods = [m for m in ['lora', 'pissa'] if m in _INIT_REGISTRY]
    device     = get_device(args.use_gpu)
    batch_size = args.batch_size or FIXED['batch_size']
    epochs     = 1 if args.smoke else (args.epochs or FIXED['epochs_exp2'])
    datasets   = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())

    results_path = EXP2_SMOKE_RESULTS_PATH if args.smoke else EXP2_RESULTS_PATH
    results: list = load_json(results_path, [])

    for dataset in datasets:
        rank, target_key = load_configopt(args, dataset)
        fft_dev_acc = args.fft_dev_acc or DEFAULT_FFT_DEV_ACC.get(dataset, DEFAULT_FFT_DEV_ACC['sst'])

        print(f'\nExp2 config [{dataset}]: rank={rank}  target={target_key}  '
              f'fft_dev_acc={fft_dev_acc}  init_methods={init_methods}')

        train_data, dev_data, test_data, num_labels, train_loader, dev_loader, test_loader = \
            build_dataloaders(args.smoke, batch_size, dataset)

        for init_method in init_methods:
            key = {'init_method': init_method, 'target': target_key, 'rank': rank, 'task': dataset}
            if already_done(results, key) and not args.rerun:
                print(f'\n=== [skip] exp2 | dataset={dataset} | init={init_method} ===')
                continue
            print(f'\n=== exp2 | dataset={dataset} | init={init_method} | rank={rank} | target={target_key} ===')
            entry = run_one(
                rank=rank, target_key=target_key, init_method=init_method,
                train_loader=train_loader, dev_loader=dev_loader, test_loader=test_loader,
                train_data=train_data, dev_data=dev_data, test_data=test_data,
                num_labels=num_labels, device=device, epochs=epochs,
                batch_size=batch_size, smoke=args.smoke, dataset=dataset,
                fft_dev_acc=fft_dev_acc,
            )
            results.append(entry)
            save_json(results_path, results)
            print(f'  => dev_acc={entry["dev_acc"]:.4f}  '
                  f'loss_after_init={entry["loss_after_init"]}  '
                  f'steps_to_80%={entry["steps_to_80pct_fft"]}  '
                  f'steps_to_95%={entry["steps_to_95pct_fft"]}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Exp1 (LoRA ablation) and Exp2 (SVD warm-start)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--task', required=True, choices=['baselines', 'exp1', 'exp2'],
                   help='Which experiment to run  (baselines → linear-probe + FFT bounds)')
    p.add_argument('--use_gpu',  action='store_true')
    p.add_argument('--smoke',    action='store_true',
                   help='Quick functional check: small data, 1 epoch, rank4 only')
    p.add_argument('--rerun',    action='store_true',
                   help='Re-run even if results already exist')
    p.add_argument('--epochs',   type=int, default=None,
                   help='Override number of training epochs')
    p.add_argument('--batch_size', type=int, default=None,
                   help='Override batch size')
    p.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()), default=None,
                   help='Run only the specified dataset (default: all datasets)')
    # Exp2 options
    fft_defaults = ', '.join(f'{k}={v}' for k, v in DEFAULT_FFT_DEV_ACC.items())
    p.add_argument('--fft-dev-acc', type=float, default=None, dest='fft_dev_acc',
                   help=f'Override FFT upper-bound dev_acc for steps_to_X%% (defaults: {fft_defaults})')
    p.add_argument('--configopt-rank', type=int, default=None, dest='configopt_rank',
                   help='Manually specify Config_opt rank (skips auto-loading from exp1)')
    p.add_argument('--configopt-target', choices=list(TARGET_CONFIGS.keys()), default=None,
                   dest='configopt_target',
                   help='Manually specify Config_opt target (skips auto-loading from exp1)')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.task == 'baselines':
        run_baselines(args)
    elif args.task == 'exp1':
        run_exp1(args)
    else:
        run_exp2(args)
