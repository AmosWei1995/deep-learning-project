#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from adapters.apply_quant_lora import apply_quant_lora
from config import default_exp3_config, load_baseline_from_results
from quant.memory import peak_memory_gb, reset_peak_memory

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Experiment 3: QLoRA vs LoftQ under quantization.")
  parser.add_argument("--dlp-root", type=str, default=os.environ.get("DLP_ROOT", ""))
  parser.add_argument("--dataset", type=str, default="sst", choices=["sst"])
  parser.add_argument("--method", type=str, default=None, choices=["qlora", "loftq"])
  parser.add_argument("--bits", type=int, default=None, choices=[2, 4])
  parser.add_argument("--rank", type=int, default=None)
  parser.add_argument("--alpha", type=float, default=None)
  parser.add_argument("--epochs", type=int, default=None)
  parser.add_argument("--batch_size", type=int, default=None)
  parser.add_argument("--baseline-dev-acc", type=float, default=None)
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument("--use_gpu", action="store_true")
  parser.add_argument("--rerun", action="store_true")
  return parser.parse_args()


def add_project_root_to_syspath(project_root: str) -> None:
  if not project_root:
    raise ValueError("Project root is empty. Set DLP_ROOT or pass --dlp-root.")
  if not os.path.exists(project_root):
    raise FileNotFoundError(f"Project root not found: {project_root}")
  if project_root not in sys.path:
    sys.path.insert(0, project_root)


def ensure_project_config_module(project_root: str) -> None:
  """
  Prevent local extension `config.py` from shadowing project `config.py`.
  """
  config_path = os.path.join(project_root, "config.py")
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"Project config not found: {config_path}")
  spec = importlib.util.spec_from_file_location("config", config_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load project config module from {config_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  sys.modules["config"] = module


def seed_everything(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)


def compute_dev_loss(model, dataloader, device, batch_size: int) -> float:
  model.eval()
  total_loss, n = 0.0, 0
  with torch.no_grad():
    for batch in dataloader:
      b_ids = batch["token_ids"].to(device)
      b_mask = batch["attention_mask"].to(device)
      b_labels = batch["labels"].to(device)
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, b_labels.view(-1), reduction="sum") / batch_size
      total_loss += loss.item()
      n += 1
  return total_loss / max(1, n)


def train_one_epoch(model, dataloader, optimizer, device, batch_size: int) -> Tuple[float, float]:
  model.train()
  total_loss, correct, total, n = 0.0, 0, 0, 0
  for batch in tqdm(dataloader, desc="train"):
    b_ids = batch["token_ids"].to(device)
    b_mask = batch["attention_mask"].to(device)
    b_labels = batch["labels"].to(device)
    optimizer.zero_grad()
    logits = model(b_ids, b_mask)
    loss = F.cross_entropy(logits, b_labels.view(-1), reduction="sum") / batch_size
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    correct += (logits.argmax(dim=-1) == b_labels.view(-1)).sum().item()
    total += b_labels.size(0)
    n += 1
  return total_loss / max(1, n), correct / max(1, total)


def save_preds_csv(path: str, sent_ids: List[str], preds: List[int], true_labels: List[int]) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    f.write("id\tpredicted\ttrue\n")
    for sid, pred, true in zip(sent_ids, preds, true_labels):
      f.write(f"{sid}\t{pred}\t{true}\n")


def load_existing_results(path: str) -> Dict:
  if not os.path.exists(path):
    return {"experiment": "exp3_quant_svd", "baseline": {}, "runs": []}
  with open(path, "r", encoding="utf-8") as f:
    content = f.read().strip()
  if not content:
    return {"experiment": "exp3_quant_svd", "baseline": {}, "runs": []}
  return json.loads(content)


def already_done(runs: List[Dict], method: str, bits: int, rank: int, dataset: str) -> bool:
  for run in runs:
    if (
      run.get("method") == method
      and int(run.get("bits", -1)) == int(bits)
      and int(run.get("rank", -1)) == int(rank)
      and run.get("dataset") == dataset
    ):
      return True
  return False


def get_device_name(device: torch.device) -> str:
  if device.type == "cuda":
    return torch.cuda.get_device_name(0)
  if device.type == "mps":
    return "Apple MPS"
  return "CPU"


def build_dataloaders(args, dataset_name: str, batch_size: int):
  from classifier import SentimentDataset, load_data

  ds_cfg = {
    "sst": {
      "train": "data/ids-sst-train.csv",
      "dev": "data/ids-sst-dev.csv",
      "test": "data/ids-sst-test.csv",
    }
  }[dataset_name]

  train_data, num_labels = load_data(ds_cfg["train"], "train")
  dev_data = load_data(ds_cfg["dev"], "valid")
  test_path = ds_cfg["test"]
  has_test = bool(test_path) and os.path.exists(test_path) and not args.smoke
  test_data = load_data(test_path, "valid") if has_test else None

  if args.smoke:
    train_data = train_data[:64]
    dev_data = dev_data[:32]

  train_dataset = SentimentDataset(train_data, args)
  dev_dataset = SentimentDataset(dev_data, args)
  train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=train_dataset.collate_fn,
  )
  dev_loader = DataLoader(
    dev_dataset,
    shuffle=False,
    batch_size=batch_size,
    collate_fn=dev_dataset.collate_fn,
  )
  test_loader = None
  if has_test:
    test_dataset = SentimentDataset(test_data, args)
    test_loader = DataLoader(
      test_dataset,
      shuffle=False,
      batch_size=batch_size,
      collate_fn=test_dataset.collate_fn,
    )
  return train_data, dev_data, test_data, num_labels, train_loader, dev_loader, test_loader


def train_one_run(
  args,
  method: str,
  bits: int,
  rank: int,
  alpha: float,
  baseline_dev_acc: float,
  results_dir: str,
) -> Dict:
  from classifier import GPT2SentimentClassifier, model_eval
  from optimizer import AdamW
  from utils import get_device

  seed_everything(args.seed)
  device = get_device(args.use_gpu)
  reset_peak_memory(device)
  t_start = time.time()

  train_data, dev_data, test_data, num_labels, train_loader, dev_loader, test_loader = build_dataloaders(
    args, args.dataset, args.batch_size
  )

  config = SimpleNamespace(
    hidden_dropout_prob=0.1,
    num_labels=num_labels,
    hidden_size=768,
    data_dir=".",
    fine_tune_mode="full-model",
  )

  model = GPT2SentimentClassifier(config)
  for p in model.gpt.parameters():
    p.requires_grad = False

  model = apply_quant_lora(
    model,
    rank=rank,
    alpha=alpha,
    num_bits=bits,
    method=method,
    target_modules=args.target_modules,
  )
  model = model.to(device)

  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  total_params = sum(p.numel() for p in model.parameters())
  print(
    f"\n=== {method} | {bits}-bit | rank={rank} ===\n"
    f"  trainable params: {trainable:,} / {total_params:,}"
    f" ({100*trainable/max(1,total_params):.2f}%)"
  )

  optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

  best_acc, best_loss, best_epoch = 0.0, float("inf"), 0
  best_preds, best_true, best_sent_ids = [], [], []
  curve = {"train_loss": [], "train_acc": [], "dev_loss": [], "dev_acc": []}

  for epoch in range(args.epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, args.batch_size)
    dev_acc, _, preds, true, _, sent_ids = model_eval(dev_loader, model, device)
    dev_loss = compute_dev_loss(model, dev_loader, device, args.batch_size)

    curve["train_loss"].append(round(train_loss, 4))
    curve["train_acc"].append(round(train_acc, 4))
    curve["dev_loss"].append(round(dev_loss, 4))
    curve["dev_acc"].append(round(dev_acc, 4))
    if dev_acc > best_acc:
      best_acc, best_loss, best_epoch = dev_acc, train_loss, epoch
      best_preds, best_true, best_sent_ids = preds, true, sent_ids

    print(
      f"  epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
      f"dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f}"
    )

  preds_dir = os.path.join(results_dir, "preds")
  dev_pred_file = os.path.join(
    preds_dir, f"sentiment_{args.dataset}_{method}_{bits}bit_rank{rank}_dev.csv"
  )
  save_preds_csv(dev_pred_file, best_sent_ids, best_preds, best_true)

  test_acc = test_f1 = None
  test_pred_file = None
  if test_loader is not None:
    test_acc_val, _, test_preds, test_true, _, test_sent_ids = model_eval(test_loader, model, device)
    test_pred_file = os.path.join(
      preds_dir, f"sentiment_{args.dataset}_{method}_{bits}bit_rank{rank}_test.csv"
    )
    save_preds_csv(test_pred_file, test_sent_ids, test_preds, test_true)
    test_acc = round(test_acc_val, 4)
    test_f1 = round(f1_score(test_true, test_preds, average="macro"), 4)

  peak_vram = peak_memory_gb(device)
  elapsed = time.time() - t_start
  best_f1 = round(f1_score(best_true, best_preds, average="macro"), 4)
  degradation = round(best_acc - baseline_dev_acc, 4)
  degradation_pct = round(100.0 * degradation / baseline_dev_acc, 2) if baseline_dev_acc > 0 else None

  return {
    "task": "sentiment",
    "dataset": args.dataset,
    "method": method,
    "bits": bits,
    "rank": rank,
    "alpha": alpha,
    "target_modules": args.target_modules,
    "hparams": {
      "seed": args.seed,
      "epochs": args.epochs,
      "lr": args.lr,
      "batch_size": args.batch_size,
      "model": args.model_size,
    },
    "train_size": len(train_data),
    "dev_size": len(dev_data),
    "test_size": len(test_data) if test_data is not None else None,
    "best_epoch": best_epoch,
    "dev_acc": round(best_acc, 4),
    "dev_f1": best_f1,
    "test_acc": test_acc,
    "test_f1": test_f1,
    "train_loss": round(best_loss, 4),
    "trainable_params": trainable,
    "total_params": total_params,
    "trainable_pct": round(100 * trainable / max(1, total_params), 2),
    "peak_vram_gb": round(peak_vram, 3) if peak_vram is not None else None,
    "degradation": degradation,
    "degradation_pct": degradation_pct,
    "elapsed_min": round(elapsed / 60.0, 1),
    "device": get_device_name(device),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "dev_pred_file": dev_pred_file,
    "test_pred_file": test_pred_file,
    "curve": curve,
  }


def main() -> None:
  cli = parse_args()
  cfg = default_exp3_config(cli.dlp_root)

  dlp_root = cli.dlp_root or cfg["dlp_root"]
  add_project_root_to_syspath(dlp_root)
  ensure_project_config_module(dlp_root)
  os.chdir(dlp_root)

  baseline = load_baseline_from_results(dlp_root, dataset=cli.dataset)
  if cli.baseline_dev_acc is not None:
    baseline["dev_acc"] = cli.baseline_dev_acc

  args = SimpleNamespace(
    dataset=cli.dataset,
    seed=cfg["seed"],
    epochs=2 if cli.smoke else (cli.epochs or cfg["epochs"]),
    lr=cfg["lr"],
    batch_size=cli.batch_size or cfg["batch_size"],
    model_size=cfg["model_size"],
    use_gpu=cli.use_gpu,
    smoke=cli.smoke,
    rerun=cli.rerun,
    rank=cli.rank or cfg["rank"],
    alpha=cli.alpha or cfg["alpha"],
    target_modules=cfg["target_modules"],
  )

  methods = [cli.method] if cli.method else cfg["methods"]
  bits_list = [cli.bits] if cli.bits else cfg["bits_list"]

  results_dir = os.path.join(EXT_ROOT, "results")
  os.makedirs(results_dir, exist_ok=True)
  results_path = (
    os.path.join(results_dir, "exp3_smoke_results.json")
    if cli.smoke
    else os.path.join(results_dir, "exp3_results.json")
  )
  payload = load_existing_results(results_path)
  payload["experiment"] = "exp3_quant_svd"
  payload["baseline"] = baseline
  payload.setdefault("runs", [])

  for method in methods:
    for bits in bits_list:
      if (not cli.rerun) and already_done(payload["runs"], method, bits, args.rank, args.dataset):
        print(f"skip: already exists {method} {bits}-bit rank={args.rank}")
        continue
      entry = train_one_run(
        args=args,
        method=method,
        bits=bits,
        rank=args.rank,
        alpha=args.alpha,
        baseline_dev_acc=baseline["dev_acc"],
        results_dir=results_dir,
      )
      payload["runs"].append(entry)
      with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
      print(
        f"  => saved: method={method}, bits={bits}, "
        f"dev_acc={entry['dev_acc']:.4f}, peak_vram={entry['peak_vram_gb']}"
      )

  print(f"\nDone. Results: {results_path}")


if __name__ == "__main__":
  main()
