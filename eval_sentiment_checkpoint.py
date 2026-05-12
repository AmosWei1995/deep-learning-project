#!/usr/bin/env python3
"""
Recompute dev accuracy (and macro-F1) from a saved sentiment classifier checkpoint.

Use when you closed the terminal and no longer have the printed `dev acc` lines.

Examples (paths match classifier.py after saving under checkpoints/):
  python3 eval_sentiment_checkpoint.py --checkpoint checkpoints/last-linear-layer-sst-classifier.pt --dev data/ids-sst-dev.csv --use_gpu
  python3 eval_sentiment_checkpoint.py --checkpoint checkpoints/full-model-cfimdb-classifier.pt --dev data/ids-cfimdb-dev.csv --use_gpu
"""

import argparse

import torch
from torch.utils.data import DataLoader

import classifier as clf
from utils import get_device


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="e.g. checkpoints/last-linear-layer-sst-classifier.pt",
  )
  parser.add_argument("--dev", type=str, required=True, help="e.g. data/ids-sst-dev.csv")
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--use_gpu", action="store_true")
  args = parser.parse_args()

  device = get_device(args.use_gpu)

  try:
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
  except TypeError:
    saved = torch.load(args.checkpoint, map_location=device)

  config = saved["model_config"]
  model = clf.GPT2SentimentClassifier(config)
  model.load_state_dict(saved["model"])
  model = model.to(device)
  model.eval()

  dev_data, _ = clf.load_data(args.dev, "valid")
  eval_args = type(
    "EvalArgs",
    (),
    {
      "batch_size": args.batch_size,
      "hidden_dropout_prob": getattr(config, "hidden_dropout_prob", 0.3),
      "fine_tune_mode": config.fine_tune_mode,
    },
  )()
  dev_dataset = clf.SentimentDataset(dev_data, eval_args)
  dev_loader = DataLoader(
    dev_dataset,
    shuffle=False,
    batch_size=args.batch_size,
    collate_fn=dev_dataset.collate_fn,
  )

  acc, f1, *_ = clf.model_eval(dev_loader, model, device)
  print(f"checkpoint: {args.checkpoint}")
  print(f"dev file:    {args.dev}")
  print(f"dev acc:     {acc:.3f}")
  print(f"dev macro-f1:{f1:.3f}")


if __name__ == "__main__":
  main()
