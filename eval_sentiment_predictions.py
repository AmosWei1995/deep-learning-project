#!/usr/bin/env python3
"""
Compute dev accuracy and macro-F1 from prediction CSV + labeled dev CSV (no checkpoint).

classifier.py writes dev predictions like:
  id \\t Predicted_Sentiment   (header)
  <sent_id>, <pred>            (body)

Examples:
  python3 eval_sentiment_predictions.py \\
    --pred predictions/full-model-sst-dev-out.csv \\
    --dev data/ids-sst-dev.csv

  python3 eval_sentiment_predictions.py \\
    --pred predictions/last-linear-layer-cfimdb-dev-out.csv \\
    --dev data/ids-cfimdb-dev.csv
"""

import argparse
import csv
import re
from typing import Dict, List

from sklearn.metrics import accuracy_score, f1_score


def load_dev_labels(dev_path: str) -> Dict[str, int]:
  """Map sentence id -> gold sentiment (same ids as classifier.load_data)."""
  gold: Dict[str, int] = {}
  with open(dev_path, newline="", encoding="utf-8") as fp:
    reader = csv.DictReader(fp, delimiter="\t")
    for row in reader:
      sid = row["id"].lower().strip()
      gold[sid] = int(row["sentiment"].strip())
  return gold


def load_predictions(pred_path: str) -> Dict[str, int]:
  """
  Parse classifier.py dev output: skip header; each line 'id, pred' or 'id, pred '.
  """
  preds: Dict[str, int] = {}
  with open(pred_path, encoding="utf-8") as fp:
    for line in fp:
      line = line.strip()
      if not line:
        continue
      lower = line.lower()
      if lower.startswith("id") and "predicted" in lower:
        continue
      # Split on first comma only (ids do not contain commas).
      parts = line.split(",", 1)
      if len(parts) != 2:
        continue
      sid = parts[0].strip().lower()
      pred_str = parts[1].strip()
      pred_str = re.sub(r"\s+", "", pred_str)
      preds[sid] = int(pred_str)
  return preds


def main():
  parser = argparse.ArgumentParser(description="Eval dev metrics from prediction file + dev labels.")
  parser.add_argument("--pred", type=str, required=True, help="e.g. predictions/full-model-sst-dev-out.csv")
  parser.add_argument("--dev", type=str, required=True, help="e.g. data/ids-sst-dev.csv")
  args = parser.parse_args()

  gold = load_dev_labels(args.dev)
  preds = load_predictions(args.pred)

  missing_in_pred = [k for k in gold if k not in preds]
  extra_in_pred = [k for k in preds if k not in gold]

  if missing_in_pred:
    raise SystemExit(
      f"{len(missing_in_pred)} dev ids missing in predictions (showing up to 5): {missing_in_pred[:5]}"
    )
  if extra_in_pred:
    print(f"warning: {len(extra_in_pred)} prediction ids not in dev file (ignored)")

  y_true: List[int] = []
  y_pred: List[int] = []
  for sid in gold:
    if sid not in preds:
      continue
    y_true.append(gold[sid])
    y_pred.append(preds[sid])

  acc = accuracy_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred, average="macro")
  print(f"pred file:   {args.pred}")
  print(f"dev file:    {args.dev}")
  print(f"dev acc:     {acc:.3f}")
  print(f"dev macro-f1:{f1:.3f}")


if __name__ == "__main__":
  main()
