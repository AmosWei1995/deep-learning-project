import json
import os
from typing import Dict, List


def _default_dlp_root() -> str:
  # extension file -> deep-learning-project/extensions/exp3_quant_lora/config.py
  # project root is two levels up from this file.
  return os.environ.get(
    "DLP_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
  )


def load_baseline_from_results(dlp_root: str, dataset: str = "sst") -> Dict:
  results_path = os.path.join(dlp_root, "predictions", "results.json")
  if not os.path.exists(results_path):
    raise FileNotFoundError(
      f"Baseline results file not found: {results_path}. "
      "Run the baseline LoRA experiment first or pass --baseline-dev-acc."
    )

  with open(results_path, "r", encoding="utf-8") as f:
    content = f.read().strip()
  if not content:
    raise ValueError(f"Baseline results file is empty: {results_path}")

  rows: List[Dict] = json.loads(content)
  candidates = [
    r for r in rows
    if r.get("task") == "sentiment"
    and r.get("dataset") == dataset
    and r.get("init_method") == "lora"
  ]
  if not candidates:
    raise ValueError(
      f"No FP LoRA baseline found for dataset={dataset} in {results_path}"
    )

  best = max(candidates, key=lambda r: r.get("dev_acc", float("-inf")))
  return {
    "source": results_path,
    "dataset": dataset,
    "dev_acc": float(best["dev_acc"]),
    "rank": int(best["rank"]),
    "alpha": float(best.get("hparams", {}).get("alpha", 16.0)),
    "seed": int(best.get("hparams", {}).get("seed", 11711)),
    "lr": float(best.get("hparams", {}).get("lr", 1e-5)),
    "batch_size": int(best.get("hparams", {}).get("batch_size", 8)),
    "epochs": int(best.get("hparams", {}).get("epochs", 10)),
    "model": best.get("hparams", {}).get("model", "gpt2"),
    "target_modules": ["query", "key", "value"],
  }


def default_exp3_config(dlp_root: str = "") -> Dict:
  root = dlp_root or _default_dlp_root()
  baseline = load_baseline_from_results(root, dataset="sst")
  return {
    "dlp_root": root,
    "dataset": "sst",
    "methods": ["qlora", "loftq"],
    "bits_list": [4, 2],
    "rank": baseline["rank"],
    "alpha": baseline["alpha"],
    "seed": baseline["seed"],
    "epochs": baseline["epochs"],
    "lr": baseline["lr"],
    "batch_size": baseline["batch_size"],
    "model_size": baseline["model"],
    "target_modules": baseline["target_modules"],
    "baseline": baseline,
  }
