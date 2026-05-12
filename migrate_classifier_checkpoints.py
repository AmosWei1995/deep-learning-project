#!/usr/bin/env python3
"""
Copy legacy sentiment checkpoints from project root into checkpoints/ with the
same naming scheme as classifier.py (after the path update).

Run from project root after training finishes, e.g.:

  python3 migrate_classifier_checkpoints.py --fine-tune-mode last-linear-layer

If the destination file already exists, the script skips (use --force to overwrite).
Use --remove-source only if you are sure you no longer need the root copies.
"""

import argparse
import os
import shutil


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--fine-tune-mode",
    default="last-linear-layer",
    choices=("last-linear-layer", "full-model"),
    help="Must match the run you want to archive (default: last-linear-layer).",
  )
  parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite destination if it already exists.",
  )
  parser.add_argument(
    "--remove-source",
    action="store_true",
    help="Delete root-level .pt after a successful copy.",
  )
  args = parser.parse_args()

  root = os.path.dirname(os.path.abspath(__file__))
  ckpt_dir = os.path.join(root, "checkpoints")
  os.makedirs(ckpt_dir, exist_ok=True)
  prefix = args.fine_tune_mode

  pairs = [
    (os.path.join(root, "sst-classifier.pt"), os.path.join(ckpt_dir, f"{prefix}-sst-classifier.pt")),
    (os.path.join(root, "cfimdb-classifier.pt"), os.path.join(ckpt_dir, f"{prefix}-cfimdb-classifier.pt")),
  ]

  for src, dst in pairs:
    if not os.path.isfile(src):
      print(f"[skip] missing source: {src}")
      continue
    if os.path.isfile(dst) and not args.force:
      print(f"[skip] destination exists (use --force): {dst}")
      continue
    shutil.copy2(src, dst)
    print(f"[ok]   {src} -> {dst}")
    if args.remove_source:
      os.remove(src)
      print(f"[rm]   {src}")


if __name__ == "__main__":
  main()
