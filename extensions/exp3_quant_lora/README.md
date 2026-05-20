# Experiment 3: Quantized QLoRA vs LoftQ (Project-integrated Extension)

This extension runs Experiment 3 inside the main project while reusing the
existing SST data pipeline and evaluation code.

## What this extension does

- Compares `qlora` vs `loftq` under quantized frozen GPT-2 weights.
- Runs 4 configurations: `{qlora, loftq} x {4-bit, 2-bit}`.
- Uses the same `rank`, `target_modules`, and training hparams as baseline.
- Reports:
  - `dev_acc`
  - degradation vs FP LoRA baseline
  - peak VRAM (GB)

## Setup

```bash
cd /home/ray/yuheng/Dsinai/project/deep-learning-project/extensions/exp3_quant_lora
python run_exp3.py --smoke --use_gpu
```

## Run

```bash
# Smoke check
python run_exp3.py --smoke --use_gpu

# Full 4-run grid
python run_exp3.py --use_gpu

# Single run
python run_exp3.py --use_gpu --method loftq --bits 4
```

## Output

- `results/exp3_results.json`: all run entries + baseline + degradation stats.
- `results/exp3_smoke_results.json`: smoke mode output.
- `results/preds/*.csv`: dev/test predictions for each run.

## Notes on quantization

This implementation intentionally uses a simple symmetric quantization path so it
stays compatible with the custom GPT-2 implementation in this repository.
It is a practical approximation for extension analysis and keeps QLoRA vs LoftQ
comparison fair by sharing the same quantizer.
