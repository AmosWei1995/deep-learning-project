# Experiment Runner

## Exp 1 — LoRA Target Module Ablation

Compares three target module sets (QV, QKV, ALL) across ranks 4 / 8 / 16 on SST and CFIMDB.
Selected config: QKV, rank 4.

```bash
# Smoke test
python run_experiment.py --task ablation --smoke

# Full run on GPU
python run_experiment.py --task ablation --use_gpu

# Single dataset or config
python run_experiment.py --task ablation --use_gpu --dataset sst
python run_experiment.py --task ablation --use_gpu --dataset sst --rank 4

# Force rerun
python run_experiment.py --task ablation --use_gpu --rerun
```

Outputs:
- `predictions/results.json` — aggregated results
- `predictions/lora/sentiment_{dataset}_{init}_rank{rank}_dev.csv`
- `predictions/lora/sentiment_{dataset}_{init}_rank{rank}_test.csv`
- `checkpoints/lora/sentiment_{dataset}_{init}_rank{rank}.pt`


## Exp 2 — LoRA vs PiSSA

Compares LoRA and PiSSA initializations across ranks 4 / 8 / 16 on SST and CFIMDB.
Target modules fixed to QKV.

```bash
# Smoke test
python run_experiment.py --task sentiment --smoke

# Full run on GPU (all configs × both datasets)
python run_experiment.py --task sentiment --use_gpu

# Single dataset
python run_experiment.py --task sentiment --use_gpu --dataset sst
python run_experiment.py --task sentiment --use_gpu --dataset cfimdb

# Single config
python run_experiment.py --task sentiment --use_gpu --dataset sst --init lora --rank 8
python run_experiment.py --task sentiment --use_gpu --dataset sst --init pissa --rank 8

# Force rerun
python run_experiment.py --task sentiment --use_gpu --rerun
```

Outputs: same paths as Exp 1.


## Exp 3 — Quantized LoRA (QLoRA vs LoftQ)

Compares QLoRA and LoftQ under 4-bit / 2-bit quantization on SST.
Reports accuracy degradation vs. full-precision LoRA baseline (dev acc 0.5186, rank 4) and peak VRAM.

Grid: method ∈ {qlora, loftq}, bits ∈ {4, 2}, rank 4.

```bash
# Requires DLP_ROOT to be set or passed via --dlp-root
DLP_ROOT=$(pwd) python extensions/exp3_quant_lora/run_exp3.py --use_gpu
python extensions/exp3_quant_lora/run_exp3.py --use_gpu --method qlora --bits 4 --rank 4 --dlp-root $(pwd)
python extensions/exp3_quant_lora/run_exp3.py --use_gpu --method loftq --bits 2 --rank 4 --dlp-root $(pwd)
python extensions/exp3_quant_lora/run_exp3.py --smoke --dlp-root $(pwd)
```

Useful flags: `--smoke` (2 epochs, tiny data), `--rerun` (ignore cache), `--rank N`, `--alpha F`, `--baseline-dev-acc F`.

Outputs:
- `extensions/exp3_quant_lora/results/exp3_results.json`
- `extensions/exp3_quant_lora/results/preds/sentiment_sst_{method}_{bits}bit_rank{rank}_dev.csv`
- `extensions/exp3_quant_lora/results/preds/sentiment_sst_{method}_{bits}bit_rank{rank}_test.csv`
