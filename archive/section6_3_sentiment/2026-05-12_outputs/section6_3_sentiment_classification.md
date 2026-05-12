# Section 6.3 — Sentiment classification (GPT-2)

SST (5-way) and CFIMDB (binary): train with **last-linear-layer** (frozen backbone) or **full-model**; use train + dev only for tuning and dev metrics. Each `classifier.py` run trains **SST then CFIMDB**; CFIMDB uses **batch size 8** in code regardless of `--batch_size`. Predictions go under `predictions/`; checkpoints under `checkpoints/<mode>-{sst,cfimdb}-classifier.pt`.

## Commands used

```bash
python3 classifier.py --fine-tune-mode last-linear-layer --use_gpu
python3 classifier.py --fine-tune-mode full-model --use_gpu --batch_size=80
```

Other flags (`lr`, `epochs`, `hidden_dropout_prob`, `seed`, …) follow **`classifier.py` defaults** for both runs.

## Dev accuracy vs handout baselines

| Mode | Dataset | Baseline | Measured (this repo’s `predictions/*-dev-out.csv`) |
|------|---------|----------|------------------------------------------------------|
| last-linear-layer | SST | 0.462 | 0.472 |
| full-model | SST | 0.513 | 0.516 |
| last-linear-layer | CFIMDB | 0.861 | 0.869 |
| full-model | CFIMDB | 0.976 | 0.976 |

Recompute after retraining, e.g.  
`python3 eval_sentiment_predictions.py --pred predictions/full-model-sst-dev-out.csv --dev data/ids-sst-dev.csv` (swap `full-model` / `last-linear-layer` and `sst` / `cfimdb` as needed).

## Output files (eight)

`predictions/{last-linear-layer,full-model}-{sst,cfimdb}-{dev,test}-out.csv`

Archived snapshot (same CSVs + this note): `archive/section6_3_sentiment/2026-05-12_outputs/`.
