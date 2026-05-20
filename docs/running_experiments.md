# 实验运行说明

**项目根目录**：`deep-learning-project/`  
**Python 环境**：`cs224n_dfp`（由 `setup.sh` / `env.yml` 创建）

---

## 0. 环境准备

首次使用：

```bash
source setup.sh          # 创建 conda 环境
```

每次运行前：

```bash
conda activate cs224n_dfp
cd /path/to/deep-learning-project
```

---

## 输出目录结构

所有实验结果统一写入 `experiments/`，正式结果与 smoke 结果完全隔离：

```
experiments/
├── baselines/
│   ├── results_baselines.json     ← 基准汇总（last-linear-layer / full-model）
│   ├── predictions/               ← 基准预测 CSV
│   └── smoke/
│       ├── results_baselines_smoke.json
│       └── predictions/
├── exp1/
│   ├── results_exp1.json          ← Exp1 正式汇总（12 组）
│   ├── predictions/               ← Exp1 正式预测 CSV
│   └── smoke/
│       ├── results_exp1_smoke.json
│       └── predictions/
├── exp2/
│   ├── results_exp2.json          ← Exp2 正式汇总（lora / pissa）
│   ├── predictions/
│   └── smoke/
│       ├── results_exp2_smoke.json
│       └── predictions/
└── exp3/
    ├── results_exp3.json          ← Exp3 正式汇总（qlora / loftq × 4-bit / 2-bit）
    ├── results_exp3_smoke.json
    └── predictions/
```

---

## Baselines — 基准上下界

**目的**：建立 Exp1 消融结果的比较基准，**两个数据集（SST + CFIMDB）各跑一次**：  
- `last-linear-layer`（线性探测，下界）：冻结 GPT 主干，仅训练分类头  
- `full-model`（全量微调 FFT，上界）：训练所有参数

### 冒烟（本地 CPU，~2 分钟）

```bash
python3 run_exp1_exp2.py --task baselines --smoke
```

### 正式运行（GPU，~40 分钟）

```bash
# 默认跑 SST + CFIMDB 共 4 组
python3 run_exp1_exp2.py --task baselines --use_gpu

# 只跑单个数据集
python3 run_exp1_exp2.py --task baselines --use_gpu --dataset sst
python3 run_exp1_exp2.py --task baselines --use_gpu --dataset cfimdb
```

跑完后终端打印汇总：

```
============================================================
BASELINES SUMMARY
  last-linear-layer          dev_acc=0.4xxx  trainable=0.00%   (sst)
  full-model                 dev_acc=0.5xxx  trainable=100.00% (sst)
  last-linear-layer          dev_acc=0.8xxx  trainable=0.00%   (cfimdb)
  full-model                 dev_acc=0.9xxx  trainable=100.00% (cfimdb)
============================================================
```

### 关键输出字段（`results_baselines.json`）

| 字段 | 含义 |
|------|------|
| `task` | 数据集名（`sst` / `cfimdb`） |
| `fine_tune_mode` | `last-linear-layer` 或 `full-model` |
| `dev_acc` | dev 集准确率（主指标，用作 Exp1 报告对照） |
| `trainable_pct` | 可训练参数占比（linear-probe ≈ 0%，FFT = 100%） |

> **与 Exp2 的关系**：`full-model` 的 `dev_acc` 即为对应数据集的 FFT 上界，Exp2 默认值已内置（SST: 0.516，CFIMDB: 0.976）。

---

## Exp1 — LoRA 消融实验

**目的**：在 rank ∈ {4, 8, 16, 32} × target ∈ {QV, QKV, ALL} 共 12 组配置上搜索最优效率–精度点 Config_opt，**两个数据集各跑一遍**（共 24 组）。

### 冒烟（本地 CPU，~10 分钟）

```bash
# 默认同时跑 SST + CFIMDB（各 3 组 × rank=4 × 1 epoch）
python3 run_exp1_exp2.py --task exp1 --smoke

# 只验证单个数据集
python3 run_exp1_exp2.py --task exp1 --smoke --dataset sst
```

验收标准：每个数据集各 3 组（QV / QKV / ALL），无报错，终端打印
`[smoke] sanity assertions passed ✓`。

### 正式运行（GPU，~4–6 小时）

```bash
# 默认 SST + CFIMDB 共 24 组
python3 run_exp1_exp2.py --task exp1 --use_gpu

# 只跑单个数据集
python3 run_exp1_exp2.py --task exp1 --use_gpu --dataset cfimdb
```

跑完后终端自动为每个数据集分别打印 **Config_opt**：

```
[SST]
============================================================
CONFIG_OPT (Pareto: dev_acc >= 95% best, fewest params)
  rank    = 8  target  = QKV  dev_acc = 0.52xx  trainable = 0.36%
============================================================

[CFIMDB]
============================================================
CONFIG_OPT ...
============================================================
```

---

## Exp2 — SVD 初始化 warm-start 对比

**目的**：在每个数据集的 Config_opt 配置下，用 `loss_after_init`（step-0 loss）和 `steps_to_80/95%_fft` 量化 PiSSA 相对 LoRA 的热启动优势，**SST + CFIMDB 各跑一遍**。

FFT 上界默认值内置：SST = 0.516，CFIMDB = 0.976（来自 Section 6.3）。

### 冒烟（本地 CPU，~5 分钟）

**先跑 exp1 smoke**，exp2 smoke 会自动从 exp1 smoke 结果里按数据集推断 Config_opt：

```bash
# Step 1：先跑 exp1 smoke
python3 run_exp1_exp2.py --task exp1 --smoke

# Step 2：再跑 exp2 smoke（per-dataset Config_opt + FFT 上界自动加载）
python3 run_exp1_exp2.py --task exp2 --smoke
```

如果只想跑单个数据集，或手动指定 Config_opt：

```bash
python3 run_exp1_exp2.py --task exp2 --smoke --dataset sst \
    --configopt-rank 4 --configopt-target QKV
```

### 正式运行（GPU，~80–120 分钟）

**前提**：Exp1 正式结果已生成（`experiments/exp1/results_exp1.json` 存在）。

Exp2 默认跑 **20 epochs**（比 Exp1 多 10 epoch），确保 PiSSA 能收敛并显现热启动优势。

```bash
# 默认 SST + CFIMDB，Config_opt 从 Exp1 结果自动推断
python3 run_exp1_exp2.py --task exp2 --use_gpu

# 只跑单个数据集
python3 run_exp1_exp2.py --task exp2 --use_gpu --dataset cfimdb
```

> **`--fft-dev-acc`**：覆盖 FFT 上界（影响所有数据集）。通常无需手动指定，默认值已内置（SST: 0.516，CFIMDB: 0.976）。

### 关键输出字段（`results_exp2.json`）

| 字段 | 含义 |
|------|------|
| `loss_after_init` | 初始化后、第一次梯度前的 dev loss |
| `steps_to_80pct_fft` | 首次达到 FFT × 80% dev_acc 所需 step 数 |
| `steps_to_95pct_fft` | 首次达到 FFT × 95% dev_acc 所需 step 数 |
| `steps_per_epoch` | 每 epoch 的 step 数（用于换算 epoch） |

---

## Exp3 — 量化对比（QLoRA vs LoftQ）

**目的**：在 Config_opt 配置下，对比 QLoRA 与 LoftQ 在 4-bit / 2-bit 量化下的精度退化和峰值显存。

**前提**：`predictions/results.json` 中存在 SST LoRA 基准结果（已有，用于自动读取 Config_opt）。

### 冒烟（本地 CPU，~5 分钟）

```bash
python3 run_experiment.py --task exp3_quant --smoke
```

### 正式运行（GPU，~30–40 分钟）

```bash
python3 run_experiment.py --task exp3_quant --use_gpu
```

只跑某一种方法或 bit 数：

```bash
python3 run_experiment.py --task exp3_quant --use_gpu --exp3_method loftq
python3 run_experiment.py --task exp3_quant --use_gpu --exp3_bits 4
```

### 关键输出字段（`results_exp3.json`）

| 字段 | 含义 |
|------|------|
| `peak_vram_gb` | 训练峰值显存（GB） |
| `degradation` | 相对 FP16 基准的 dev_acc 下降（`baseline - quant`，正数 = 退化） |
| `degradation_pct` | 退化百分比 |

---

## 通用选项

| 选项 | 适用 | 说明 |
|------|------|------|
| `--smoke` | 全部 | 小数据（64 train / 32 dev）+ 1 epoch，快速验证管道 |
| `--use_gpu` | 全部 | 启用 GPU；不加则使用 CPU（仅 smoke 可接受） |
| `--dataset sst\|cfimdb` | Baselines/Exp1/Exp2 | 只跑指定数据集（默认两个都跑） |
| `--rerun` | Baselines/Exp1/Exp2 | 忽略已有结果，从头重跑所有配置 |
| `--epochs N` | Baselines/Exp1/Exp2 | 覆盖训练轮数（smoke 模式固定 1 epoch，忽略此选项） |
| `--batch_size N` | Baselines/Exp1/Exp2/Exp3 | 覆盖 batch size（默认 8） |

---

## 中断续跑

每完成一组实验，结果**立即追加写盘**。中断后直接重跑同一条命令，已完成的组自动跳过：

```bash
# 例：Exp1 中途中断，重跑会从上次断点继续
python3 run_exp1_exp2.py --task exp1 --use_gpu
```

---

## 推荐完整运行顺序

```
Step 0  smoke baselines      验证 baselines 管道（SST + CFIMDB × 2 modes × 1 epoch）
Step 1  smoke exp1           验证 Exp1 管道（SST + CFIMDB × 3 targets × 1 epoch）
Step 2  smoke exp2           验证 Exp2 管道（SST + CFIMDB × lora + pissa × 1 epoch）
Step 3  正式 baselines        跑完 4 组基准（两数据集 × 下界/上界）
Step 4  正式 exp1 --use_gpu   跑完 24 组，各数据集确认 Config_opt
Step 5  正式 exp2 --use_gpu   lora vs pissa @ Config_opt（每数据集）
Step 6  exp3_quant --use_gpu  qlora vs loftq @ 4-bit / 2-bit（SST）
```
