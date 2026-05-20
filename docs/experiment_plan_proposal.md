# 实验方案：Paraphrase 任务 PEFT 对比（Proposal 对齐版）

**任务**：Quora 问句对 cloze-style 释义检测（paraphrase detection）  
**模型**：GPT-2 Small（d=768，12 层，12 头）  
**脚本入口**：`python3 run_experiment.py --task <ablation|exp2|exp3> --use_gpu`  
**截止**：2026-05-22 | **Feature Freeze**：2026-05-20

---

## 0. 前置：实现层（Implementation）

在跑任何实验之前，以下四项须全部就绪并通过冒烟测试。

| 组件 | 文件 | 现状 | 负责 |
|------|------|------|------|
| Cloze-style paraphrase pipeline | `paraphrase_detection.py` + `run_experiment.py --task paraphrase` | `forward()` 已实现，`--task paraphrase` 入口**待接入** | Ziyuan |
| LoRA from scratch | `modules/lora_linear.py` | **已完成** | Ziran |
| PiSSA from scratch | `lora_pissa.py` | **已完成** | Ziyuan |
| LoftQ from scratch | `lora_loftq.py` | **缺失，待实现** | Yuheng |
| Linear probing bound | `classifier.py --fine-tune-mode last-linear-layer`（已跑 Section 6.3） | **已有**，需在 paraphrase 上补跑 | Yuheng |
| FFT（full fine-tuning）bound | `paraphrase_detection.py`（full-model 模式）| **未在 paraphrase 上建立**，需补跑 | Yuheng |

**冒烟验收命令**（功能检查，不计入正式结果）：

```bash
# 每个入口跑 --smoke，确认 loss 下降、梯度流通
python3 run_experiment.py --task ablation   --smoke
python3 run_experiment.py --task exp2       --smoke
python3 run_experiment.py --task exp3       --smoke
```

---

## 全局固定配置

以下参数在所有实验中**不得更改**，开跑前全员锁定。

| 参数 | 值 | 说明 |
|------|----|------|
| `seed` | 11711 | 与已有 sentiment 实验一致 |
| `epochs` | 10 | 全程统一 |
| `lr` | 1e-5 | LoRA 推荐学习率 |
| `batch_size` | 8 | 显存约束 |
| `model` | gpt2 (Small) | 125M 参数 |
| `alpha` | 16.0 | 缩放系数 |
| `optimizer` | AdamW（自实现） | Section 5.3 |
| `dataset` | Quora（paraphrase） | Exp1–Exp3 均用此任务 |

---

## Bounds（基准上下界）

跑 Exp1 之前先建立两个参考点，写入 `predictions/results_paraphrase.json`。

| 模式 | 可训练参数 | 预期 dev acc | 命令 |
|------|-----------|-------------|------|
| **Linear probe**（下界） | 分类头，GPT 冻结 | TBD | `python3 paraphrase_detection.py --fine-tune-mode last-linear-layer --use_gpu` |
| **FFT**（上界） | 全部 ~125M | TBD | `python3 paraphrase_detection.py --fine-tune-mode full-model --use_gpu` |

> LoRA 系列实验的目标：**可训练参数 < 1%，dev acc 接近 FFT 上界**。

---

## Experiment 1 — LoRA Ablation → Config_opt

### 目标

穷举 rank × target_modules 网格，找到**效率–精度最优配置 Config_opt**，作为 Exp2 / Exp3 的输入。

### 变量网格（12 组）

| 组别 | rank | target_modules | 可训练参数（估算） |
|------|------|----------------|-----------------|
| 1 | 4 | Q+V | ~144K（2 proj × 12 blocks） |
| 2 | 8 | Q+V | ~288K |
| 3 | 16 | Q+V | ~576K |
| 4 | 32 | Q+V | ~1.15M |
| 5 | 4 | Q+K+V | ~216K |
| 6 | 8 | Q+K+V | ~432K |
| 7 | 16 | Q+K+V | ~864K |
| 8 | 32 | Q+K+V | ~1.73M |
| 9 | 4 | all linear（Q+K+V+out+FFN×2） | ~540K |
| 10 | 8 | all linear | ~1.08M |
| 11 | 16 | all linear | ~2.16M |
| 12 | 32 | all linear | ~4.32M |

所有组别固定：`init_method=lora`（标准初始化），其余参数见全局配置。

### 命令

```bash
# 跑全网格（需在 run_experiment.py 里实现 --task ablation）
python3 run_experiment.py --task ablation --use_gpu

# 单组调试示例
python3 run_experiment.py --task ablation --use_gpu \
    --rank 8 --target Q+K+V
```

### 输出文件

```
predictions/ablation/paraphrase_lora_rank{r}_{target}_dev.csv
predictions/results_ablation.json    ← 汇总 12 条记录
```

### Config_opt 选取规则

在 `predictions/results_ablation.json` 跑完后，按以下 Pareto 标准选点：

1. **主指标**：dev_acc 不低于 FFT 上界的 **95%**
2. **效率指标**：在满足①的条件下，选 `trainable_pct` 最小的配置
3. 若多组并列，优先 rank 较小（泛化性更好）

**选定结果记录**（跑完后填入）：

```
Config_opt:
  rank        = ___
  target      = ___
  trainable%  = ___
  dev_acc     = ___
```

---

## Experiment 2 — SVD 初始化 warm-start（全精度）

### 目标

在 **Config_opt** 配置下，用 **step-0 loss** 和 **早期收敛速度** 量化 PiSSA 相对标准 LoRA 的 warm-start 优势。

### 变量（3 组）

| 组别 | init_method | rank | target | 说明 |
|------|------------|------|--------|------|
| E2-1 | `lora` | Config_opt | Config_opt | 随机初始化基准 |
| E2-2 | `pissa` | Config_opt | Config_opt | SVD principal 初始化 |


### 额外记录指标

除全局指标外，Exp2 须额外记录以下数字：

| 指标 | 含义 | 记录方式 |
|------|------|---------|
| `loss_after_init` | 初始化后、第一次 backward 前的 dev loss（step 0） | 在训练 loop 最开头、epoch=0 第一个 batch 之前调用 `compute_dev_loss` |
| `epoch1_dev_acc` | 第 1 个 epoch 结束时的 dev_acc | 已有 `curve['dev_acc'][0]` |
| `steps_to_80pct_fft` | 达到 FFT dev_acc × 0.80 所需 step 数 | 在 loop 里按 step 粒度检查 |
| `steps_to_95pct_fft` | 达到 FFT dev_acc × 0.95 所需 step 数 | 同上 |

### 命令

```bash
python3 run_experiment.py --task exp2 --use_gpu
```

### 输出文件

```
predictions/exp2/paraphrase_{init}_rank{r}_{target}_dev.csv
predictions/results_exp2.json
```

### 分析问题

1. step-0 loss：PiSSA 是否显著低于 LoRA？（量化 warm-start 优势的直接证据）
2. epoch 1–3 的 dev_acc 曲线：PiSSA 是否更早超越 FFT × 80%？
3. 最终 epoch 10：PiSSA vs LoRA dev_acc 差距是否缩小（初始优势是否持续）？

---

## Experiment 3 — 量化下 SVD 初始化（QLoRA vs. LoftQ）

### 目标

在 **Config_opt** 配置下，对比 QLoRA 与 LoftQ 在 4-bit / 2-bit 量化下的精度退化和显存开销。

### 变量（5 组）

| 组别 | 方法 | bits | rank | target | 说明 |
|------|------|------|------|--------|------|
| E3-ref | LoRA（FP16 基准） | — | Config_opt | Config_opt | 与 E2-1 同 |
| E3-1 | QLoRA | 4 | Config_opt | Config_opt | backbone NF4 量化，LoRA adapter FP16 |
| E3-2 | QLoRA | 2 | Config_opt | Config_opt | backbone INT2 量化 |
| E3-3 | LoftQ | 4 | Config_opt | Config_opt | LoftQ 初始化 + backbone 量化 |
| E3-4 | LoftQ | 2 | Config_opt | Config_opt | LoftQ 初始化 + 2-bit backbone |

### 额外记录指标

| 指标 | 含义 | 记录方式 |
|------|------|---------|
| `peak_vram_mb` | 训练过程中 GPU 峰值显存 | `torch.cuda.max_memory_allocated()` |
| `degradation_acc` | 相对 E3-ref 的 dev_acc 下降（绝对值） | `ref_acc − quant_acc` |
| `degradation_pct` | 百分比退化 | `degradation_acc / ref_acc × 100` |
| `vram_saving_pct` | 相对 FP16 baseline 的显存节省 | `(ref_vram − quant_vram) / ref_vram × 100` |

### 命令

```bash
python3 run_experiment.py --task exp3 --use_gpu
```

### 输出文件

```
predictions/exp3/paraphrase_{method}_{bits}bit_rank{r}_{target}_dev.csv
predictions/results_exp3.json
```

### 分析问题

1. 4-bit vs 2-bit：精度退化随量化粒度的变化趋势？
2. QLoRA vs LoftQ（同 bits）：LoftQ 的 SVD 残差初始化能在多大程度上补偿量化误差？
3. 显存–精度 trade-off：哪种方案在显存 < 4 GB 时仍保持最接近 FP16 的精度？

---

## 汇总：实验设计全图

```
Bounds（paraphrase）
├── linear probe（下界）   dev_acc = TBD
└── FFT（上界）            dev_acc = TBD

Exp 1 — LoRA Ablation（12 组）
  rank ∈ {4,8,16,32} × target ∈ {Q+V, Q+K+V, all linear}
  ↓ 选 Config_opt（rank=?, target=?）

Exp 2 — SVD warm-start（3 组，全精度）
  LoRA / PiSSA / LoftQ-fp  @  Config_opt
  主指标：step-0 loss，steps_to_80/95% FFT

Exp 3 — 量化对比（5 组）
  QLoRA-4bit / QLoRA-2bit / LoftQ-4bit / LoftQ-2bit  @  Config_opt
  主指标：degradation_acc，peak_vram_mb
```

---

## 需要新增 / 修改的代码

| 文件 | 需要做什么 | 优先级 |
|------|-----------|--------|
| `lora_loftq.py` | 实现 `init_loftq`（量化 + SVD 残差补偿）；支持 `num_bits` 参数 | P0 |
| `run_experiment.py` | 实现 `--task ablation`（12 组 × paraphrase） | P0 |
| `run_experiment.py` | 实现 `--task exp2`（记录 `loss_after_init`、`steps_to_X%`） | P0 |
| `run_experiment.py` | 实现 `--task exp3`（量化训练 loop + `peak_vram_mb` / degradation） | P0 |
| `paraphrase_detection.py` | 补 `--fine-tune-mode last-linear-layer` 模式（linear probe bound） | P1 |
| `modules/lora_linear.py` | `apply_lora` 支持 `target='Q+V'` / `'Q+K+V'` / `'all'` 三种简写 | P1 |
| `run_experiment.py` | `--rank` 允许值扩展至包含 32 | P1 |

---

## 进度检查表

### Implementation
- [ ] `lora_loftq.py` 实现完成，smoke test 通过
- [ ] paraphrase linear probe 跑通，dev_acc 记录
- [ ] paraphrase FFT 跑通，dev_acc 记录（FFT 上界确认）

### Experiment 1
- [ ] `--task ablation` 实现完成
- [ ] 12 组全部跑完，写入 `predictions/results_ablation.json`
- [ ] Config_opt 选定，记录到本文档上方空白处

### Experiment 2
- [ ] `loss_after_init` 指标接入训练 loop
- [ ] `steps_to_80/95pct_fft` 接入训练 loop
- [ ] 3 组全部跑完，写入 `predictions/results_exp2.json`

### Experiment 3
- [ ] 量化 backbone 路径接入（QLoRA / LoftQ）
- [ ] `peak_vram_mb` + degradation 字段记录
- [ ] 5 组全部跑完，写入 `predictions/results_exp3.json`
