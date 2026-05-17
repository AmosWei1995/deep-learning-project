# LoRA Sentiment 实验方案（Ziran）

**任务**：SST 情感分类（5分类）  
**方法**：在 GPT-2 注意力层的 Q/K/V 上应用 LoRA，对比三种初始化方式 × 三种秩  
**脚本**：`python3 run_experiment.py --task sentiment --use_gpu`

---

## 实验目标

1. 验证 LoRA 在情感分类任务上是否有效（对比 Section 6.3 的 full-model baseline）
2. 找到最优的 rank 取值（r = 4 / 8 / 16）
3. 对比三种初始化方式（LoRA / PiSSA / LoftQ）对收敛速度和最终精度的影响

---

## 固定配置

| 参数 | 值 | 说明 |
|---|---|---|
| seed | 42 | 全员统一，保证可复现 |
| epochs | 10 | 与 Section 6.3 保持一致 |
| lr | 1e-5 | LoRA 推荐学习率，比全量微调低一个量级 |
| batch_size | 8 | 受限于本地显存 |
| model | gpt2 (Small) | d=768，12层，12头 |
| alpha | 16.0 | 缩放系数，全组固定 |
| target_modules | query, key, value | attention.py 中的三个投影层 |
| dataset | SST（5分类） | train: 8544 / dev: 1101 |
| optimizer | AdamW（自实现） | Section 5.3 |

---

## 变量网格（9 组实验）

| 组别 | init_method | rank | 可训练参数量 |
|---|---|---|---|
| 1 | lora | 4 | ~6K × 3层 × 12块 = ~216K |
| 2 | lora | 8 | ~12K × 3 × 12 = ~432K |
| 3 | lora | 16 | ~24K × 3 × 12 = ~864K |
| 4 | pissa | 4 | 同上（参数量相同，初始化不同） |
| 5 | pissa | 8 | 同上 |
| 6 | pissa | 16 | 同上 |
| 7 | loftq | 4 | 同上 |
| 8 | loftq | 8 | 同上 |
| 9 | loftq | 16 | 同上 |

---

## 对照组

| 名称 | 来源 | dev_acc |
|---|---|---|
| last-linear-layer | Section 6.3 | 0.472 |
| full-model | Section 6.3 | 0.516 |

LoRA 的目标：**可训练参数远少于 full-model，精度接近或超过 full-model**。

---

## 评估指标

| 指标 | 含义 | 记录位置 |
|---|---|---|
| dev_acc | dev 集准确率（主指标） | `predictions/results.json` |
| train_loss | 最佳 epoch 的训练 loss | `predictions/results.json` |
| dev_loss | 最佳 epoch 的 dev loss | `predictions/results.json` |
| 收敛曲线 | 每 epoch 的 train_loss / dev_acc | 终端输出，手动记录到报告 |

---

## 输出文件

```
checkpoints/lora/sentiment_{init_method}_rank{rank}.pt   ← 最佳 epoch 权重（9个）
predictions/lora/sentiment_{init_method}_rank{rank}_dev.csv  ← dev 预测（9个）
predictions/results.json                                 ← 汇总数字
```

---

## 运行步骤

**本地验证（功能检查）**
```bash
python3 run_experiment.py --task sentiment --smoke
```
预期：10 epoch 内 dev_acc 从随机水平（~0.2）上升，loss 下降，说明梯度流通正常。

**远程 GPU 正式实验**
```bash
python3 run_experiment.py --task sentiment --use_gpu
```
预期运行时间：~2 小时（9 组 × 10 epoch，单 A100）

---

## 分析计划

实验跑完后在报告中回答以下问题：

1. **rank 的影响**：rank 越大精度是否单调上升？还是存在最优值？
2. **初始化的影响**：PiSSA / LoftQ 相比标准 LoRA 是否收敛更快（前几个 epoch 的 dev_acc 更高）？
3. **参数效率**：LoRA 用不到 1% 的可训练参数，能达到 full-model 的多少比例的精度？
4. **过拟合风险**：train_loss 和 dev_loss 是否出现分叉？小 rank 是否比大 rank 更稳定？
