# CS 224N GPT-2 Project Schedule
**Team**: Ziran / Ziyuan / Yuheng  
**开始**: May 10, 2026 | **Deadline**: May 22, 2026

---

## 官方结构对应

| 官方章节 | 内容 | 验收脚本 |
|---|---|---|
| Section 5 | GPT-2 实现（attention / gpt2_layer / gpt2.embed / optimizer） | `optimizer_test.py` + `sanity_check.py` |
| Section 6 | 情感分类（classifier.py，两个模式） | `classifier.py` × 2 |
| Section 7 | Extensions：paraphrase + sonnet + LoRA 等 | `paraphrase_detection.py` + `sonnet_generation.py` |

---

## 原则

- 三人并行写代码（接口已在 `interfaces.md` 约定），依赖只在联调时体现
- Section 5+6（Part 1）必须全部完成再进入 Section 7（Part 2）
- paraphrase + sonnet baseline 是 Part 2 的必做项，LoRA 是其上的 extension
- **20 日 feature freeze，之后只允许 bugfix**

---

## Phase 1：Section 5 实现（5月10日 – 5月12日上午）

> 目标：四个核心函数写完，`optimizer_test.py` + `sanity_check.py` 通过

**报告：不碰。**

| 日期 | 任务 | 负责 |
|------|------|------|
| 5/10 | 阅读 `interfaces.md`，对齐接口约定；`source setup.sh` 配环境 | 全员 |
| 5/11 | `CausalSelfAttention.attention` / `AdamW.step`+`GPT2Layer.add`+`GPT2Layer.forward` / `GPT2Model.embed`+`hidden_state_to_token` 同时写完 | Ziran / Ziyuan / Yuheng |
| 5/12 上午 | 联调：`python3 optimizer_test.py` → `python3 sanity_check.py` | 全员 |

**⚑ 硬节点：5月12日上午，sanity_check 通过。**

---

## Phase 2：Section 6 — 情感分类（5月12日下午）

> 目标：classifier.py 两个模式跑通，8 个 CSV 生成，达到 baseline 精度

| 任务 | 说明 | 负责 |
|------|------|------|
| `GPT2SentimentClassifier.__init__` + `forward` | last-linear-layer 和 full-model 两个模式 | Yuheng |
| 跑训练 | `classifier.py --fine-tune-mode last-linear-layer`（SST + CFIMDB） | 全员 |
| 跑训练 | `classifier.py --fine-tune-mode full-model`（SST + CFIMDB） | 全员 |
| 验收精度 | SST last-linear: ≥0.462 / full: ≥0.513；CFIMDB last-linear: ≥0.861 / full: ≥0.976 | 全员 |

**⚑ 硬节点：5月12日结束，Part 1（Section 5+6）全部完成。**

---

## Phase 3：Section 7 Baseline — Paraphrase + Sonnet（5月13日）

> 目标：两个 baseline forward() 实现，跑通训练，首次提交 Leaderboard dev

**Part 2 必须在 Part 1 基础上做，不可跳步。**

| 任务 | 说明 | 负责 |
|------|------|------|
| `ParaphraseGPT.forward` | cloze-style，取 last_token → Linear(d,2) | Yuheng |
| `SonnetGPT.forward` | hidden_state_to_token → (B,T,V) logits | Yuheng |
| 跑 paraphrase baseline | `python3 paraphrase_detection.py --use_gpu` | 全员 |
| 跑 sonnet baseline | `python3 sonnet_generation.py --use_gpu` | 全员 |

**⚑ 硬节点：5月13日，paraphrase + sonnet baseline 跑通。**

---

## Phase 4：Section 7 Extension — LoRA / PiSSA / LoftQ（5月13日 – 5月18日）

> 目标：三种 PEFT 实现完成，9 组实验数据齐备；报告方法章节完成

**分人负责，互不干扰：**

| 日期 | 代码任务 | 报告任务 | 负责 |
|------|----------|----------|------|
| 5/13–14 | `LoRALinear` + `apply_lora` + 三种 init 函数框架 | 报告提纲 + related work | 代码：各自 / 报告：全员 |
| 5/14–16 | 集成进 attention 层，验证 loss 正常下降 | 方法章节：GPT-2 架构、三种方法原理 | 代码：各自 / 报告：全员 |
| 5/16–17 | 跑对比实验（rank∈{4,8,16} × 3方法 = 9组）× paraphrase + sonnet 两个任务 | 方法章节：实验设置 | Ziran（sentiment）/ Ziyuan（ablation）/ Yuheng（generation+paraphrase）|
| 5/17–18 | 整理结果，绘制对比图表 | — | 全员 |

> 实验配置开跑前全员锁定（seed=42），不在跑实验期间调参。

**⚑ 硬节点：5月18日，实验数据齐备，Leaderboard test 已提交。**

---

## Phase 5：收尾（5月18日 – 5月22日）

| 日期 | 任务 | 负责 |
|------|------|------|
| 5/18–19 | 报告结果章节 + 分析章节 | 全员 |
| 5/19–20 | 报告润色：摘要、结论、格式 | 全员 |
| 5/19–20 | 代码整理：注释、README | Ziyuan |
| **5/20** | **Feature freeze — 之后只允许 bugfix** | 全员 |
| 5/20–21 | 讲稿 + 幻灯片 | 全员 |
| 5/22 上午 | 录制 pre 视频、剪辑 | 全员 |
| **5/22** | **提交截止** | 全员 |

---

## 分工

| 函数 / 模块 | 负责人 | 所属 | 知识点 |
|-------------|--------|------|--------|
| `CausalSelfAttention.attention()` | Ziran | Section 5 | attention 计算核心 |
| `AdamW.step()` | Ziyuan | Section 5 | 优化器数学原理 |
| `GPT2Layer.add()` + `GPT2Layer.forward()` | Ziyuan | Section 5 | transformer block 结构 |
| `GPT2Model.embed()` + `hidden_state_to_token()` | Yuheng | Section 5 | embedding + LM head |
| `GPT2SentimentClassifier`（两个模式） | Yuheng | Section 6 | 生成模型→分类模型 |
| `ParaphraseGPT.forward()` | Ziyuan | Section 7 | cloze-style 任务设计 |
| `SonnetGPT.forward()` + `generate()` 优化 | Ziran | Section 7 | 解码策略（beam/top-k）|
| `lora_linear.py`（LoRALinear + apply_lora + init_lora） | Ziran | Section 7 Extension | LoRA 基础原理 |
| `lora_pissa.py`（init_pissa） | Ziyuan | Section 7 Extension | SVD 初始化 |
| `lora_loftq.py`（init_loftq） | Yuheng | Section 7 Extension | 量化感知初始化 |
| 情感分类实验 + 图表分析 | Ziran | Section 7 Extension | — |
| LoRA ablation study | Ziyuan | Section 7 Extension | — |
| paraphrase + sonnet 实验 | Yuheng | Section 7 Extension | — |
| 报告主笔 | 全员 | — | — |
| 视频录制 | 全员 | — | — |

---

## 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| sanity_check 12日上午未过 | 中 | 遇到 shape bug 立刻全员看；注意函数名是 `.attention`/`.add`/`.embed` 不是 `.forward` |
| classifier 精度未达 baseline | 低 | 检查 last_token 取法；确认两个模式的参数冻结逻辑 |
| paraphrase/sonnet 13日未跑通 | 低 | forward() 极简，依赖 `gpt.hidden_state_to_token` 已实现 |
| lora_pissa/loftq 依赖 lora_linear | 低 | Ziran 在 Phase 3 第一天优先完成基类，通知另外两人再开始 |
| 实验数据 18日前未齐 | 低 | 提前锁定配置，每组实验 < 10 分钟，9组 2 小时内跑完 |
