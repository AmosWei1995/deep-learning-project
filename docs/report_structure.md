# Building and Extending GPT-2: Initialisation Strategy Analysis for Parameter-Efficient Fine-Tuning

**Group 41**: Ziran Wei, Yuheng Qiao, Ziyuan Guo
**Target Grade**: A
**Project Type**: Default Project 3b + Extension

---

## Abstract (摘要)
* **研究任务**：本项目基于自研代码库从零实现 GPT-2 核心模块，并在 Stanford Sentiment Treebank (SST)、CFIMDB、Quora Question Pairs (QQP) 释义检测及莎士比亚十四行诗文本生成任务上进行参数高效微调（PEFT）评估。
* **核心方法**：针对传统 LoRA 盲初始化的局限性，实现了基于奇异值分解（SVD）的 PiSSA（主成分初始化）与 LoftQ（量化残差基础上的 SVD 补偿初始化）策略。
* **主要目标**：在全精度与低比特（4-bit/2-bit）量化设置下，定量对比各模型变体的收敛速度、Step-0 初始损失、准确率退化（degradation）及峰值显存（Peak VRAM）占用，考察 SVD 初始化是否带来可测量的热启动优势。

---

## 1. Introduction (引言) [权重: 10%]
* **背景与痛点**：大型自回归语言模型全量微调（FFT）面临极高的计算与显存挑战，参数高效微调（PEFT）已成为低资源环境下模型适配的必然选择。
* **核心研究问题**：探讨旁路低秩矩阵的不同**初始化策略**（如基于 SVD 或量化残差补偿）能否显著改善 GPT-2 在下游特定任务上的优化动力学、提升热启动（Warm-start）效率并增强抗量化噪声的能力。
* **研究贡献**：声明所有核心 Transformer 组件、优化器及三种 PEFT 变体（LoRA, PiSSA, LoftQ）均由团队独立手写重构，不依赖第三方 PEFT 库；通过严密的控制变量实验确立了最佳效率-精度边界。

---

## 2. Related Work (相关工作) [权重: 10%]
* **GPT-2 语言模型**：分析自回归 Transformer 架构作为无监督多任务学习器的理论基础（Radford et al., 2019）。
* **低秩适配技术 (LoRA)**：阐述冻结预训练权重、通过内在低秩空间 $\Delta W = \frac{\alpha}{r}(B \cdot A)$ 逼近参数更新并大幅削减可训练参数量的经典机制（Hu et al., 2021）。
* **主成分适配 (PiSSA)**：探讨利用 SVD 提取原权重矩阵主奇异值和奇异向量初始化旁路、进而直击核心优化通道并加速收敛的原理（Meng et al., 2024）。
* **量化感知低秩适配 (LoftQ)**：剖析在高位权重低比特（2-bit/4-bit）量化时，利用低秩矩阵交替迭代逼近并补偿量化误差残差的数学逻辑（Li et al., 2024）。

---

## 3. Data (数据) [权重: 10%]
* **情感分类数据集 (SST & CFIMDB)**：
    * SST 包含训练集 8,544、验证集 1,101、测试集 2,210，用于评估情感分类基准。
    * CFIMDB 包含训练集 1,701、验证集 245、测试集 488，用于多文本模式的情感分类验证。
* **释义检测数据集 (Quora Question Pairs - QQP)**：
    * 包含训练集 141,506、验证集 20,215、测试集 40,431。
    * 由于官方测试集为 label-free（无标签），依据实验规范将验证集准确率（Dev Accuracy）确立为主要评测指标，同时报告 macro-F1 以衡量类别不平衡下的综合表现。
* **文本生成数据集 (Shakespeare Sonnets)**：
    * 包含 143 首训练集和 12 首测试集。
    * 开发阶段引入预留验证集（`sonnets_held_out_dev.txt`）作为参考，使用字符级 n-gram F-score (chrF) 进行生成质量的自动评测。

---

## 4. Methods (方法) [权重: 30%]
* **自研底座基础架构实现 (From Scratch)**：
    * 详细推导因果自注意力（Causal Self-Attention）中因果掩码（`torch.triu` 上三角处理）与 Padding 掩码的软硬件结合实现。
    * 阐述前馈网络（FFN）中 GeLU 激活函数映射与 Pre-LN 残差连接（`GPT2Layer.forward`）的结构拓扑。
    * 写出高效解耦权重衰减（Decoupled Weight Decay）的 AdamW 优化器单步更新迭代算法。
* **PEFT 初始化算法的自研重构**：
    * **LoRA**：矩阵 $A \sim \mathcal{N}(0, \sigma^2)$ 高斯随机化，矩阵 $B = 0$，保证第 0 步输出无偏，完全不破坏原生表征。
    * **PiSSA**：实现对原权重 $W$ 的快速奇异值分解（SVD），将前 $r$ 个主成分赋予 $A$ 和 $B$，同时将原权重矩阵动态修改为残差底座 $W_{frozen} = W - B \cdot A$（冻结），低秩分支从非零热点出发开始训练 (Meng et al., 2024)。
    * **LoftQ**：在量化骨干权重 $Q(W)$ 的基础上，对残差 $W - Q(W)$ 进行 SVD 分解初始化低秩补偿对，使 LoRA adapter 从量化误差最大的方向开始修正 (Li et al., 2024)。

---

## 5. Experiments (实验) [权重: 30%]
* **性能边界确立 (Bounds)**：
    * **下界 (Linear Probing)**：完全冻结 GPT-2 Small 原生权重，仅微调最后一层线性分类头在 QQP 上的表现，借此隔离底座原生表征能力。
    * **上界 (Full Fine-Tuning, FFT)**：125M 全量参数参与更新时的表征极限。
    * **文本生成基准**：未微调的原生 GPT-2 Small 在莎士比亚诗歌上的零样本生成表现。
* **Experiment 1：LoRA 架构网格消融与最优选点 (LoRA Ablation)**：
    * 展示秩大小 $r \in \{4, 8, 16, 32\}$ 与作用目标模块（$Q+V$、$Q+K+V$、All Linear）交叉组合的 12 组网格实验结果。
    * 基于 Pareto 效率-精度前沿规则（Dev Acc $\ge$ 95% FFT 且训练参数占比最小），明确推导并锁定最优配置 **$Config_{opt}$**。
* **Experiment 2：全精度下 SVD 初始化的热启动验证 (PiSSA vs. LoRA)**：
    * 在固定 $Config_{opt}$ 下，严格控制变量对比标准 LoRA 与 PiSSA 的收敛轨迹。
    * 引入 **`loss_after_init`（Step-0 初始损失）** 指标，定量描述两种初始化方式在训练开始前的损失差异，以数据而非先验判断 PiSSA 热启动效果是否成立。
    * 统计两方案达到 FFT 性能的 80% 和 95% 分别所需的具体训练步数（**`steps_to_80pct_fft` / `steps_to_95pct_fft`**），量化收敛加速比。
* **Experiment 3：极低比特量化下的误差补偿对抗 (QLoRA vs. LoftQ)**：
    * 在 4-bit 与 2-bit 量化骨干网络下，横向对比传统 QLoRA 与 LoftQ 的适配能力。
    * 追踪并报告系统运行时的真实/理论峰值显存（**`peak_vram_mb`**）及显存节省百分比。
    * 定量分析量化带来的绝对精度退化（**`degradation_acc`**），论证 LoftQ 的交替量化残差机制在极低比特（2-bit）下防止模型性能崩溃的决定性作用。
* **定性评估与错误分析 (Qualitative Analysis)**：
    * **分类错误剖析**：对 QQP 验证集中的典型误判（如语义相同但句式大幅倒装、或关键词相同但逻辑取反的句子）进行分类统计（Error Categorization）。
    * **生成文本对比**：横向抽样对比未微调原生底座、标准微调及各 PEFT 变体生成的莎士比亚十四行诗，定性探讨解码多样性与文本重复率。

---

## 6. Conclusion (结论) [权重: 5%]
* **核心结论归纳**：总结奇异值分解（SVD）介入 PEFT 初始化在提升微调效率、加速早期收敛及抵抗极端低比特量化噪声方面的核心科学发现。
* **局限性反思**：客观指出实验受限于手写量化算子的模拟性质（Fake Quantization 内存行为表现）以及特定单任务闭环的局限性。
* **未来展望**：提出将 PiSSA 与 LoftQ 思想进一步推广至最新微调优化器（如 Muon），或在生成任务中结合更复杂的高阶解码策略（如带押韵与格律约束的 Beam Search）的潜在可行性。

---

## References (参考文献)
* [1] Radford et al. (2019) Language models are unsupervised multitask learners. OpenAI Blog.
* [2] Hu et al. (2021) LoRA: Low-rank adaptation of large language models. ICLR 2022.
* [3] Popović (2015) chrF: character n-gram F-score for automatic MT evaluation. WMT 2015.
* [4] F. Meng, Z. Wang, and M. Zhang (2024) PiSSA: Principal singular values and singular vectors adaptation of large language models. NeurIPS 2024.
* [5] Y. Li, Y. Yu, C. Liang, P. He, N. Karampatziakis, W. Chen, and T. Zhao (2024) LoftQ: LoRA-fine-tuning-aware quantization for large language models. ICLR 2024.