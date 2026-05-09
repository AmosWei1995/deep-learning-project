# CS 224N GPT-2 Project — Interface Contracts

> 严格对应官方 handout 的 Section 5+6（Part 1）和 Section 7（Part 2）结构。
> 函数名以官方 handout 为准。

---

## 全局 Shape 约定

来源：`config.py` 的 `GPT2Config` + 各脚本的 `add_arguments()`。

| 符号 | 含义 | gpt2 (Small) | gpt2-medium | gpt2-large |
|---|---|---|---|---|
| `B` | batch size | 由 `--batch_size` 决定 | 同左 | 同左 |
| `T` | sequence length | 最大 **1024** | 同左 | 同左 |
| `d` | hidden dimension | **768** | 1024 | 1280 |
| `H` | number of attention heads | **12** | 16 | 20 |
| `d_head` | d / H | **64** | 64 | 64 |
| `d_ffn` | FFN inner dim = 4 × d | **3072** | 4096 | 5120 |
| `V` | vocabulary size | **50257** | 同左 | 同左 |

> 用 `d // num_heads` 计算 `d_head`，不要硬编码 64。

---

# Part 1：Section 5 + Section 6

验收顺序：`optimizer_test.py` → `sanity_check.py` → `classifier.py`（两个模式）

---

## 1. `optimizer.py` — `AdamW.step()`

**负责人**：Ziyuan  
**来源**：Section 5.3

```python
class AdamW(Optimizer):
    def step(self, closure: Callable = None) -> Optional[float]:
        """
        实现 efficient version（handout Algorithm 1 末尾两行替换版）：
          αt = α * sqrt(1 - β2^t) / (1 - β1^t)
          θt = θt-1 - αt * mt / (sqrt(vt) + ε)

        步骤：
          1. 初始化 state：step=0, exp_avg（mt）, exp_avg_sq（vt）
          2. step += 1
          3. mt = β1 * mt-1 + (1 - β1) * grad
          4. vt = β2 * vt-1 + (1 - β2) * grad²
          5. αt = lr * sqrt(1 - β2^t) / (1 - β1^t)
          6. θ = θ - αt * mt / (sqrt(vt) + ε)
          7. weight decay（decoupled）：θ = θ * (1 - lr * weight_decay)

        Returns:
            loss (float | None)
        """
```

验证：`python3 optimizer_test.py`

---

## 2. `modules/attention.py` — `CausalSelfAttention.attention()`

**负责人**：Ziran  
**来源**：Section 5.2（函数名：`attention.CausalSelfAttention.attention`）

```python
class CausalSelfAttention(nn.Module):
    def attention(
        self,
        query: torch.Tensor,                # (B, H, T, d_head)
        key: torch.Tensor,                  # (B, H, T, d_head)
        value: torch.Tensor,                # (B, H, T, d_head)
        attention_mask: torch.Tensor,       # (B, T)，1=有效，0=padding
    ) -> torch.Tensor:
        """
        Masked multi-head scaled dot-product attention（handout Eq.1）：
          Attention(Q,K,V) = Softmax(QK^T / sqrt(dk)) * V

        步骤：
          1. scores = Q @ K^T / sqrt(d_head)               # (B, H, T, T)
          2. 应用 causal mask（upper-triangular，torch.triu）
          3. 应用 attention_mask（padding 位置填 -inf）
          4. softmax → weights
          5. output = weights @ V                           # (B, H, T, d_head)

        Returns:
            output: torch.Tensor, shape (B, H, T, d_head)
        """
```

### ⚠️ Causal Mask 方向约定

官方 handout 明确说：用 `torch.triu` 做 upper-triangular mask，在 softmax 之前应用。

```python
# 上三角（不含对角线）= 未来位置 → 填 -inf
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(causal_mask, float('-inf'))

# 等价写法（用下三角）：
# keep_mask = torch.tril(torch.ones(T, T)).bool()   # True = 保留
# scores = scores.masked_fill(~keep_mask, float('-inf'))
```

### ⚠️ Padding Mask

```python
pad_mask = attention_mask[:, None, None, :].bool()  # (B,1,1,T)
scores = scores.masked_fill(~pad_mask, float('-inf'))
```

验证：`python3 sanity_check.py`

---

## 3. `modules/gpt2_layer.py` — `GPT2Layer.add()` + `GPT2Layer.forward()`

**负责人**：Ziyuan  
**来源**：Section 5.2（函数名：`modules.gpt2_layer.add` 和 `modules.gpt2_layer.forward`）

```python
class GPT2Layer(nn.Module):
    def add(
        self,
        x: torch.Tensor,                    # (B, T, d)
        residual: torch.Tensor,             # (B, T, d)
    ) -> torch.Tensor:
        """
        残差连接 + Dropout。
        GPT-2 在残差连接之前对子层输出应用 dropout（p=0.1）。

        Returns:
            output: torch.Tensor, shape (B, T, d)
        """

    def forward(
        self,
        hidden_states: torch.Tensor,        # (B, T, d)
        attention_mask: torch.Tensor,       # (B, T)
    ) -> torch.Tensor:
        """
        单个 GPT-2 Transformer Block（Pre-LN，见 handout Figure 2）：

          1. residual = hidden_states
          2. hidden_states = LayerNorm(hidden_states)           # ln_1（Pre-LN）
          3. hidden_states = CausalSelfAttention(hidden_states, attention_mask)
          4. hidden_states = self.add(hidden_states, residual)  # 残差 1 + dropout
          5. residual = hidden_states
          6. hidden_states = LayerNorm(hidden_states)           # ln_2（Pre-LN）
          7. hidden_states = FFN(hidden_states)
          8. hidden_states = self.add(hidden_states, residual)  # 残差 2 + dropout

        FFN（handout Section 5.1）：
          Linear(d → 4d) → GELU → Linear(4d → d)
          ⚠️ handout 公式写的是 ReLU，但 GPT-2 实际权重用 GELU；
             加载预训练权重时以 GELU 为准。

        Returns:
            output: torch.Tensor, shape (B, T, d)
        """
```

---

## 4. `models/gpt2.py` — `GPT2Model.embed()` + `GPT2Model.hidden_state_to_token()`

**负责人**：Yuheng  
**来源**：Section 5.2（函数名：`models.gpt2.embed`）+ Section 7.3.1

```python
class GPT2Model(nn.Module):
    @classmethod
    def from_pretrained(
        cls,
        model: str = 'gpt2',
        d: int = 768,
        l: int = 12,
        num_heads: int = 12,
    ) -> 'GPT2Model':
        """加载 HuggingFace 预训练权重。"""

    def embed(
        self,
        input_ids: torch.Tensor,            # (B, T)
    ) -> torch.Tensor:
        """
        官方要求实现的函数（models.gpt2.embed）。

        token embedding + position embedding：
          1. token_emb = token_embedding(input_ids)     # (B, T, d)
          2. pos_ids = arange(T).expand(B, T)
          3. pos_emb = pos_embedding(pos_ids)           # (B, T, d)
          4. return Dropout(token_emb + pos_emb)        # (B, T, d)，p=0.1

        Returns:
            embeddings: torch.Tensor, shape (B, T, d)
        """

    def forward(
        self,
        input_ids: torch.Tensor,            # (B, T)
        attention_mask: torch.Tensor,       # (B, T)
    ) -> dict:
        """
        完整前向传播：embed → L 个 GPT2Layer → ln_f

        Returns dict 包含两个字段（均来自最后一个 GPT2Layer 的输出）：
            'last_hidden_state': torch.Tensor, shape (B, T, d)
                ← 所有位置的隐状态
            'last_token':        torch.Tensor, shape (B, d)
                ← 最后一个 token 的隐状态，= last_hidden_state[:, -1, :]
                ← classifier.py 用这个做情感分类
        """

    def hidden_state_to_token(
        self,
        hidden_state: torch.Tensor,         # (B, d) 或 (B, T, d)
    ) -> torch.Tensor:
        """
        将隐状态映射到词表上的 logits。
        与 token_embedding 权重共享（weight tying）。

        output = hidden_state @ token_embedding.weight.T

        Returns:
            logits: torch.Tensor, shape (B, V) 或 (B, T, V)
            ← paraphrase_detection.py 用这个取 "yes"(8505)/"no"(3919) 的概率
            ← sonnet_generation.py 用这个生成下一个 token
        """
```

**⚠️ `forward()` 返回 dict，不是单个 tensor。下游代码用 `output['last_token']` 取最后 token。**

---

## 5. `classifier.py` — `GPT2SentimentClassifier`

**负责人**：Yuheng  
**来源**：Section 6.2

```python
class GPT2SentimentClassifier(nn.Module):
    def __init__(self, config):
        """
        - self.gpt = GPT2Model.from_pretrained()
        - self.classifier = nn.Linear(768, config.num_labels)
        - self.dropout = nn.Dropout(config.hidden_dropout_prob)
        - fine_tune_mode == 'last-linear-layer'：冻结 GPT 参数，只训练 classifier
        - fine_tune_mode == 'full-model'：全部参数参与训练
        """

    def forward(
        self,
        input_ids: torch.Tensor,            # (B, T)
        attention_mask: torch.Tensor,       # (B, T)
    ) -> torch.Tensor:
        """
        用 GPT-2 last token embedding 做情感分类（Section 6.2）：
          1. output = self.gpt(input_ids, attention_mask)
          2. last = output['last_token']                  # (B, d)
          3. logits = self.classifier(self.dropout(last)) # (B, num_labels)

        Returns:
            logits: torch.Tensor, shape (B, num_labels)
            ← raw logits，训练用 F.cross_entropy，不加 softmax
        """
```

### Baseline 精度要求（Section 6.3）

| 模式 | 数据集 | Dev Accuracy |
|------|--------|-------------|
| last-linear-layer | SST | **0.462** |
| full-model | SST | **0.513** |
| last-linear-layer | CFIMDB | **0.861** |
| full-model | CFIMDB | **0.976** |

---

# Part 2：Section 7（Extensions）

验收：`paraphrase_detection.py` → `sonnet_generation.py`

---

## 6. `paraphrase_detection.py` — `ParaphraseGPT.forward()`

**负责人**：Ziyuan  
**来源**：Section 7.3.1

```python
class ParaphraseGPT(nn.Module):
    """
    Cloze-style 释义检测（Section 7.1）。
    输入格式：
        'Is "{s2}" a paraphrase of "{s1}"? Answer "yes" or "no": '
    预测最后一个 token 位置的 next token：
        "yes"（BPE id = 8505）→ label 1（是释义）
        "no" （BPE id = 3919）→ label 0（不是释义）
    """

    def forward(
        self,
        input_ids: torch.Tensor,            # (B, T)
        attention_mask: torch.Tensor,       # (B, T)
    ) -> torch.Tensor:
        """
          1. output = self.gpt(input_ids, attention_mask)
          2. last = output['last_token']                          # (B, d)
          3. logits = self.paraphrase_detection_head(last)        # (B, 2)
             paraphrase_detection_head = nn.Linear(d, 2)

          ⚠️ 也可以用 hidden_state_to_token 取 yes/no 两个 token 的 logit：
             token_logits = self.gpt.hidden_state_to_token(last)  # (B, V)
             logits = token_logits[:, [3919, 8505]]               # (B, 2)，[no, yes]

        Returns:
            logits: torch.Tensor, shape (B, 2)
            ← 训练用 F.cross_entropy(logits, labels)
        """
```

验证：`python3 paraphrase_detection.py --use_gpu`

---

## 7. `sonnet_generation.py` — `SonnetGPT.forward()` + `SonnetGPT.generate()`

**负责人**：Ziran  
**来源**：Section 7.3.2

```python
class SonnetGPT(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,            # (B, T)
        attention_mask: torch.Tensor,       # (B, T)
    ) -> torch.Tensor:
        """
          1. output = self.gpt(input_ids, attention_mask)
          2. hidden = output['last_hidden_state']                # (B, T, d)
          3. logits = self.gpt.hidden_state_to_token(hidden)     # (B, T, V)

        Returns:
            logits: torch.Tensor, shape (B, T, V)
            ← 训练：loss = F.cross_entropy(logits[:, :-1], input_ids[:, 1:])
            ← 评测：chrF score（字符级 n-gram，类似 BLEU）
        """

    def generate(
        self,
        encoding: torch.Tensor,            # (1, T) 前 3 行的 token ids
        temperature: float = 1.2,
        top_p: float = 0.9,
        max_length: int = 128,
    ) -> tuple[torch.Tensor, str]:
        """
        官方提供了 top-p sampling 的基础实现（TODO 注释建议改进）。
        Extension 目标：在此基础上改进生成质量，可选方向：
          - beam search：维护 k 个候选序列，选概率最高的
          - top-k sampling：限制每步只从概率最高的 k 个 token 中采样
          - repetition penalty：对已出现 token 降权，减少重复
          - 结合韵律约束：鼓励押韵的 token

        Returns:
            (token_ids, generated_text)
        """
```

验证：`python3 sonnet_generation.py --use_gpu`

---

## 8. Extension — LoRA 文件（Section 7.4 PEFT）

三个文件完全独立，零 merge 冲突。`lora_pissa.py` 和 `lora_loftq.py` 都 import `LoRALinear` 自 `lora_linear.py`，所以 Ziran 需要先把基类写完再通知另外两人。

### `lora_linear.py` — Ziran

```python
class LoRALinear(nn.Module):
    """
    基类，其他两个文件都依赖它。Ziran 在 Phase 3 第一天优先完成。
    forward(x) = original(x) + (alpha / rank) * x @ A @ B
    - A: (in_features, rank)
    - B: (rank, out_features)
    原始权重冻结，只训练 A 和 B。
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        init_method: str = 'lora',          # 调用对应 init_ 函数
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

def init_lora(A: torch.Tensor, B: torch.Tensor, **kwargs) -> None:
    """A ~ N(0,1)，B = 0。训练开始时低秩分支输出为 0，不破坏预训练权重。"""

def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    init_method: str = 'lora',
    target_modules: list = ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
) -> nn.Module:
    """遍历模型，将 target_modules 中的 nn.Linear 替换为 LoRALinear。"""
```

### `lora_pissa.py` — Ziyuan

```python
from lora_linear import LoRALinear

def init_pissa(A: torch.Tensor, B: torch.Tensor,
               weight: torch.Tensor, rank: int, **kwargs) -> None:
    """
    SVD 初始化：
      U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
      A = U[:, :rank] * sqrt(S[:rank])
      B = sqrt(S[:rank]).unsqueeze(1) * Vt[:rank, :]
      weight.data -= A @ B   ← 原始权重替换为残差
    """
```

### `lora_loftq.py` — Yuheng

```python
from lora_linear import LoRALinear

def init_loftq(A: torch.Tensor, B: torch.Tensor,
               weight: torch.Tensor, rank: int,
               num_bits: int = 4, **kwargs) -> None:
    """
    量化 + SVD 补偿：
      quantized = quantize(weight, num_bits)
      residual = weight - quantized
      A, B 通过 SVD(residual) 初始化（同 PiSSA 的 SVD 步骤）
    """
```

**import 方式（`run_experiment.py` 里）：**

```python
from lora_linear import apply_lora, init_lora
from lora_pissa import init_pissa
from lora_loftq import init_loftq
```

---

## 9. 实验脚本 — `run_experiment.py`

```python
# 开跑前全员锁定，之后不改
FIXED_CONFIG = {'seed': 42, 'epochs': 10, 'lr': 1e-5, 'batch_size': 8, 'model_size': 'gpt2'}
LORA_GRID    = {'init_method': ['lora', 'pissa', 'loftq'], 'rank': [4, 8, 16], 'alpha': 16.0}

def evaluate(model, dataloader, device, task) -> dict:
    """Returns: {'accuracy': float, 'perplexity': float, 'loss': float}"""

# 结果存 predictions/results.json
# {'task', 'init_method', 'rank', 'metric', 'loss'}
```

---

# 依赖链与验证顺序

```
【Part 1 — Section 5+6】
optimizer.py                  →  python3 optimizer_test.py
attention.py（.attention）    ┐
gpt2_layer.py（.add/.forward）├→  python3 sanity_check.py
gpt2.py（.embed）             ┘
classifier.py（两个模式）     →  python3 classifier.py --fine-tune-mode last-linear-layer
                                 python3 classifier.py --fine-tune-mode full-model

【Part 2 — Section 7】
paraphrase_detection.py       →  python3 paraphrase_detection.py --use_gpu
sonnet_generation.py          →  python3 sonnet_generation.py --use_gpu
lora_linear.py（先完成）      ┐
lora_pissa.py                 ├→  python3 run_experiment.py
lora_loftq.py                 ┘
```

---

# 开工前 Checklist

- [ ] 统一 Python 环境：`source setup.sh`（不要改 `setup.sh` 里的库）
- [ ] 确认函数名与 handout 一致（`attention`，`add`，`embed`，不是 `forward`）
- [ ] 锁定实验配置（seed / epochs / lr / batch_size）后写入 `run_experiment.py`
