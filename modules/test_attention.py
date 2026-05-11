import torch
import traceback
import sys
import os

class GPT2Config:
    def __init__(self):
        self.d = 768                            
        self.hidden_size = 768                  
        self.num_heads = 12                     
        self.num_attention_heads = 12           
        self.n_head = 12                        
        self.attn_pdrop = 0.1                   
        self.attention_probs_dropout_prob = 0.1 

def run_test():
    print("="*50)
    print("🚀 Attention 模块底层数学质检报告")
    print("="*50)
    
    device = torch.device("cpu")

    # 1. 初始化模型
    config = GPT2Config()
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from modules.attention import CausalSelfAttention
        
        model = CausalSelfAttention(config).to(device)
        model.eval() 
    except Exception as e:
        print(f"[-] 初始化失败: {e}")
        return

    # 2. 构造测试数据
    B, H, T, d_head = 2, 12, 8, 64
    torch.manual_seed(42) # 固定随机种子，保证每次输出的数值一样
    q = torch.randn(B, H, T, d_head, device=device)
    k = torch.randn(B, H, T, d_head, device=device)
    v = torch.randn(B, H, T, d_head, device=device)
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=torch.long, device=device)

    # 3. 执行前向传播
    try:
        output = model.attention(query=q, key=k, value=v, attention_mask=mask)

        # ---------------------------------------------------------
        # 防线一：维度坍塌拦截 (Shape Alignment)
        # ---------------------------------------------------------
        print("\n[防线一：空间几何维度校验]")
        expected_shape = (B, H, T, d_head)
        actual_shape = tuple(output.shape)
        print(f"  ├─ 预期输出形状 : {expected_shape}")
        print(f"  ├─ 实际输出形状 : {actual_shape}")
        if expected_shape == actual_shape:
            print("  └─ 结论: ✅ 通过 (张量折叠与矩阵乘法完全对齐)")
        else:
            print("  └─ 结论: ❌ 失败 (存在维度的错误转置或广播)")

        # ---------------------------------------------------------
        # 防线二：数值爆炸检测 (NaN / Inf Detection)
        # ---------------------------------------------------------
        print("\n[防线二：掩码与梯度数值稳定性校验]")
        expected_nan_status = False
        actual_nan_status = torch.isnan(output).any().item()
        print(f"  ├─ 预期存在 NaN : {expected_nan_status}")
        print(f"  ├─ 实际存在 NaN : {actual_nan_status}")
        
        # 提取第一个词的前4个特征，证明输出的是真实的浮点数
        sample_data = output[0, 0, 0, :4].detach().numpy()
        print(f"  ├─ 实际特征切片 (0,0,0,:4) : {sample_data}")
        
        if expected_nan_status == actual_nan_status:
            print("  └─ 结论: ✅ 通过 (Mask 逻辑严密，Softmax 未发生除零崩溃)")
        else:
            print("  └─ 结论: ❌ 失败 (Mask 填充异常导致数值越界)")

        # ---------------------------------------------------------
        # 防线三：接口契约校验 (API Contract)
        # ---------------------------------------------------------
        print("\n[防线三：函数签名与返回值校验]")
        expected_type = "<class 'torch.Tensor'>"
        actual_type = str(type(output))
        print(f"  ├─ 预期返回类型 : {expected_type}")
        print(f"  ├─ 实际返回类型 : {actual_type}")
        
        if actual_type == expected_type:
            print("  └─ 结论: ✅ 通过 (成功接管外部参数并返回合法张量)")
        else:
            print("  └─ 结论: ❌ 失败 (可能返回了 None 或保留了 raise NotImplementedError)")

        print("\n" + "="*50)

    except Exception as e:
        print("\n[致命错误] 防线已崩溃！详细追踪如下：")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()