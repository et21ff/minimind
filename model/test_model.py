import torch
import torch.nn as nn
from model import RMSNorm
from model import precompute_freqs_cis


def test_rmsnorm_consistency():
    dim = 512
    eps = 1e-5
    batch_size, seq_len = 2, 16

    # 1. 初始化你的模块
    my_rms = RMSNorm(dim, eps)

    # 2. 初始化 PyTorch 官方模块 (要求 torch >= 2.4)
    # 如果版本较低，可以手动写一个最基础的公式版作为 Benchmark
    official_rms = nn.RMSNorm(dim, eps=eps)

    # 3. 统一权重（保证对比前提一致）
    with torch.no_grad():
        official_rms.weight.copy_(my_rms.weight)

    # 4. 构造随机输入
    x = torch.randn(batch_size, seq_len, dim)

    # 5. 计算结果
    out_custom = my_rms(x)
    out_official = official_rms(x)

    # 6. 比较差异
    # 使用 allclose 检查浮点数误差，atol 是绝对误差容忍度
    diff = torch.abs(out_custom - out_official).max()
    is_correct = torch.allclose(out_custom, out_official, atol=1e-6)

    print(f"最大数值差异: {diff.item():.2e}")
    print(f"测试通过: {is_correct}")


def test_precompute():
    # 参数设置
    DIM = 128
    ORIG_MAX = 2048
    END = 4096  # 触发 YaRN 缩放
    FACTOR = 16.0

    scaling = {
        "original_max_position_embeddings": ORIG_MAX,
        "factor": FACTOR,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attention_factor": 1.0,
    }

    # 执行函数
    cos, sin = precompute_freqs_cis(DIM, END, rope_scaling=scaling)

    # --- 检查点 1: 形状 ---
    print(f"Check 1 - Shape: cos {cos.shape}, sin {sin.shape}")
    assert cos.shape == (END, DIM), "形状不匹配！"

    # --- 检查点 2: 位置 0 的值 ---
    # cos(0)=1, sin(0)=0
    print(f"Check 2 - Pos 0: cos[0,0]={cos[0, 0]:.4f}, sin[0,0]={sin[0, 0]:.4f}")
    assert torch.allclose(cos[0], torch.ones(DIM)), "位置 0 的 cos 必须为 1"
    assert torch.allclose(sin[0], torch.zeros(DIM)), "位置 0 的 sin 必须为 0"

    # --- 检查点 3: YaRN 缩放验证 ---
    # 比较没有 scaling 和有 scaling 的情况
    cos_no_scale, _ = precompute_freqs_cis(DIM, END, rope_scaling=None)

    # 在低频维度（索引较大处），YaRN 应该让频率变慢，导致 cos 变化更小
    # 也就是 cos_yarn 的相位更接近 1 (角度更小)
    diff = (cos[1, -1] - cos_no_scale[1, -1]).abs()
    print(f"Check 3 - Scaling Diff at last dim: {diff:.6f}")
    if diff > 0:
        print("✅ YaRN 缩放已生效：频率已根据因子调整。")
    else:
        print("❌ 警告：未检测到缩放差异。")


test_precompute()

# test_rmsnorm_consistency()
