import torch
import torch.nn as nn
from model import RMSNorm
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

test_rmsnorm_consistency()