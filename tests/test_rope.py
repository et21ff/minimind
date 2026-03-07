import torch
import sys
import math

sys.path.append("/root/minimind")
from model.model import precompute_freqs_cis


# --- 测试1: 无 scaling，检查输出形状 ---
def test_shape():
    dim, end = 128, 4096
    cos, sin = precompute_freqs_cis(dim, end)
    assert cos.shape == (end, dim), f"Expected ({end},{dim}), got {cos.shape}"
    assert sin.shape == (end, dim), f"Expected ({end},{dim}), got {sin.shape}"
    print("✓ Shape test passed")


# --- 测试2: 无 scaling，cos/sin 值域在 [-1, 1] ---
def test_value_range():
    cos, sin = precompute_freqs_cis(128, 4096)
    # attnfactor=1.0 时，值域应在 [-1, 1]
    assert cos.abs().max() <= 1.0 + 1e-6
    assert sin.abs().max() <= 1.0 + 1e-6
    print("✓ Value range test passed")


# --- 测试3: position=0 时，sin 应全为 0，cos 应全为 1 ---
def test_position_zero():
    cos, sin = precompute_freqs_cis(128, 4096)
    assert torch.allclose(sin[0], torch.zeros_like(sin[0]), atol=1e-6)
    assert torch.allclose(cos[0], torch.ones_like(cos[0]), atol=1e-6)
    print("✓ Position zero test passed")


# --- 测试4: 带 scaling 的情况 ---
def test_with_scaling():
    rope_scaling = {
        "original_max_position_embeddings": 2048,
        "factor": 16,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attention_factor": 1.0,
    }
    cos, sin = precompute_freqs_cis(128, 32768, rope_scaling=rope_scaling)
    assert cos.shape == (32768, 128)
    # 确保不含 NaN / Inf
    assert not torch.isnan(cos).any(), "Contains NaN"
    assert not torch.isinf(cos).any(), "Contains Inf"
    print("✓ Scaling test passed")


# --- 测试5: scaling 不应改变短序列的频率 ---
def test_scaling_preserves_short():
    dim, short_end = 128, 2048
    cos_base, sin_base = precompute_freqs_cis(dim, short_end)
    # end <= orig_max 时不触发 scaling
    rope_scaling = {
        "original_max_position_embeddings": 2048,
        "factor": 16,
    }
    cos_scaled, sin_scaled = precompute_freqs_cis(
        dim, short_end, rope_scaling=rope_scaling
    )
    assert torch.allclose(cos_base, cos_scaled, atol=1e-5)
    print("✓ Short sequence preservation test passed")


def test_scaling_frequency_modification():
    """验证 scaling 确实修改了频率"""
    dim, rope_base = 128, 1e6
    rope_scaling = {
        "original_max_position_embeddings": 2048,
        "factor": 16,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attention_factor": 1.0,
    }
    cos_base, _ = precompute_freqs_cis(dim, 32768)
    cos_scaled, _ = precompute_freqs_cis(dim, 32768, rope_scaling=rope_scaling)
    # scaling 后结果应该不同
    assert not torch.allclose(cos_base, cos_scaled, atol=1e-5), (
        "Scaling should change the output"
    )
    print("✓ Scaling actually modifies frequencies")


def test_scaling_ramp_behavior():
    """调用原函数，验证 ramp 三段行为"""
    dim, rope_base = 128, 1e6
    end = 32768
    rope_scaling = {
        "original_max_position_embeddings": 2048,
        "factor": 16,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attention_factor": 1.0,
    }
    factor = 16
    orig_max = 2048

    # 调用原函数：无缩放 vs 有缩放
    cos_base, sin_base = precompute_freqs_cis(dim, end, rope_base)
    cos_scaled, sin_scaled = precompute_freqs_cis(dim, end, rope_base, rope_scaling)

    # 计算 ramp 边界
    def inv_dim(b):
        return dim * math.log(orig_max / (b * 2 * math.pi)) / (2 * math.log(rope_base))

    low = max(0, inv_dim(32.0))
    high = min(inv_dim(1.0), dim // 2 - 1)
    low_idx = int(math.floor(low))
    high_idx = int(math.ceil(high)) + 1
    half = dim // 2

    # ---- 区间 [0, low_idx): ramp=0, 频率不变 ----
    # 直接跟无缩放版本比，应该完全一致
    if low_idx > 0:
        assert torch.allclose(
            cos_scaled[:, :low_idx], cos_base[:, :low_idx], atol=1e-6
        ), "Unchanged zone: scaled should equal base"
        assert torch.allclose(
            sin_scaled[:, :low_idx], sin_base[:, :low_idx], atol=1e-6
        ), "Unchanged zone (sin): scaled should equal base"
        print(f"  dims [0, {low_idx}): unchanged ✓")

    # ---- 区间 [high_idx, half): ramp=1, freq = base_freq / factor ----
    if high_idx < half:
        base_freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:half].float() / dim))
        t = torch.arange(end).float()
        expected_cos = torch.cos(torch.outer(t, base_freqs[high_idx:] / factor))
        expected_sin = torch.sin(torch.outer(t, base_freqs[high_idx:] / factor))

        assert torch.allclose(cos_scaled[:, high_idx:half], expected_cos, atol=1e-5), (
            "Fully scaled zone: freq should be base/factor"
        )
        assert torch.allclose(sin_scaled[:, high_idx:half], expected_sin, atol=1e-5), (
            "Fully scaled zone (sin): freq should be base/factor"
        )
        print(f"  dims [{high_idx}, {half}): fully scaled by 1/{factor} ✓")

    # ---- 过渡区 [ceil(low), floor(high)+1): 既不等于 base 也不等于 base/factor ----
    mid_start = int(math.ceil(low))
    mid_end = int(math.floor(high)) + 1
    if mid_start < mid_end:
        actual = cos_scaled[:, mid_start:mid_end]

        # 不等于无缩放
        assert not torch.allclose(actual, cos_base[:, mid_start:mid_end], atol=1e-5), (
            "Transition should differ from unscaled"
        )

        # 不等于完全缩放
        base_freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:half].float() / dim))
        t = torch.arange(end).float()
        full_scale = torch.cos(torch.outer(t, base_freqs[mid_start:mid_end] / factor))
        assert not torch.allclose(actual, full_scale, atol=1e-5), (
            "Transition should differ from fully scaled"
        )

        print(f"  dims [{mid_start}, {mid_end}): transition (neither extreme) ✓")

    print("✓ Ramp three-segment behavior verified")


def test_attn_factor():
    """验证 attention_factor 正确缩放输出"""
    dim = 128
    scaling_1x = {"attention_factor": 1.0}
    scaling_2x = {"attention_factor": 2.0}
    cos_1x, sin_1x = precompute_freqs_cis(dim, 4096, rope_scaling=scaling_1x)
    cos_2x, sin_2x = precompute_freqs_cis(dim, 4096, rope_scaling=scaling_2x)
    # end/orig_max = 4096/2048 = 2 > 1, scaling 生效
    assert torch.allclose(cos_2x, cos_1x * 2.0, atol=1e-5), (
        "attention_factor=2 should double the output"
    )
    print("✓ attention_factor scaling correct")


test_shape()
test_value_range()
test_position_zero()
test_with_scaling()
test_scaling_preserves_short()
test_scaling_frequency_modification()
test_scaling_ramp_behavior()
test_attn_factor()
