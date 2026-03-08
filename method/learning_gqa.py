import torch
import torch.nn as nn
import math

# ========== 配置 ==========
bsz, seq_len, hidden = 2, 5, 512
n_heads, n_kv_heads = 8, 2
head_dim = hidden // n_heads  # 64
n_rep = n_heads // n_kv_heads  # 4

# ========== 构建 ==========
q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
k_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
v_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)

x = torch.randn(bsz, seq_len, hidden)
print(f"输入 x:                {x.shape}")

# ========== Step 1: 线性投影 ==========
xq = q_proj(x)
xk = k_proj(x)
xv = v_proj(x)
print(f"\n--- Step 1: 线性投影 ---")
print(f"xq = q_proj(x):       {xq.shape}")
print(f"xk = k_proj(x):       {xk.shape}")
print(f"xv = v_proj(x):       {xv.shape}")

# ========== Step 2: view 拆头 ==========
xq = xq.view(bsz, seq_len, n_heads, head_dim)
xk = xk.view(bsz, seq_len, n_kv_heads, head_dim)
xv = xv.view(bsz, seq_len, n_kv_heads, head_dim)
print(f"\n--- Step 2: view 拆分多头 ---")
print(f"xq.view:              {xq.shape}")
print(f"xk.view:              {xk.shape}")
print(f"xv.view:              {xv.shape}")

# ========== Step 3: RoPE (shape 不变) ==========
# xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
print(f"\n--- Step 3: RoPE (shape不变) ---")
print(f"xq after RoPE:        {xq.shape}")
print(f"xk after RoPE:        {xk.shape}")

# ========== Step 4: KV Cache 拼接 (假设有缓存) ==========
past_k = torch.randn(bsz, 10, n_kv_heads, head_dim)  # 历史10个token
past_v = torch.randn(bsz, 10, n_kv_heads, head_dim)
print(f"\n--- Step 4: KV Cache ---")
print(f"past_k:               {past_k.shape}")
print(f"past_v:               {past_v.shape}")
xk = torch.cat([past_k, xk], dim=1)
xv = torch.cat([past_v, xv], dim=1)
print(f"xk after cat:         {xk.shape}")
print(f"xv after cat:         {xv.shape}")
total_len = xk.shape[1]  # 15

# ========== Step 5: transpose → [bsz, heads, seq, dim] ==========
xq = xq.transpose(1, 2)
print(f"\n--- Step 5: transpose (交换 seq 和 heads) ---")
print(f"xq.transpose(1,2):    {xq.shape}")

# ========== Step 6: repeat_kv ==========
# 每个 kv head 复制 n_rep 次
print(f"\n--- Step 6: repeat_kv (复制KV头) ---")
print(f"xk before repeat:     {xk.shape}")
# repeat_kv 内部操作:
xk_expanded = xk[:, :, :, None, :].expand(bsz, total_len, n_kv_heads, n_rep, head_dim)
print(f"  expand:              {xk_expanded.shape}")
xk = xk_expanded.reshape(bsz, total_len, n_heads, head_dim)
print(f"  reshape:             {xk.shape}")
xk = xk.transpose(1, 2)
print(f"  transpose:           {xk.shape}")

# 对 xv 做同样操作
xv = xv[:, :, :, None, :].expand(bsz, total_len, n_kv_heads, n_rep, head_dim)
xv = xv.reshape(bsz, total_len, n_heads, head_dim).transpose(1, 2)
print(f"xv after repeat+trans: {xv.shape}")

# ========== Step 7: Q @ K^T ==========
scores = xq @ xk.transpose(-2, -1)
print(f"\n--- Step 7: Q @ K^T ---")
print(f"xq:                   {xq.shape}")
print(f"xk^T:                 {xk.transpose(-2, -1).shape}")
print(f"scores:               {scores.shape}")

# ========== Step 8: scale ==========
scores = scores / math.sqrt(head_dim)
print(f"\n--- Step 8: scale (shape不变) ---")
print(f"scores:               {scores.shape}")

# ========== Step 9: causal mask ==========
causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
print(f"\n--- Step 9: causal mask ---")
print(f"causal mask:           {causal.shape}")
print(f"scores 的 key 维度切片: scores[:,:,:,-{seq_len}:]")
print(f"  即 scores[:,:,:,{total_len - seq_len}:{total_len}]")
scores[:, :, :, -seq_len:] += causal

# ========== Step 10: attention_mask (padding mask) ==========
attention_mask = torch.tensor(
    [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
)  # 当前输入的 mask [2, 5]
print(f"原始 attention_mask:              {attention_mask.shape}")
# 有 KV Cache 时，past tokens 全部有效，前面补 1
if past_k is not None:
    past_len = past_k.shape[1]  # 10
    past_mask = torch.ones(bsz, past_len, dtype=attention_mask.dtype)  # [2, 10] 全1
    attention_mask = torch.cat([past_mask, attention_mask], dim=-1)  # [2, 15]
    print(f"拼接后 attention_mask:            {attention_mask.shape}")
    # 样本1: [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1]  全有效
    # 样本2: [1,1,1,1,1,1,1,1,1,1, 1,1,1,0,0]  最后2个padding
ext = attention_mask.unsqueeze(1).unsqueeze(2)  # [2, 1, 1, 15]
ext = (1.0 - ext.float()) * -1e9
print(f"extended_mask:                    {ext.shape}")
print(f"scores:                           {scores.shape}")
print(f"广播: {ext.shape} → {scores.shape}")  # [2,1,1,15] → [2,8,5,15] ✓
scores = scores + ext

# ========== Step 11: softmax ==========
scores = torch.softmax(scores.float(), dim=-1)
print(f"\n--- Step 11: softmax (shape不变, 沿 dim=-1) ---")
print(f"scores:               {scores.shape}")

# ========== Step 12: scores @ V ==========
output = scores @ xv
print(f"\n--- Step 12: scores @ V ---")
print(f"scores:               {scores.shape}")
print(f"xv:                   {xv.shape}")
print(f"output:               {output.shape}")

# ========== Step 13: transpose + reshape ==========
output = output.transpose(1, 2)
print(f"\n--- Step 13: 合并多头 ---")
print(f"  transpose(1,2):     {output.shape}")
output = output.reshape(bsz, seq_len, -1)
print(f"  reshape:            {output.shape}")

# ========== Step 14: o_proj ==========
output = o_proj(output)
print(f"\n--- Step 14: o_proj ---")
print(f"output:               {output.shape}")
