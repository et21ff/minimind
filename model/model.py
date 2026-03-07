import math

from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        # 需要将weigh设置为可训练的参数

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 注意需要keepdim

    def forward(self, x):
        return self.weight * self._norm(x).type_as(x)
        # 注意需要保持返回的数据类型和传入的一致


def precompute_freqs_cis(
    dim: int, end: int = 32768, rope_base=1e6, rope_scaling: Optional[dict] = None
):
    freqs, attnfactor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)),
        1.0,
    )
    print(freqs.shape)
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attnfactor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )
        if end > orig_max:

            def inv_dim(b: float):
                return (
                    dim
                    * math.log(orig_max / (2 * math.pi * b))
                    / (2 * math.log(rope_base))
                )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max((high - low), 0.001),
                0,
                1,
            )

            freqs = (1 - ramp + ramp / factor) * freqs

    t = torch.arange(end, device=freqs.device).float()
    freqs = torch.outer(t, freqs)
    freq_cos = torch.cat((torch.cos(freqs), torch.cos(freqs)), dim=-1) * attnfactor
    freq_sin = torch.cat((torch.sin(freqs), torch.sin(freqs)), dim=-1) * attnfactor

    return freq_cos, freq_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            -x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :], dim=-1
        )

    q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(
        unsqueeze_dim
    )
    k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(
        unsqueeze_dim
    )

    return q_embed, k_embed
