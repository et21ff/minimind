import math

import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union


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
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(
        unsqueeze_dim
    )
    k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(
        unsqueeze_dim
    )

    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        assert args.hidden_size % args.num_attention_heads == 0, (
            "hidden_size must be divisible by num_attention_heads"
        )

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        assert self.head_dim % 2 == 0, "RoPE requires even head_dim"

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, args: MokioMindConfig):
        # silu函数
        # up_proj,down_proj.gate_porj
        # 计算扩增的维度数
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.up_proj = nn.Linear(
            in_features=args.hidden_size,
            out_features=args.intermediate_size,
            bias=False,
        )
        self.gate_proj = nn.Linear(
            in_features=args.hidden_size,
            out_features=args.intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            in_features=args.intermediate_size,
            out_features=args.hidden_size,
            bias=False,
        )

        self.dropout = nn.Dropout(args.dropout)
        self.actfn = ACT2FN[args.hidden_act]

    def forward(self, x: torch.Tensor):
        gated = self.actfn(self.gate_proj(x))
        return self.dropout(self.down_proj(gated * self.up_proj(x)))


class MokioMindBlock(nn.Module):
    # mlp层（FFN）
    # attention层
    # RMS_NORM
    def __init__(self, layer_id, args: MokioMindConfig):
        super().__init__()
        self.layer_id = layer_id
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.input_layer_norm = RMSNorm(self.hidden_size, args.rms_norm_eps)
        self.post_layer_norm = RMSNorm(self.hidden_size, args.rms_norm_eps)
        self.self_attention = Attention(args)
        self.mlp = FeedForward(args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attention(
            self.input_layer_norm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.post_layer_norm(hidden_states))
        return hidden_states, present_key_value


class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        # layer数量
        # 词表 embedding维度
        # embeding
        # 初始化freq_cos，freq_sin
        # dropout
        # layernorm
        super().__init__()
        self.vocab_size, self.hidden_size = (config.vocab_size, config.hidden_size)
        self.config = config
        self.embeded_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = RMSNorm(dim=self.hidden_size, eps=config.rms_norm_eps)
        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList(
            MokioMindBlock(i, config) for i in range(config.num_hidden_layers)
        )
        freq_cos, freq_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        self.register_buffer("freq_cos", freq_cos, persistent=False)
        self.register_buffer("freq_sin", freq_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache=False,
        attention_mask: Optional[torch.tensor] = None,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )
        position_embeddings = (
            self.freq_cos[start_pos : start_pos + seq_len],
            self.freq_sin[start_pos : start_pos + seq_len],
        )
        hidden_states = self.dropout(self.embeded_tokens(input_ids))
        presents = []

        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)

        return hidden_states, presents


class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MokioMindConfig

    def __init__(self, config: MokioMindConfig):
        super().__init__(config)
        self.model = MokioMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.model.embeded_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.tensor] = None,
        attention_mask: Optional[torch.tensor] = None,
        past_key_values: Optional[Tuple[torch.tensor, torch.tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache=False,
        logits_to_keep: Union[int, torch.tensor] = 0,
        **args,
    ):
        h, past_kvs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        logits = self.lm_head(h[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_label = labels[
                ...,
                1:,
            ].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_label.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=past_kvs, hidden_states=h
        )
