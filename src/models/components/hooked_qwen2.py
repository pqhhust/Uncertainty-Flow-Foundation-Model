"""Hooked Qwen2 model with TransformerLens HookPoints.

Re-implements Qwen2 model components with HookPoint instances at key locations
for extracting intermediate representations. Follows the pattern from the
hallucination project's LLaVA hooked model (text_llava.py).

Hook points:
  - Per attention: hook_q, hook_k, hook_v (after Q/K/V projections)
  - Per decoder layer: hook_resid_pre, hook_attn_out, hook_resid_mid,
                       hook_mlp_out, hook_resid_post
  - Model level: collects x_t (hidden states after each layer) and
                 means (post-attention residuals) for distillation
"""

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from transformer_lens.hook_points import HookPoint, HookedRootModule


# =============================================================================
# Qwen2 sub-components (RMSNorm, RotaryEmbedding, MLP) - standard implementation
# =============================================================================


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# =============================================================================
# Eager attention forward (fallback)
# =============================================================================

def eager_attention_forward(
    module,
    query_states,
    key_states,
    value_states,
    attention_mask,
    dropout=0.0,
    scaling=None,
    **kwargs,
):
    """Standard scaled dot-product attention."""
    if scaling is None:
        scaling = module.head_dim ** -0.5

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# =============================================================================
# Hooked Qwen2 Attention with HookPoints on Q, K, V
# =============================================================================


class HookedQwen2Attention(nn.Module):
    """Qwen2 attention with HookPoint on Q, K, V projections."""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=True,  # Qwen2 uses bias in attention
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

        # TransformerLens HookPoints
        self.hook_q = HookPoint()
        self.hook_q.layer_idx = layer_idx
        self.hook_k = HookPoint()
        self.hook_k.layer_idx = layer_idx
        self.hook_v = HookPoint()
        self.hook_v.layer_idx = layer_idx

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for GQA."""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Q, K, V projections with hooks
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states = self.hook_q(query_states)

        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.hook_k(key_states)

        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.hook_v(value_states)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # KV cache
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # GQA: repeat KV heads
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Attention
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# =============================================================================
# Hooked Qwen2 Decoder Layer with 5 HookPoints
# =============================================================================


class HookedQwen2DecoderLayer(nn.Module):
    """Qwen2 decoder layer with HookPoints at residual stream positions."""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HookedQwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # For masking compatibility
        self.attention_type = getattr(config, "attention_type", "full_attention")
        if hasattr(config, "layer_types") and layer_idx < len(config.layer_types):
            self.attention_type = config.layer_types[layer_idx]
        else:
            self.attention_type = "full_attention"

        # TransformerLens HookPoints at all 5 residual stream positions
        self.hook_resid_pre = HookPoint()
        self.hook_resid_pre.layer_idx = layer_idx
        self.hook_attn_out = HookPoint()
        self.hook_attn_out.layer_idx = layer_idx
        self.hook_resid_mid = HookPoint()
        self.hook_resid_mid.layer_idx = layer_idx
        self.hook_mlp_out = HookPoint()
        self.hook_mlp_out.layer_idx = layer_idx
        self.hook_resid_post = HookPoint()
        self.hook_resid_post.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Returns hidden_states (post-full-block) for distillation."""
        # Hook: pre-residual
        residual = hidden_states
        residual = self.hook_resid_pre(residual)

        # LayerNorm + Self Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # Hook: attention output (before residual add)
        hidden_states = self.hook_attn_out(hidden_states)

        # Residual add
        hidden_states = residual + hidden_states
        # Hook: mid-residual (after attention, before MLP)
        hidden_states = self.hook_resid_mid(hidden_states)

        # LayerNorm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # Hook: MLP output (before residual add)
        hidden_states = self.hook_mlp_out(hidden_states)

        # Residual add
        hidden_states = residual + hidden_states
        # Hook: post-residual (final output)
        hidden_states = self.hook_resid_post(hidden_states)

        return hidden_states


# =============================================================================
# Hooked Qwen2 Pre-trained Model base class
# =============================================================================


class HookedQwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HookedQwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]


# =============================================================================
# Hooked Qwen2 Model (base model with hook collection)
# =============================================================================


class HookedQwen2Model(HookedQwen2PreTrainedModel, HookedRootModule):
    """Qwen2 base model with TransformerLens hooks.

    Collects intermediate hidden states (x_t) and post-attention residuals (means)
    for distillation to flow-based models.
    """

    def __init__(self, config: Qwen2Config):
        HookedQwen2PreTrainedModel.__init__(self, config)
        HookedRootModule.__init__(self)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                HookedQwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Check for sliding window attention
        self.has_sliding_layers = hasattr(config, "layer_types") and "sliding_attention" in getattr(config, "layer_types", [])

        # Initialize weights
        self.post_init()

        # SetupTT TransformerLens hook management (discovers all HookPoint instances)
        self.setup()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_hidden_trajectory: bool = False,
        **kwargs,
    ):
        """Forward pass collecting intermediate representations.

        Args:
            return_hidden_trajectory: If True, returns (past_kv, last_hidden, x_t, means)
                where x_t[i] = hidden state after layer i, means[i] = post-attn residual.
                If False, returns standard BaseModelOutputWithPast.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create causal mask
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                from transformers.masking_utils import create_sliding_window_causal_mask
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        # Collect trajectories for distillation
        x_t: List[torch.Tensor] = []

        hidden_states = inputs_embeds
        x_t.append(hidden_states)  # x_t[0] = embeddings

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_mask = causal_mask_mapping.get(
                getattr(decoder_layer, "attention_type", "full_attention"),
                causal_mask_mapping["full_attention"],
            )
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache if use_cache is not None else False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            x_t.append(hidden_states)  # x_t[i+1] = output after layer i

        hidden_states = self.norm(hidden_states)
        past_key_values = past_key_values if use_cache else None

        if return_hidden_trajectory:
            return past_key_values, hidden_states, x_t

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# =============================================================================
# Hooked Qwen2 For Sequence Classification
# =============================================================================


class HookedQwen2ForSequenceClassification(HookedQwen2PreTrainedModel, HookedRootModule):
    """Qwen2 for sequence classification with TransformerLens hooks.

    Wraps HookedQwen2Model with a classification head. Pools on the last
    non-padding token (standard for decoder-only models).
    """

    def __init__(self, config: Qwen2Config):
        HookedQwen2PreTrainedModel.__init__(self, config)
        HookedRootModule.__init__(self)

        self.num_labels = config.num_labels
        self.model = HookedQwen2Model(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        self.post_init()
        self.setup()

    def _get_last_non_pad_token_idx(
        self, input_ids: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """Get index of last non-padding token for each sample in the batch."""
        if self.config.pad_token_id is None:
            return torch.full(
                (input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device
            )
        non_pad_mask = (input_ids != self.config.pad_token_id).to(
            logits.device, torch.int32
        )
        token_indices = torch.arange(
            input_ids.shape[-1], device=logits.device, dtype=torch.int32
        )
        return (token_indices * non_pad_mask).argmax(-1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_hidden_trajectory: bool = False,
        **kwargs,
    ):
        """Forward pass with optional trajectory collection.

        Returns:
            If return_hidden_trajectory=True:
                (logits, x_t) where x_t is a list of per-layer hidden states
            Otherwise:
                SequenceClassifierOutputWithPast
        """
        if return_hidden_trajectory:
            past_kv, last_hidden, x_t = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                return_hidden_trajectory=True,
                **kwargs,
            )
            # Classify from last_hidden (already normed)
            logits = self.score(last_hidden)
            batch_size = logits.shape[0]
            last_idx = self._get_last_non_pad_token_idx(input_ids, logits)
            pooled_logits = logits[
                torch.arange(batch_size, device=logits.device), last_idx
            ]
            return pooled_logits, x_t

        # Standard forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_hidden_trajectory=False,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.score(hidden_states)

        batch_size = logits.shape[0]
        last_idx = self._get_last_non_pad_token_idx(input_ids, logits)
        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), last_idx
        ]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss = nn.functional.mse_loss(
                    pooled_logits.squeeze(), labels.squeeze().float()
                )
            else:
                loss = nn.functional.cross_entropy(
                    pooled_logits, labels.view(-1)
                )

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=None,
        )


def load_pretrained_hooked_qwen2(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    num_labels: int = 2,
    device: str = "cuda",
) -> HookedQwen2ForSequenceClassification:
    """Load a pretrained Qwen2 model with hooks for classification.

    Loads the base Qwen2 weights into the hooked model architecture.
    """
    from transformers import Qwen2ForSequenceClassification, Qwen2Config

    # Load the original model to get config and weights
    config = Qwen2Config.from_pretrained(model_name)
    config.num_labels = num_labels
    config.pad_token_id = config.eos_token_id

    # Create hooked model
    hooked_model = HookedQwen2ForSequenceClassification(config)

    # Load pretrained base weights
    from transformers import Qwen2Model

    pretrained_base = Qwen2Model.from_pretrained(model_name)

    # Transfer weights: embed_tokens, layers, norm, rotary_emb
    hooked_model.model.embed_tokens.load_state_dict(
        pretrained_base.embed_tokens.state_dict()
    )
    hooked_model.model.norm.load_state_dict(
        {"weight": pretrained_base.norm.weight.data}
    )

    # Transfer each layer's weights
    for i, (hooked_layer, pretrained_layer) in enumerate(
        zip(hooked_model.model.layers, pretrained_base.layers)
    ):
        # Attention weights
        hooked_layer.self_attn.q_proj.load_state_dict(
            pretrained_layer.self_attn.q_proj.state_dict()
        )
        hooked_layer.self_attn.k_proj.load_state_dict(
            pretrained_layer.self_attn.k_proj.state_dict()
        )
        hooked_layer.self_attn.v_proj.load_state_dict(
            pretrained_layer.self_attn.v_proj.state_dict()
        )
        hooked_layer.self_attn.o_proj.load_state_dict(
            pretrained_layer.self_attn.o_proj.state_dict()
        )
        # MLP weights
        hooked_layer.mlp.gate_proj.load_state_dict(
            pretrained_layer.mlp.gate_proj.state_dict()
        )
        hooked_layer.mlp.up_proj.load_state_dict(
            pretrained_layer.mlp.up_proj.state_dict()
        )
        hooked_layer.mlp.down_proj.load_state_dict(
            pretrained_layer.mlp.down_proj.state_dict()
        )
        # LayerNorm weights
        hooked_layer.input_layernorm.load_state_dict(
            {"weight": pretrained_layer.input_layernorm.weight.data}
        )
        hooked_layer.post_attention_layernorm.load_state_dict(
            {"weight": pretrained_layer.post_attention_layernorm.weight.data}
        )

    del pretrained_base
    torch.cuda.empty_cache()

    return hooked_model.to(device)


__all__ = [
    "HookedQwen2Model",
    "HookedQwen2ForSequenceClassification",
    "HookedQwen2DecoderLayer",
    "HookedQwen2Attention",
    "load_pretrained_hooked_qwen2",
]
