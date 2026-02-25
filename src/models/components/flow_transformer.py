"""Flow Transformer Text model for distillation.

Replaces TransDiff's Diffusion_Transformer_Text with a flow-matching equivalent.
Instead of predicting (mean, std) for a diagonal Gaussian, this model predicts
a velocity field v_θ(x_t, t) for the continuous OT flow matching framework.

Architecture:
  - Frozen prefix: first `from_layer` Qwen2 decoder layers (copied from teacher)
  - Flow block: shared DiT backbone applied `num_steps` times, predicting velocity
  - Frozen suffix: layers `to_layer` onward + classification head (from teacher)
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Qwen2Model
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2MLP,
    Qwen2RMSNorm,
)
from transformers.masking_utils import create_causal_mask

from src.models.components.dit import DiT


class FlowTransformerText(nn.Module):
    """Flow-matching student model for distilling a fine-tuned Qwen2.

    The architecture mirrors TransDiff's Diffusion_Transformer_Text but replaces
    the Gaussian (mean,std) prediction with velocity field prediction for
    continuous flow matching.

    Args:
        config: Qwen2Config from the teacher model
        dit_depth: Number of DiT blocks in the shared backbone
        num_heads: Number of attention heads in DiT
        mlp_ratio: MLP hidden dim ratio in DiT
        dropout: Dropout rate for velocity head
        num_labels: Number of classification labels
        from_layer: First layer to replace (0-indexed, inclusive)
        to_layer: Last layer to replace (exclusive)
    """

    def __init__(
        self,
        config: Qwen2Config,
        dit_depth: int = 2,
        num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_labels: int = 2,
        from_layer: int = 0,
        to_layer: int = 12,
    ):
        super().__init__()
        # Ensure attention implementation is set (required by newer transformers)
        if not hasattr(config, "_attn_implementation") or config._attn_implementation is None:
            config._attn_implementation = "eager"
        self.config = config
        self.d_model = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.num_steps = to_layer - from_layer

        if num_heads is None:
            num_heads = config.num_attention_heads

        # --- Frozen prefix: first `from_layer` Qwen2 decoder layers ---
        config_prefix = deepcopy(config)
        config_prefix.num_hidden_layers = from_layer
        self.qwen_prefix = Qwen2Model(config_prefix)
        self.qwen_prefix.norm = nn.Identity()  # Skip final norm in prefix

        # --- Flow block: shared DiT backbone ---
        self.dit = DiT(
            hidden_size=self.d_model,
            depth=dit_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # Velocity prediction head: predicts v_θ(x_t, t)
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        # --- Frozen suffix components ---
        # MLP from the last replaced layer (for the transition from flow → suffix)
        self.transition_mlp = Qwen2MLP(config)
        self.transition_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Suffix decoder layers: to_layer → end
        self.suffix_layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(to_layer, self.num_layers)
            ]
        )

        # Final norm and classification head
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.score = nn.Linear(config.hidden_size, num_labels, bias=False)

    def predict_velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity field v_θ(x_t, t).

        Args:
            x: (B, S, D) noisy hidden states at time t
            t: (B,) timestep values

        Returns:
            (B, S, D) predicted velocity
        """
        dit_out = self.dit(x, t)
        velocity = self.velocity_head(dit_out)
        return velocity

    def flow_forward(
        self, x_start: torch.Tensor, num_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the flow steps (Euler integration) during inference.

        Args:
            x_start: (B, S, D) starting hidden state (output of prefix)
            num_steps: Number of Euler steps (defaults to self.num_steps)

        Returns:
            (final_x, velocities) where velocities is a list of predicted v at each step
        """
        if num_steps is None:
            num_steps = self.num_steps

        x = x_start
        velocities = []
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.full(
                (x.shape[0],), step * dt, device=x.device, dtype=x.dtype
            )
            v = self.predict_velocity(x, t)
            velocities.append(v)
            x = x + v * dt  # Euler step

        return x, velocities

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        train_flow: bool = False,
        x_t_teacher: Optional[List[torch.Tensor]] = None,
    ):
        """Forward pass.

        Args:
            input_ids: (B, S) input token IDs
            attention_mask: (B, S) attention mask
            train_flow: If True, takes teacher x_t and returns velocities for flow loss
            x_t_teacher: Teacher's hidden states at each replaced layer (for training)

        Returns:
            If train_flow=True:
                List of predicted velocities at each step
            Otherwise:
                (pooled_logits, velocities) for inference
        """
        if train_flow and x_t_teacher is not None:
            # Training mode: predict velocities given teacher trajectories
            velocities = []
            x = x_t_teacher[0]  # Start from the first hidden state in the replaced range
            dt = 1.0 / self.num_steps

            for step in range(self.num_steps):
                t = torch.full(
                    (x.shape[0],), step * dt, device=x.device, dtype=x.dtype
                )
                v = self.predict_velocity(x, t)
                velocities.append(v)
                # Use the predicted velocity to step forward
                x = x + v * dt

            return velocities

        # --- Inference mode ---
        # 1. Run frozen prefix
        inputs_embeds = self.qwen_prefix.embed_tokens(input_ids)
        position_ids = torch.arange(
            input_ids.size(1), device=input_ids.device
        ).unsqueeze(0)
        position_embeddings = self.qwen_prefix.rotary_emb(inputs_embeds, position_ids)

        prefix_output = self.qwen_prefix(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        x = prefix_output.last_hidden_state

        # 2. Run flow steps (replace layers from_layer to to_layer-1)
        x, velocities = self.flow_forward(x)

        # 3. Transition: apply MLP + residual from last replaced layer
        x_layer = x + self.transition_mlp(self.transition_layernorm(x))

        # 4. Run frozen suffix layers
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": torch.arange(
                input_ids.shape[1], device=input_ids.device
            ),
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }

        for layer in self.suffix_layers:
            x_layer = layer(
                x_layer,
                attention_mask=causal_mask_mapping["full_attention"],
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            # In transformers >= 4.57, DecoderLayer returns tensor directly;
            # older versions return a tuple with hidden_states at index 0
            if isinstance(x_layer, tuple):
                x_layer = x_layer[0]

        # 5. Classify: pool on last non-pad token
        batch_size = x_layer.shape[0]
        logits = self.score(self.norm(x_layer))

        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        else:
            non_pad_mask = (input_ids != self.config.pad_token_id).to(
                logits.device, torch.int32
            )
            token_indices = torch.arange(
                input_ids.shape[-1], device=logits.device, dtype=torch.int32
            )
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), last_non_pad_token
        ]

        return pooled_logits, velocities

    def load_teacher_weights(self, teacher_model) -> None:
        """Transfer weights from a fine-tuned HookedQwen2ForSequenceClassification teacher.

        Copies:
          - Prefix layers (embed_tokens, rotary_emb, layers 0..from_layer-1)
          - Transition MLP + LayerNorm from the last replaced layer
          - Suffix layers (to_layer..end)
          - Final norm and classification head
        """
        teacher_base = teacher_model.model  # HookedQwen2Model

        # 1. Token embeddings
        self.qwen_prefix.embed_tokens.load_state_dict(
            teacher_base.embed_tokens.state_dict()
        )

        # 2. Rotary embeddings
        self.qwen_prefix.rotary_emb.load_state_dict(
            teacher_base.rotary_emb.state_dict()
        )

        # 3. Prefix decoder layers
        for i in range(self.from_layer):
            src_layer = teacher_base.layers[i]
            dst_layer = self.qwen_prefix.layers[i]
            # Copy attention
            dst_layer.self_attn.q_proj.load_state_dict(
                src_layer.self_attn.q_proj.state_dict()
            )
            dst_layer.self_attn.k_proj.load_state_dict(
                src_layer.self_attn.k_proj.state_dict()
            )
            dst_layer.self_attn.v_proj.load_state_dict(
                src_layer.self_attn.v_proj.state_dict()
            )
            dst_layer.self_attn.o_proj.load_state_dict(
                src_layer.self_attn.o_proj.state_dict()
            )
            # Copy MLP
            dst_layer.mlp.gate_proj.load_state_dict(
                src_layer.mlp.gate_proj.state_dict()
            )
            dst_layer.mlp.up_proj.load_state_dict(
                src_layer.mlp.up_proj.state_dict()
            )
            dst_layer.mlp.down_proj.load_state_dict(
                src_layer.mlp.down_proj.state_dict()
            )
            # Copy layernorms
            dst_layer.input_layernorm.weight.data.copy_(
                src_layer.input_layernorm.weight.data
            )
            dst_layer.post_attention_layernorm.weight.data.copy_(
                src_layer.post_attention_layernorm.weight.data
            )

        # 4. Transition MLP from the to_layer-1 (last replaced layer)
        last_replaced = teacher_base.layers[self.to_layer - 1]
        self.transition_mlp.gate_proj.load_state_dict(
            last_replaced.mlp.gate_proj.state_dict()
        )
        self.transition_mlp.up_proj.load_state_dict(
            last_replaced.mlp.up_proj.state_dict()
        )
        self.transition_mlp.down_proj.load_state_dict(
            last_replaced.mlp.down_proj.state_dict()
        )
        self.transition_layernorm.weight.data.copy_(
            last_replaced.post_attention_layernorm.weight.data
        )

        # 5. Suffix decoder layers
        for i in range(self.to_layer, self.num_layers):
            src_layer = teacher_base.layers[i]
            dst_layer = self.suffix_layers[i - self.to_layer]
            dst_layer.load_state_dict(src_layer.state_dict(), strict=False)

        # 6. Final norm
        self.norm.weight.data.copy_(teacher_base.norm.weight.data)

        # 7. Classification head
        self.score.load_state_dict(teacher_model.score.state_dict())

        print(
            f"[FlowTransformerText] Loaded teacher weights: "
            f"prefix={self.from_layer} layers, "
            f"flow={self.num_steps} steps, "
            f"suffix={self.num_layers - self.to_layer} layers"
        )

    def freeze_non_flow_params(self) -> None:
        """Freeze everything except the DiT backbone and velocity head (Stage 1)."""
        for param in self.qwen_prefix.parameters():
            param.requires_grad = False
        for param in self.transition_mlp.parameters():
            param.requires_grad = False
        for param in self.transition_layernorm.parameters():
            param.requires_grad = False
        for param in self.suffix_layers.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
        for param in self.score.parameters():
            param.requires_grad = False

        # Keep flow components trainable
        for param in self.dit.parameters():
            param.requires_grad = True
        for param in self.velocity_head.parameters():
            param.requires_grad = True

    def freeze_flow_unfreeze_suffix(self) -> None:
        """Freeze flow components, unfreeze suffix + classifier (Stage 2)."""
        for param in self.dit.parameters():
            param.requires_grad = False
        for param in self.velocity_head.parameters():
            param.requires_grad = False

        # Unfreeze suffix
        for param in self.transition_mlp.parameters():
            param.requires_grad = True
        for param in self.transition_layernorm.parameters():
            param.requires_grad = True
        for param in self.suffix_layers.parameters():
            param.requires_grad = True
        for param in self.norm.parameters():
            param.requires_grad = True
        for param in self.score.parameters():
            param.requires_grad = True


__all__ = ["FlowTransformerText"]
