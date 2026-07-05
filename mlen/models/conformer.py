from typing import Optional, Tuple
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import BaseModelOutput

from muq.muq.modules.flash_conformer import (
    Wav2Vec2ConformerConvolutionModule,
    Wav2Vec2ConformerFeedForward,
    Wav2Vec2ConformerPositionalConvEmbedding,
    Wav2Vec2ConformerSelfAttention,
    Wav2Vec2ConformerRelPositionalEmbedding,
    Wav2Vec2ConformerRotaryPositionalEmbedding,
)

from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerSelfAttention as Wav2Vec2ConformerSelfAttentionOrig
from transformers.models.wav2vec2_conformer.configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig

class CondConformerConfig(Wav2Vec2ConformerConfig):
    model_type = "cond-wav2vec2-conformer"

    def __init__(self, cond_dim=None, flash_attention=True, **kwargs):
        super().__init__(**kwargs)
        self.cond_dim = cond_dim
        self.flash_attention = flash_attention


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, feature_dim, cond_dim=None,eps=1e-5):
        super().__init__()
        self.feature_dim = feature_dim
        self.eps = eps
        self.conditional = cond_dim is not None
        if self.conditional:
            self.cond_mlp = nn.Linear(cond_dim, feature_dim * 2)
            self.gamma = nn.Parameter(torch.ones(feature_dim))
            self.beta = nn.Parameter(torch.zeros(feature_dim))
            nn.init.zeros_(self.cond_mlp.weight)
            nn.init.zeros_(self.cond_mlp.bias)
        else:
            self.layer_norm = nn.LayerNorm(feature_dim, eps=self.eps)

    def forward_conditional(self, x, cond=None):
        # x: (batch_size, seq_len, feature_dim)
        # cond: (batch_size, cond_dim)
        assert cond is not None, "Conditioning vector is required for conditional layer normalization."
        cond_params = self.cond_mlp(cond)  # (batch_size, feature_dim * 2)
        gamma, beta = cond_params.chunk(2, dim=-1)  # Each is (batch_size, feature_dim)
        x = F.layer_norm(x, (self.feature_dim,), eps=self.eps, weight=self.gamma, bias=self.beta)  # (batch_size, seq_len, feature_dim)
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(self, x, cond=None):
        if self.conditional:
            return self.forward_conditional(x, cond)
        else:
            return self.layer_norm(x)


class CondConformerSelfAttention(Wav2Vec2ConformerSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # self-attention mechanism
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        if self.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'"
                )
            query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        dropout_p = self.dropout_p if self.training else 0.0
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=self.is_causal,
            )
        probs = None

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        hidden_states = self.linear_out(hidden_states)

        return hidden_states, probs




class CondConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    def __init__(self, config: CondConformerConfig):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout
        flash_attention = config.flash_attention
        self.cond_dim = getattr(config, "cond_dim", None)
        self.layernorm = partial(AdaptiveLayerNorm, cond_dim=self.cond_dim)

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = Wav2Vec2ConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = self.layernorm(embed_dim)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        if flash_attention:
            self.self_attn = CondConformerSelfAttention(config)
        else:
            self.self_attn = Wav2Vec2ConformerSelfAttentionOrig(config)

        # Conformer Convolution
        self.conv_module = Wav2Vec2ConformerConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn2 = Wav2Vec2ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states,
        cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        hidden_states = hidden_states

        # 1. Feed-Forward 1 layer
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states, cond)
        hidden_states, attn_weigts = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states)
        hidden_states = residual + hidden_states

        # 4. Feed-Forward 2 Layer
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, attn_weigts


class CondConformerEncoder(nn.Module):
    def __init__(self, config, is_causal=False):
        super().__init__()
        config.is_causal = is_causal
        self.config = config

        if config.position_embeddings_type == "relative":
            self.embed_positions = Wav2Vec2ConformerRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            self.embed_positions = Wav2Vec2ConformerRotaryPositionalEmbedding(config)
        else:
            self.embed_positions = None

        # self.pos_conv_embed = Wav2Vec2ConformerPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [CondConformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        cond=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states[~attention_mask] = 0.0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        hidden_states = self.dropout(hidden_states)

        if self.embed_positions is not None:
            relative_position_embeddings = self.embed_positions(hidden_states)
        else:
            relative_position_embeddings = None

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        cond,
                        attention_mask,
                        relative_position_embeddings,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        cond=cond,
                        attention_mask=attention_mask,
                        relative_position_embeddings=relative_position_embeddings,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
