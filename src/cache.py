import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from dataclasses import dataclass
from contextlib import contextmanager

from src.frame import Frame, FrameDelta
from src.utils import certainty_density, nucleus_select
from scipy.optimize import minimize_scalar
from typing import Callable, Protocol, Tuple

class LLADARotaryEmbCallable(Protocol):
    def __call__(self, q: torch.Tensor, k: torch.Tensor, q_position_ids: torch.Tensor | None, kv_position_ids: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        ...
class DreamRotaryEmbCallable(Protocol):
    def __call__(self, x: torch.Tensor, position_ids: torch.Tensor | None) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

@dataclass
class AttentionContext:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    input_x: torch.Tensor
    o: torch.Tensor | None = None  # assigned from model

    # if config._attn_implementation == "eager", this will be the attention weights
    # of shape (B, nh, q_len, seq_len)
    attn_weight: torch.Tensor | None = None

    # if you select a subset of qkv, you must also provide these properties
    q_position_ids: torch.Tensor | None = None
    kv_position_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    @property
    def normalized_input(self) -> torch.Tensor:
        if self.norm_fn is not None:
            return self.norm_fn(self.input_x)
        return self.input_x

    @classmethod
    def select_position_ids(
        cls,
        position_ids: torch.Tensor | None = None,
        q_mask: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ):
        q_position_ids, kv_position_ids = position_ids, position_ids
        if position_ids is not None:
            if q_mask is not None:
                q_position_ids = position_ids[q_mask].view(q_mask.size(0), -1)
            if kv_mask is not None:
                kv_position_ids = position_ids[kv_mask].view(kv_mask.size(0), -1)
        return q_position_ids, kv_position_ids

    @classmethod
    def convert_attention_mask(
        cls,
        attention_mask: torch.Tensor | None,
        dtype: torch.dtype,
        query_length: int | None = None,
        key_value_length: int | None = None,
    ):
        if attention_mask is not None:
            if attention_mask.dim() == 2:  # (B, kv_len) -> (B, 1, q_len, kv_len)
                try:
                    attention_mask = attention_mask[:, None, None, :].expand(
                        attention_mask.size(0),
                        1,
                        query_length or attention_mask.size(1),
                        key_value_length or attention_mask.size(1),
                    )
                except Exception:
                    # if there is an exception raised, we assume the subclass will process attention mask properly
                    return attention_mask
            elif attention_mask.dim() != 4:
                raise ValueError(
                    f"Expected attention_mask to have 2 or 4 dimensions, but got {attention_mask.dim()}."
                )

            attention_mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min

        return attention_mask


@dataclass
class FFNContext:
    x: torch.Tensor
    residual: torch.Tensor
    ffn_out: torch.Tensor | None = None  # assigned from model


@dataclass
class ModelForwardContext:
    x: torch.Tensor
    logits: torch.Tensor | None = None  # assigned from model

class dCache:
    """
    A cache structure used during diffusion language models decoding to reuse intermediate states.
    """

    def __init__(self, model_config):
        self.model_config = model_config
        self.active_q_mask: torch.Tensor | None = None
        self.count = 0
        self.ratio_sum = 0.0

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        """
        A context manager that modifies the input/output tensors for the forward pass of model layers. In this function,
        it can select a subset to feed into model layers, but it must recover the final logits to be the shape of (batch_size, seq_len, vocab_size).

        Args:
            x (torch.Tensor): The input tensor after embedding layers, with shape (batch_size, seq_len, d_model).

        """
        input_shape = x.shape
        ctx = ModelForwardContext(x=x)

        yield ctx

        if ctx.logits is None:
            raise RuntimeError("The logits are not set in the context.")

        if ctx.logits.shape[:2] != input_shape[:2]:
            raise RuntimeError(
                f"The logits shape {ctx.logits.shape!r} is not compatible with the input shape {input_shape!r}."
            )
    
    def accumulate_recompute_ratio(self, q_len, v_len): # this should be called at each layer
        self.count += 1
        ratio = q_len / v_len
        self.ratio_sum += ratio
    def get_recompute_ratio(self):
        if self.count == 0:
            return 0.0
        return self.ratio_sum / self.count

    def adjust_kv(self, ctx, layer_idx):
        """Adjust key and value tensors during attention sublayer after rotary embedding."""
        ...

    @contextmanager
    def attention(
        self,
        layer_idx: int,
        input_x: torch.Tensor,
        norm_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        """
        A context manager that modifies the input/output tensors for attention computation. In this function, it should
        compute query, key, and value projections, and yield a `AttentionContext` object that stores `q`, `k`, `v`, `residual` tensors.
        The outer code should handle the actual attention computation, then add `o` to the context object.

        Args:
            layer_idx (int): The index of the layer to update.
            input_x (torch.Tensor): The input tensor before pre-layer norm, with shape (batch_size, seq_len, d_model).
            norm_fn (Callable[[torch.Tensor], torch.Tensor] | None): layer normalization function to be applied on input_x before projections.
            q_proj (nn.Linear): The query projection layer.
            k_proj (nn.Linear): The key projection layer.
            v_proj (nn.Linear): The value projection layer.
            attention_mask (torch.Tensor, *optional*): An optional attention mask, with shape (batch_size, seq_len, seq_len).
            position_ids (torch.Tensor, *optional*): An optional tensor of position IDs, with shape (batch_size, seq_len).
        """
        x = norm_fn(input_x) if norm_fn is not None else input_x
        if x.numel() > 0:
            q, k, v = q_proj(x), k_proj(x), v_proj(x)
        else:
            q, k, v = x[:, 0:0], x[:, 0:0], x[:, 0:0]
        ctx = AttentionContext(
            q=q,
            k=k,
            v=v,
            input_x=input_x,
            attention_mask=AttentionContext.convert_attention_mask(
                attention_mask,
                dtype=q.dtype,
                query_length=q.shape[1],
                key_value_length=k.shape[1],
            ),
            q_position_ids=position_ids,
            kv_position_ids=position_ids,
        )
        yield ctx

        if ctx.o is None:
            raise RuntimeError("The attention output is not set in the context.")

        if ctx.input_x.shape != ctx.o.shape:
            raise RuntimeError(
                f"The attention output shape {ctx.o.shape!r} is not compatible with the residual shape {ctx.input_x.shape!r}."
            )

    @contextmanager
    def ffn(self, layer_idx: int, x: torch.Tensor, residual: torch.Tensor):
        """
        A context manager that modifies the input/output tensors for feed-forward network computation. In this function,
        it should yield a `FFNContext` object that stores `x` and `residual` tensors. The outer code should handle the
        actual feed-forward network computation, then add `ffn_out` to the context object.

        Args:
            layer_idx (int): The index of the layer to update.
            x (torch.Tensor): The input tensor after self-attention, with shape (batch_size, seq_len, d_model).
            residual (torch.Tensor): The residual of the feed-forward network, which is obtained before the layer norm.
        """
        ctx = FFNContext(x=x, residual=residual)
        yield ctx

        if ctx.ffn_out is None:
            raise RuntimeError(
                "The feed-forward network output is not set in the context."
            )

        if ctx.residual.shape != ctx.ffn_out.shape:
            raise RuntimeError(
                f"The feed-forward network output shape {ctx.ffn_out.shape!r} is not compatible with the residual shape {ctx.residual.shape!r}."
            )

    def on_inference_start(self, frame: Frame, window_or_block_size: int | None = None):
        """
        Called at the start of inference to initialize the cache with the current frame.

        Args:
            frame (Frame): The initial frame before any generation steps.
            window_or_block_size (int | None): The size of the sliding window or block used in semi-autoregressive decoding.
        """
        ...

    def on_inference_end(self, frame: Frame):
        """
        Called at the end of inference to finalize the cache with the current frame.

        Args:
            frame (Frame): The final frame after all generation steps.
        """
        ...

    def on_step_start(self, block_mask: torch.Tensor | None, frame: Frame):
        """
        Called at the start of each generation step to update the cache with the current frame.

        Args:
            block_mask (torch.Tensor | None): A boolean mask indicating which positions in the block are active.
                It can be None if block semi-autoregressive decoding is not used.
            frame (Frame): The frame before applying the delta.
        """
        ...

    def on_step_end(
        self, block_mask: torch.Tensor | None, frame: Frame, delta: FrameDelta
    ):
        """
        Called at the end of each generation step to update the cache with the current frame and delta.

        Args:
            block_mask (torch.Tensor | None): A boolean mask indicating which positions in the block are active.
                It can be None if block semi-autoregressive decoding is not used.
            frame (Frame): The frame before applying the delta.
            delta (FrameDelta): The delta to apply to the frame.
        """
        ...

    def on_block_start(self, block_mask: torch.Tensor | None, frame: Frame):
        """
        Called at the start of each block to update the cache with the current frame.

        Args:
            block_mask (torch.Tensor | None): A boolean mask indicating which positions in the block are active.
                It can be None if block semi-autoregressive decoding is not used.
            frame (Frame): The frame before applying any deltas in the block.
        """
        ...

    def on_block_end(
        self, block_mask: torch.Tensor | None, frame: Frame, deltas: list[FrameDelta]
    ):
        """
        Called at the end of each block to update the cache with the current frame and deltas.

        Args:
            block_mask (torch.Tensor | None): A boolean mask indicating which positions in the block are active.
                It can be None if block semi-autoregressive decoding is not used.
            frame (Frame): The frame before applying all deltas in the block.
            deltas (list[FrameDelta]): The list of deltas applied in the block.
        """
        ...

    @property
    def mask_token_id(self):
        return int(os.environ["MASK_TOKEN_ID"])


class d2Cache(dCache):

    def __init__(
        self,
        model_config,
        rollout_p: float = 0.1,
        current_k: int = 60,
        sigma: float = 10.0,
    ):
        super().__init__(model_config)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self._conf_cache: torch.Tensor | None = None
        self.rollout_p = rollout_p
        self.current_k = current_k
        self.sigma = sigma

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        with super().model_forward(x=x) as ctx:
            B, T, C = x.shape
            if self.active_q_mask is not None:
                ctx.x = x[self.active_q_mask].view(B, -1, C)
            yield ctx

            if self.active_q_mask is not None:
                assert ctx.logits is not None
                ctx.logits = torch.zeros(
                    (B, T, ctx.logits.size(-1)),
                    dtype=ctx.logits.dtype,
                    device=ctx.logits.device,
                ).masked_scatter_(self.active_q_mask.unsqueeze(-1), ctx.logits)

    @contextmanager
    def attention(
        self,
        layer_idx: int,
        x: torch.Tensor,
        residual: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        B, T, C = x.shape

        with super().attention(
            layer_idx, x, residual, q_proj, k_proj, v_proj, attention_mask, position_ids
        ) as ctx:
            # with self.timer.measure("kv_update"):
            if len(self.key_cache) <= layer_idx:
                # the first forward pass, store states as cache
                self.key_cache.append(ctx.k)
                self.value_cache.append(ctx.v)
            else:
                self.key_cache[layer_idx][self.active_q_mask] = ctx.k
                self.value_cache[layer_idx][self.active_q_mask] = ctx.v
                ctx.k = self.key_cache[layer_idx]
                ctx.v = self.value_cache[layer_idx]
                ctx.q_position_ids, ctx.kv_position_ids = (
                    AttentionContext.select_position_ids(
                        position_ids, self.active_q_mask
                    )
                )
                B, T, C = ctx.k.shape

            ctx.attention_mask = AttentionContext.convert_attention_mask(
                attention_mask,
                dtype=ctx.k.dtype,
                query_length=ctx.q.shape[1],
                key_value_length=self.value_cache[layer_idx].shape[1],
            )

            yield ctx

            assert (
                ctx.attn_weight is not None
            ), 'The attention weights must be outputed, make sure you\'ve set attn_implementation="eager"'
            if layer_idx == 0:
                self._attn_rollout = torch.eye(
                    T, device=x.device, dtype=x.dtype
                ).expand(B, -1, -1)
            # with self.timer.measure("attentionrollout_overhead"):
            self.accumulate_attn_rollout(ctx.attn_weight)

    def accumulate_attn_rollout(self, attn_scores: torch.Tensor):
        """
        Computes one step of the Attention Rollout for attention maps.
        In this setup, only a subset of tokens act as queries.

        Args:
            attn_scores (torch.Tensor):
                Attention scores for the current layer, with shape of (B, num_heads, q_len, seq_len).
        """
        B, n_heads, q_len, seq_len = attn_scores.shape
        device, dtype = attn_scores.device, attn_scores.dtype

        # inject the rectangular attention map into the rows
        effective_attn = torch.eye(seq_len, device=device, dtype=dtype).expand(
            B, -1, -1
        )
        effective_attn[self.active_q_mask] = attn_scores.mean(dim=1).reshape(
            -1, seq_len
        )

        residual_attn = effective_attn + torch.eye(seq_len, device=device, dtype=dtype)
        # re-normalize the matrix so that each row sums to 1
        residual_attn = residual_attn / residual_attn.sum(dim=-1, keepdim=True)

        self._attn_rollout = residual_attn @ self._attn_rollout

    def on_step_end(
        self, block_mask: torch.Tensor | None, frame: Frame, delta: FrameDelta
    ):
        # with self.timer.measure("position_selection_overhead"):
        confidence = delta.confidence
        assert confidence is not None
        B, G_old = frame.generated_tokens.shape
        B, P = frame.prompts.shape
        new_frame = frame.apply_delta(delta)
        device = confidence.device

        if self._conf_cache is None:
            self._conf_cache = confidence

        # prepare active mask for query.
        # 1. for the positions that are not unmasked, we only calculate those have large score variance
        # (B, G_old)
        remaining_mask = new_frame.generated_tokens == self.mask_token_id
        if self.active_q_mask is not None:
            # only position where are selected at previous step and are still masked
            # can produce valid confidence scores
            self._conf_cache = torch.where(
                self.active_q_mask[:, P:] & frame.generated_tokens
                == self.mask_token_id,
                confidence,
                self._conf_cache,
            )
        density = certainty_density(~remaining_mask, self.sigma)
        _, indices = torch.topk(
            torch.where(remaining_mask, self._conf_cache * density, -torch.inf),
            k=min(self.current_k, remaining_mask.size(-1)),
            dim=-1,
        )
        selected_mask = (
            torch.zeros_like(remaining_mask, dtype=torch.bool).scatter_(
                1, indices, True
            )
            & remaining_mask
        )
        # if model is dream, we need to retain the token before masked tokens
        if self.model_config.model_type.lower() == "dream":
            response_mask = F.pad(selected_mask[:, 1:], (0, 1), value=False)
        else:
            response_mask = selected_mask
        # 2. recompute all new generated tokens, as they transform from mask to real tokens
        transfer_src_index = (
            delta.transfer_src_index
            if delta.transfer_src_index is not None
            else delta.transfer_index
        )
        lengths = torch.tensor([ti.numel() for ti in transfer_src_index], device=device)
        row_indices = torch.repeat_interleave(
            torch.arange(B, device=confidence.device), lengths
        )
        col_indices = torch.cat(transfer_src_index)  # type: ignore
        response_mask[row_indices, col_indices] = True

        q_mask = torch.cat(
            [
                torch.zeros_like(frame.prompts, dtype=torch.bool),
                response_mask,
            ],
            dim=-1,
        )

        # 3. for other tokens, select top-k tokens based on attention rollout
        global_importance = self._attn_rollout.sum(dim=1)
        q_mask |= nucleus_select(global_importance, self.rollout_p, mask=~q_mask)
        if self.model_config.model_type.lower() == "dream":
            # if the first mask token is selected, we need to select the token before it
            # i.e., the last prompt token
            q_mask[:, P - 1] = selected_mask[:, 0]  # type: ignore

        # TODO: make sure each sequence has the same number of active tokens
        self.active_q_mask = q_mask


class PrefixCache(dCache):

    def __init__(self, model_config, use_dual: bool = False):
        super().__init__(model_config)
        self.use_dual = use_dual

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        with super().model_forward(x=x) as ctx:
            B, T, C = x.shape
            if self.active_q_mask is not None:
                ctx.x = x[self.active_q_mask].view(B, -1, C)
            yield ctx

            if self.active_q_mask is not None:
                assert ctx.logits is not None
                ctx.logits = torch.zeros(
                    (B, T, ctx.logits.size(-1)),
                    dtype=ctx.logits.dtype,
                    device=ctx.logits.device,
                ).masked_scatter_(self.active_q_mask.unsqueeze(-1), ctx.logits)

    @contextmanager
    def attention(
        self,
        layer_idx: int,
        input_x: torch.Tensor,
        norm_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        with super().attention(
            layer_idx, input_x, norm_fn, q_proj, k_proj, v_proj, attention_mask, position_ids
        ) as ctx:
            if len(self.key_cache) <= layer_idx:
                # the first forward pass, store states as cache
                self.key_cache.append(ctx.k)
                self.value_cache.append(ctx.v)
            else:
                self.key_cache[layer_idx][self.active_q_mask] = ctx.k
                self.value_cache[layer_idx][self.active_q_mask] = ctx.v
                ctx.k = self.key_cache[layer_idx]
                ctx.v = self.value_cache[layer_idx]
                ctx.q_position_ids, ctx.kv_position_ids = (
                    AttentionContext.select_position_ids(
                        position_ids, self.active_q_mask
                    )
                )

            ctx.attention_mask = AttentionContext.convert_attention_mask(
                attention_mask,
                dtype=ctx.k.dtype,
                query_length=ctx.q.shape[1],
                key_value_length=self.value_cache[layer_idx].shape[1],
            )

            yield ctx

    def on_step_end(
        self, block_mask: torch.Tensor | None, frame: Frame, delta: FrameDelta
    ):
        assert (
            block_mask is not None
        ), "PrefixCache only works with block semi-autoregressive decoding."
        if self.active_q_mask is None:
            q_mask = torch.cat(
                [
                    torch.zeros_like(frame.prompts, dtype=torch.bool),
                    block_mask,
                ],
                dim=-1,
            )
            if not self.use_dual:
                block_start = int(block_mask[0].int().argmax().item() + 1)
                q_mask[:, frame.prompts.size(-1) + block_start :] = True

            if self.model_config.model_type.lower() == "dream":
                q_mask = F.pad(q_mask[:, 1:], (0, 1), value=False)

            self.active_q_mask = q_mask

    def on_block_start(self, block_mask: torch.Tensor | None, frame: Frame):
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.active_q_mask: torch.Tensor | None = None


class dLLMCache(dCache):
    def __init__(self, model_config, kp: int = 50, kr: int = 2, rou: float = 0.25):
        super().__init__(model_config)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.attn_cache: list[torch.Tensor] = []
        self.ffn_cache: list[torch.Tensor] = []
        self.kp = kp
        self.kr = kr
        self.rou = rou

    @contextmanager
    def attention(
        self,
        layer_idx: int,
        input_x: torch.Tensor,
        norm_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        refresh_prompt = self.refresh_prompt or layer_idx == 0
        refresh_response = self.refresh_response or layer_idx == 0
        x = norm_fn(input_x) if norm_fn is not None else input_x
        # select prompt or/and response part to feed into the projections
        x_prompt = x[:, : self._prompt_length]
        x_response = x[:, self._prompt_length :]
        x = x[:, 0:0]  # make it empty
        refresh_index = torch.tensor([], device=input_x.device, dtype=torch.long)
        if refresh_prompt:
            x = x_prompt
            refresh_index = torch.arange(self._prompt_length, device=input_x.device)

        if self.rou > 0 or refresh_response:
            x = torch.cat([x, x_response], dim=1)
            if refresh_response:
                refresh_index = torch.cat(
                    [
                        refresh_index,
                        self._prompt_length
                        + torch.arange(x_response.size(1), device=input_x.device),
                    ]
                )
        refresh_index = refresh_index.unsqueeze(0).expand(x.size(0), -1)

        B, T, C = input_x.shape
        # if response part needs to be refreshed or adaptive refreshing is disabled or it is the first
        # forward pass, we calculate all projections
        q = torch.empty((B, 0, q_proj.out_features), dtype=x.dtype, device=input_x.device)
        k = torch.empty((B, 0, k_proj.out_features), dtype=x.dtype, device=input_x.device)
        v = torch.empty((B, 0, v_proj.out_features), dtype=x.dtype, device=input_x.device)
        if refresh_response or self.rou == 0 or len(self.key_cache) <= layer_idx:
            if x.numel() > 0:
                q, k, v = q_proj(x), k_proj(x), v_proj(x)
        else:
            if refresh_prompt:
                x_prompt = x[:, : self._prompt_length]
                x_response = x[:, self._prompt_length :]
                q, k, v = q_proj(x_prompt), k_proj(x_prompt), v_proj(x_prompt)
            else:
                x_response = x

            # refresh response part adaptively
            v_response = v_proj(x_response)
            num_replace = int(x_response.size(1) * self.rou)
            cos_sim = F.cosine_similarity(
                v_response,
                self.value_cache[layer_idx][:, self._prompt_length :],
                dim=-1,
            )
            refresh_index_response = torch.topk(
                cos_sim, largest=False, k=num_replace
            ).indices

            selected_x_response = torch.gather(
                x_response, 1, refresh_index_response.unsqueeze(-1).expand(-1, -1, C)
            )
            q = torch.cat([q, q_proj(selected_x_response)], dim=1)
            k = torch.cat([k, k_proj(selected_x_response)], dim=1)
            v = torch.cat([v, v_response], dim=1)

        # update cache
        if len(self.key_cache) <= layer_idx:
            # the first forward pass, store states as cache
            self.key_cache.append(k)
            self.value_cache.append(v)
            q_position_ids = position_ids
        else:
            if refresh_prompt:
                self.key_cache[layer_idx][:, : self._prompt_length] = k[
                    :, : self._prompt_length
                ]
                self.value_cache[layer_idx][:, : self._prompt_length] = v[
                    :, : self._prompt_length
                ]
                prompt_offset = self._prompt_length
            else:
                prompt_offset = 0

            q_position_ids = (
                position_ids[:, :prompt_offset] if position_ids is not None else None
            )

            if self.rou > 0 or refresh_response:
                if refresh_response:
                    # if adaptive refreshing is disabled, we refresh all response
                    refresh_index_response = (
                        torch.arange(x_response.size(1)).unsqueeze(0).expand(B, -1)
                    )

                refresh_index_response = refresh_index_response + self._prompt_length  # type: ignore

                row_indices = torch.arange(B).unsqueeze(-1).expand_as(refresh_index_response)  # type: ignore
                self.key_cache[layer_idx][row_indices, refresh_index_response] = k[
                    :, prompt_offset:
                ]
                # note that for value states, we recompute all even we are using adaptive refreshing
                self.value_cache[layer_idx][:, self._prompt_length :] = v[
                    :, prompt_offset:
                ]

                if not refresh_response:
                    # we've concatenated index before if refresh_response is true
                    refresh_index = torch.cat([refresh_index, refresh_index_response], dim=-1)  # type: ignore

                if q_position_ids is not None:
                    assert position_ids is not None
                    q_position_ids = torch.cat(
                        [
                            q_position_ids,
                            position_ids[row_indices, refresh_index_response],
                        ],
                        dim=-1,
                    )

        self._refresh_index = refresh_index
        ctx = AttentionContext(
            q=q,
            k=self.key_cache[layer_idx],
            v=self.value_cache[layer_idx],
            input_x=input_x,
            attention_mask=AttentionContext.convert_attention_mask(
                attention_mask,
                dtype=q.dtype,
                query_length=q.shape[1],
                key_value_length=self.key_cache[layer_idx].shape[1],
            ),
            q_position_ids=q_position_ids,
            kv_position_ids=position_ids,
        )

        yield ctx

        assert ctx.o is not None
        if len(self.attn_cache) <= layer_idx:
            self.attn_cache.append(ctx.o)
        else:
            if ctx.o.numel() > 0:
                self.attn_cache[layer_idx].scatter_(
                    1, refresh_index.unsqueeze(-1).expand(-1, -1, C), ctx.o
                )

        ctx.o = self.attn_cache[layer_idx]

    @contextmanager
    def ffn(self, layer_idx: int, x: torch.Tensor, residual: torch.Tensor):
        B, _, C = x.shape
        row_indices = torch.arange(B).unsqueeze(-1).expand_as(self._refresh_index)
        x = x[row_indices, self._refresh_index]
        ctx = FFNContext(x=x, residual=residual)

        yield ctx

        assert ctx.ffn_out is not None
        if len(self.ffn_cache) <= layer_idx:
            self.ffn_cache.append(ctx.ffn_out)
        else:
            self.ffn_cache[layer_idx].scatter_(
                1, self._refresh_index.unsqueeze(-1).expand(-1, -1, C), ctx.ffn_out
            )
        ctx.ffn_out = self.ffn_cache[layer_idx]

    def on_step_start(self, block_mask: torch.Tensor | None, frame: Frame):
        current_steps = frame.steps.max(-1, keepdim=True).values
        can_generate = torch.sum(
            frame.generated_tokens == self.mask_token_id, dim=-1, keepdim=True
        ).bool()
        refresh_prompt = (current_steps + 1) % self.kp == 0
        refresh_response = (current_steps + 1) % self.kr == 0
        self._prompt_length = frame.prompts.size(-1)

        assert (
            torch.unique(refresh_prompt[can_generate]).numel() <= 1
            and torch.unique(refresh_response[can_generate]).numel() <= 1
        ), "All unfinished sequences must have the same refresh schedule."

        if refresh_prompt[can_generate].numel() > 0:
            # all fisequences are
            self.refresh_prompt = refresh_prompt[can_generate][0].item()
            self.refresh_response = refresh_response[can_generate][0].item()
            self.can_generate = can_generate

class elasticCache(dCache):
    def __init__(
        self,
        model_config,
        similarity_threshold: float = 0.9,
    ):
        super().__init__(model_config)
        self.query_cache: list[torch.Tensor] = []
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.x_cache: list[torch.Tensor] = []

        self.prev_q: torch.Tensor | None = None
        self.prev_k: torch.Tensor | None = None

        self.similarity_threshold: float = similarity_threshold

        self.prev_t: torch.Tensor | None = None # list of most attended positions at previous time step assuming batch size 1 for simplicity
        self.curr_t: torch.Tensor | None = None # list of most attended positions at current time step assuming batch size 1 for simplicity
        self._decoded: torch.Tensor | None = None # decoded tokens so far, D^t in the algorithm
        self._masked: torch.Tensor | None = None  # masked tokens within the current window
        

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        with super().model_forward(x=x) as ctx:
            B, T, C = x.shape

            if self.active_q_mask is not None:
                ctx.x = x[self.active_q_mask].view(B, -1, C)

            self.l_recompute = 999

            yield ctx

            if self.active_q_mask is not None:
                assert ctx.logits is not None
                ctx.logits = torch.zeros(
                    (B, T, ctx.logits.size(-1)),
                    dtype=ctx.logits.dtype,
                    device=ctx.logits.device,
                ).masked_scatter_(self.active_q_mask.unsqueeze(-1), ctx.logits)

    @contextmanager
    def attention(
        self,
        layer_idx: int,
        input_x: torch.Tensor,
        norm_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):  
        if len(self.x_cache) <= layer_idx:
            # the first forward pass, store states as cache
            self.x_cache.append(input_x)
        elif self.active_q_mask is not None:
            self.x_cache[layer_idx][self.active_q_mask] = input_x
        else:
            self.x_cache[layer_idx] = input_x

        if layer_idx >= self.l_recompute:
            if layer_idx == self.l_recompute + 1:  # subsequent layers can just use output from previous layer
                # prepare input x for full sequence recompute
                input_x = self.x_cache[layer_idx]
                self.active_q_mask = None


        B, T, C = input_x.shape
        with super().attention(
            layer_idx, input_x, norm_fn, q_proj, k_proj, v_proj, attention_mask, position_ids
        ) as ctx:
            
            ctx.attention_mask = attention_mask
            ctx.q_position_ids, ctx.kv_position_ids = (
                AttentionContext.select_position_ids(
                    position_ids, self.active_q_mask, self.active_q_mask
                )
            )

            yield ctx
            assert (
                ctx.attn_weight is not None
            ), 'The attention weights must be outputed, make sure you\'ve set attn_implementation="eager"'
            attn_scores = ctx.attn_weight  # (B, n_heads, q_len, seq_len)
            B, n_heads, q_len, seq_len = attn_scores.shape
            
            # with self.timer.measure("thresholding_overhead"):
            if layer_idx < self.l_recompute and self.step > 0: # we only need to compute kv drift before threshold is triggered
            
                assert self.prev_t is not None
                assert self.active_q_mask is not None

                # Implementation based on https://github.com/VILA-Lab/Elastic-Cache/blob/46c0bc329f8e4ebc5fae34405b05cc6933e1a9e0/llada/model/modeling_llada.py#L824-L832

                # calculate attention score of most attended tokens from previous time step
                prev_t_mask = self.prev_t[self.active_q_mask]
                s_curr = attn_scores[:,:, prev_t_mask, :] # (num_active_q, seq_len) # TODO: fix for batch size > 1

                scale_factor = 1 / math.sqrt(self.prev_q.size(-1))

                # if model is dream, repeat kv
                if self.model_config.model_type.lower() == "dream":
                    n_rep = self.model_config.num_attention_heads // self.model_config.num_key_value_heads
                    batch, num_key_value_heads, slen, head_dim = self.prev_k.shape
                    self.prev_k = self.prev_k[:, :, None, :, :].expand(
                        batch, num_key_value_heads, n_rep, slen, head_dim
                    ).reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

                # compute cosine similarity between \mathbf{S}^{t-1,l}_{[\mathcal{T}^{t-1}]} and \mathbf{S}^{t,l}_{[\mathcal{T}^{t-1}]}
                s_prev = torch.matmul(self.prev_q[:,:,prev_t_mask,:], self.prev_k.transpose(-2, -1)) * scale_factor  # (num_active_q, seq_len)
                s_prev = torch.softmax(s_prev, dim=-1)

                cos_sim = F.cosine_similarity(
                    s_prev,
                    s_curr,
                    dim=1,
                ).mean().item() # TODO: fix for batch size > 1
        
                if cos_sim < self.similarity_threshold:
                    self.l_recompute = layer_idx + 1
            masked_attn_sum = attn_scores.sum(dim=(1,2))
            masked_attn_sum[self._masked] = 0.0  # zero out attention scores to masked positions
            most_attentive_position = masked_attn_sum.argmax(dim=-1)  # (B,) # TODO: fix for batch size > 1
            self.curr_t[torch.arange(B), most_attentive_position] = True
    
    def adjust_kv(self, ctx, layer_idx):
        if len(self.key_cache) <= layer_idx:
            # the first forward pass, store states as cache
            # with self.timer.measure("extra_caching_overhead"):
            self.query_cache.append(ctx.q)
            # with self.timer.measure("kv_update"):
            self.key_cache.append(ctx.k)
            self.value_cache.append(ctx.v)
        else:
            if self.active_q_mask is not None:
                B, num_heads, q_len, head_dim = ctx.q.shape
                # with self.timer.measure("extra_caching_overhead"):
                self.prev_q = self.query_cache[layer_idx][self.active_q_mask.unsqueeze(1).unsqueeze(-1).expand_as(self.query_cache[layer_idx])].clone().view(B, num_heads, q_len, head_dim)
                self.prev_k = self.key_cache[layer_idx].clone()
                self.query_cache[layer_idx].masked_scatter_(self.active_q_mask.unsqueeze(1).unsqueeze(-1), ctx.q)
                # with self.timer.measure("kv_update"):
                self.key_cache[layer_idx].masked_scatter_(self.active_q_mask.unsqueeze(1).unsqueeze(-1), ctx.k)
                self.value_cache[layer_idx].masked_scatter_(self.active_q_mask.unsqueeze(1).unsqueeze(-1), ctx.v)
                ctx.k = self.key_cache[layer_idx]
                ctx.v = self.value_cache[layer_idx]
            else:
                # with self.timer.measure("kv_update"):
                self.key_cache[layer_idx] = ctx.k
                self.value_cache[layer_idx] = ctx.v
                # with self.timer.measure("extra_caching_overhead"):
                self.query_cache[layer_idx] = ctx.q
        
        ctx.attention_mask = AttentionContext.convert_attention_mask(
            ctx.attention_mask,
            dtype=ctx.k.dtype,
            query_length=ctx.q.shape[2],
            key_value_length=self.value_cache[layer_idx].shape[2],
        )

    def on_inference_start(self, frame: Frame, window_or_block_size: int | None = None):
        self.step = 0
        self.x_cache = []
        self.key_cache = []
        self.value_cache = []

        self.curr_t = torch.cat([
            torch.zeros_like(frame.prompts, dtype=torch.bool), torch.zeros_like(frame.generated_tokens, dtype=torch.bool)
        ], dim=-1)
        self.full_decoded = torch.zeros_like(self.curr_t, dtype=torch.bool)

    def on_step_start(self, window_mask: torch.Tensor | None, frame: Frame):
        B, P = frame.prompts.size()
        # update self._masked via sliding window ({M^t}_β ← {M^t}[:β])
        # if model is dream, we need to retain the token before masked tokens
        if self.model_config.model_type.lower() == "dream":
            self._masked = F.pad(window_mask, (P - 1, 1), value=False)
        else:
            self._masked = F.pad(window_mask, (P, 0), value=False)
        if self.step == 0:
            # initialize self._decoded and self._masked / D^1 ← {1, . . . , P}; M^1 ← {P + 1, . . . , P + N};
            # construct prompt mask
            # seq = torch.cat([frame.prompts, frame.generated_tokens], dim=-1) < _decoded = mask where prompt tokens are 1
            self._decoded = torch.cat((torch.ones_like(frame.prompts, dtype=torch.bool), torch.zeros_like(frame.generated_tokens, dtype=torch.bool)), dim=-1)
        self.full_decoded.masked_fill_
        # Qt ← T^{t−1} ∪ M^{t−1} = T^{t-1} ∪ M^t ∪ D^t
        self.active_q_mask = self._masked | self._decoded | self.prev_t if self.prev_t is not None else None # don't apply active_q_mask at the first step
        # write self.active_q_mask to local file: i for 1, | for 0, d for self._decoded = 1, m for self._masked = 1
        
        
    def on_step_end(
        self, block_mask: torch.Tensor | None, frame: Frame, delta: FrameDelta
    ):
        # update prev_t and curr_t for next step
        self.prev_t = self.curr_t
        self.curr_t = torch.zeros_like(self.prev_t, dtype=torch.bool)

        B = frame.prompts.size(0)
        device = frame.prompts.device
        T_gen = frame.generated_tokens.size(-1)

        transfer_src_index = delta.transfer_src_index if delta.transfer_src_index is not None else delta.transfer_index

        mask = torch.zeros((B, T_gen), dtype=torch.bool, device=device)

        for i, indices in enumerate(transfer_src_index):
            mask[i, indices] = True

        if self.step == 1:
            self._decoded[:, :frame.prompts.size(-1)] = False

        self._decoded[:, -T_gen:] = mask

        self.step += 1

class entropyCache(dCache):
    def __init__(
        self,
        model_config,
        entropy_threshold: float = 0.5,
        k_recent_keep: int = 64,
    ):
        super().__init__(model_config)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

        self.entropy_threshold: float = entropy_threshold
        self.k_recent_keep: int = k_recent_keep

        self._decoded: torch.Tensor | None = None # decoded tokens from the previous step, D^t in the algorithm
        self._masked: torch.Tensor | None = None  # masked tokens within the current window

        self.skip_flag = False
        self.active_q_mask = None

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        with super().model_forward(x=x) as ctx:
            B, T, C = x.shape
            if self.step > 0: 
                if self.skip_flag:
                    self.active_q_mask = self._decoded | self._masked
                else:
                    self.active_q_mask = None

            if self.active_q_mask is not None:
                ctx.x = x[self.active_q_mask].view(B, -1, C)

            yield ctx

            if self.active_q_mask is not None :
                assert ctx.logits is not None
                ctx.logits = torch.zeros(
                    (B, T, ctx.logits.size(-1)),
                    dtype=ctx.logits.dtype,
                    device=ctx.logits.device,
                ).masked_scatter_(self.active_q_mask.unsqueeze(-1), ctx.logits)

    @contextmanager
    def attention(
        self,
        layer_idx: int,
        input_x: torch.Tensor,
        norm_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):  
        B, T, C = input_x.shape

        with super().attention(
            layer_idx, input_x, norm_fn, q_proj, k_proj, v_proj, attention_mask, position_ids
        ) as ctx:
            
            ctx.attention_mask = attention_mask
            ctx.q_position_ids, ctx.kv_position_ids = (
                AttentionContext.select_position_ids(
                    position_ids, self.active_q_mask, self.active_q_mask
                )
            )

            yield ctx
            assert (
                ctx.attn_weight is not None
            ), 'The attention weights must be outputed, make sure you\'ve set attn_implementation="eager"'

    def adjust_kv(self, ctx, layer_idx):
        if len(self.key_cache) <= layer_idx:
            # the first forward pass, store states as cache
            self.key_cache.append(ctx.k)
            self.value_cache.append(ctx.v)
        else:

            if self.active_q_mask is not None:
                self.key_cache[layer_idx][:, :, self.active_q_mask[0],:] = ctx.k
                self.value_cache[layer_idx][:, :, self.active_q_mask[0],:] = ctx.v
                ctx.k = self.key_cache[layer_idx]
                ctx.v = self.value_cache[layer_idx]
            else:
                self.key_cache[layer_idx] = ctx.k
                self.value_cache[layer_idx] = ctx.v

        ctx.attention_mask = AttentionContext.convert_attention_mask(
            ctx.attention_mask,
            dtype=ctx.k.dtype,
            query_length=ctx.q.shape[2],
            key_value_length=self.value_cache[layer_idx].shape[2]
        )

    def on_inference_start(self, frame: Frame, window_or_block_size: int | None = None):
        self.key_cache = []
        self.value_cache = []
        self.step = 0

        self.decoded_history = torch.zeros_like(frame.generated_tokens, dtype=torch.int)
        self.decoded_history.fill_(-9999)

    def on_step_start(self, window_mask: torch.Tensor | None, frame: Frame):
        B, P = frame.prompts.size()
        _, T_gen = frame.generated_tokens.size()
        # update self._masked via sliding window ({M^t}_β ← {M^t}[:β])
        # if model is dream, we need to retain the token before masked tokens
        if self.model_config.model_type.lower() == "dream":
            self._masked = F.pad(window_mask, (P - 1, 1), value=False)
        else:
            self._masked = F.pad(window_mask, (P, 0), value=False)

        if self.step == 0:
            # initialize self._decoded and self._masked / D^1 ← {1, . . . , P}; M^1 ← {P + 1, . . . , P + N};
            # construct prompt mask
            # seq = torch.cat([frame.prompts, frame.generated_tokens], dim=-1) < _decoded = mask where prompt tokens are 1
            self._decoded = torch.cat((torch.ones_like(frame.prompts, dtype=torch.bool), torch.zeros_like(frame.generated_tokens, dtype=torch.bool)), dim=-1)
        if self.step == 1:
            self._decoded[:, :frame.prompts.size(-1)] = False

    def on_step_end(
        self, block_mask: torch.Tensor | None, frame: Frame, delta: FrameDelta
    ):
        # update self._decoded

        B = frame.prompts.size(0)
        device = frame.prompts.device
        T_gen = frame.generated_tokens.size(-1)

        transfer_src_index = delta.transfer_src_index if delta.transfer_src_index is not None else delta.transfer_index

        mask = torch.zeros((B, T_gen), dtype=torch.bool, device=device)

        for i, indices in enumerate(transfer_src_index):
            mask[i, indices] = True

        if self.step == 1:
            self._decoded[:, :frame.prompts.size(-1)] = False
        
        # calculate masked token entropy
        probs = delta.probs # (B, T_gen, Vocab_size)
        # compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # (B, T_gen)
        if self.model_config.model_type.lower() == "dream":
            decoded_entropy = entropy[:, mask[self._masked[:, -(T_gen + 1):-1]]]
        else:
            decoded_entropy = entropy[:, mask[self._masked[:, -T_gen:]]]

        # update decoded history (value = self.step for mask tokens)
        self.decoded_history[mask] = self.step

        # keep recent k tokens
        # topk on decoded history to obtain step threshold
        thresh = torch.topk(self.decoded_history, self.k_recent_keep, largest=True).values.min().item()
        recent_decoded_mask = self.decoded_history >= thresh
        if self.model_config.model_type.lower() == "dream":
            self._decoded[:, -(T_gen + 1):-1] = mask | recent_decoded_mask
            self._decoded[:, -T_gen:] |= mask | recent_decoded_mask
        else:
            self._decoded[:, -T_gen:] = mask | recent_decoded_mask


        if (decoded_entropy > self.entropy_threshold).any():
            self.skip_flag = False
        else:
            self.skip_flag = True

        self.step += 1
