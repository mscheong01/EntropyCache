import os
import torch
import torch.distributions as dists

from typing import Type, Tuple

from src.cache import dCache
from src.frame import INVALID_TOKEN_ID, Frame, FrameDelta, DecodeRecord, Intermediate
from src.generation.utils import prepare_logits_for_generation
from src.generation.vanilla import sample_tokens, confidence_unmasking, get_num_transfer_tokens
from src.dream.generation_utils import top_k_logits, top_p_logits
from src.utils import register, prefill_register, generate_register
from src.third_party import get_token_freq

@torch.no_grad()
def generate_step(
    model,
    frame: Frame,
    window_mask: torch.Tensor,
    num_transfer_tokens: torch.Tensor | int,
    attention_mask: torch.Tensor | None = None,
    past_key_values: dCache | None = None,
    alg: str = "maskgit_plus",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: float | None = None,
    debias: bool = True,
    mask_token_id: int = None,  # type: ignore
    threshold: float | None = None,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> FrameDelta | None:
    """
    Generate a single step from a given frame. If all mask tokens have been filled, return None.
    For `num_transfer_tokens`, a generation step selects a fixed number of tokens to transfer.
    For `threshold`, a generation step selects all tokens whose confidence is above the threshold.

    Args:
        model: Mask predictor.
        frame: The current frame containing the prompt and generated tokens.
        num_transfer_tokens: The number of tokens to transfer at this step. It can be a tensor with shape (batch_size,) or
            a single integer that represents the number of tokens to transfer for all batches.
        temperature: Categorical distribution sampling temperature.
        threshold: A threshold for remasking. If provided, all tokens whose confidence is above this threshold will be kept.
        factor: factor-based parallel decoding factor, see https://arxiv.org/pdf/2505.22618.
        output_hidden_states: Whether to return the hidden states of all decoded tokens.
        output_probs: Whether to return the probabilities of the decoded tokens.
    """
    if threshold is not None and factor is not None:
        raise ValueError(
            "Only one of `threshold` or `factor` can be specified, not both."
        )

    frame = frame.as_batch()
    batch_size, prompt_length = frame.prompts.shape

    if isinstance(num_transfer_tokens, torch.Tensor):
        if num_transfer_tokens.numel() != batch_size or num_transfer_tokens.dim() != 1:
            raise ValueError(
                f"`num_transfer_tokens` must be a tensor of shape ({batch_size},) or a single integer, "
                f"but got shape of {num_transfer_tokens.shape}."
            )
    else:
        num_transfer_tokens = torch.full(
            (batch_size,),
            num_transfer_tokens,
            device=model.device,
            dtype=torch.long,
        )

    transfer_index_mask = frame.generated_tokens == mask_token_id
    can_generate = (window_mask & transfer_index_mask).any(dim=-1)
    if not torch.any(can_generate):
        return None
    # skip sequence that doesn't require to generate
    can_generate &= num_transfer_tokens > 0

    x = torch.cat([frame.prompts, frame.generated_tokens], dim=-1)[can_generate]
    attention_mask = (
        attention_mask[can_generate] if attention_mask is not None else None
    )
    num_transfer_tokens = (
        num_transfer_tokens[can_generate]
        if isinstance(num_transfer_tokens, torch.Tensor)
        else num_transfer_tokens
    )
    outputs = model(
        x,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        past_key_values=past_key_values,
        use_cache=past_key_values is not None,
    )
    # main difference (1) with LLaDA, see https://github.com/DreamLM/Dream/issues/31
    logits = prepare_logits_for_generation(model, outputs.logits)
    if past_key_values is not None and past_key_values.active_q_mask is not None:
        if model.config.model_type.lower() == "dream":
            valid_mask = past_key_values.active_q_mask[:, prompt_length - 1 : -1]
        else:
            valid_mask = past_key_values.active_q_mask[:, prompt_length:]
        transfer_index_mask &= valid_mask
    logits = logits[:, prompt_length:][transfer_index_mask].view(
        batch_size, -1, logits.size(-1)
    )

    hidden_states = (
        tuple((i, hs) for i, hs in enumerate(outputs.hidden_states))
        if output_hidden_states
        else None
    )

    if debias:
        global _token_freq
        _token_freq = get_token_freq(model, device=logits.device, dtype=logits.dtype)

    # sampling tokens for all generated positions
    if alg == "maskgit_plus":
        # same as LLaDA
        confidence, x0, p = sample_tokens(
            logits, temperature=temperature, top_p=top_p, top_k=top_k, debias=debias
        )
    elif alg == "topk_margin":
        # main difference (2) with LLaDA, the confidence may not be the token probabilities
        confidence, x0, p = sample_tokens(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            margin_confidence=True,
            debias=debias,
        )
    elif alg == "entropy":
        confidence, x0, p = sample_tokens(
            logits,
            temperature,
            top_p=top_p,
            top_k=top_k,
            neg_entropy=True,
            debias=debias,
        )
    else:
        raise RuntimeError(f"Unknown alg: {alg}")

    confidence = torch.full_like(
        frame.generated_tokens, -float("inf"), dtype=confidence.dtype
    ).masked_scatter_(transfer_index_mask, confidence)
    # select tokens to transfer based on probs and mask
    transfer_index = confidence_unmasking(
        scores=confidence,
        transfer_index_mask=(transfer_index_mask & window_mask)[can_generate],
        num_transfer_tokens=num_transfer_tokens,
        threshold=threshold,
        factor=factor,
    )
    transfer_index_iter = iter(transfer_index)
    transfer_index = tuple(
        (
            next(transfer_index_iter)
            if is_not_finished
            else torch.tensor([], dtype=torch.long, device=x0.device)
        )
        for is_not_finished in can_generate
    )

    delta = FrameDelta(
        transfer_index=transfer_index,
        decoded_tokens=torch.full_like(
            frame.generated_tokens, INVALID_TOKEN_ID
        ).masked_scatter_(transfer_index_mask, x0),
        confidence=confidence,
        probs=p[:, window_mask[transfer_index_mask], :] if output_probs else None,
        intermediate=Intermediate(
            hidden_states=hidden_states if hidden_states is not None else tuple()
        ),
    )
    return delta.unbatch() if not frame.is_batched else delta

def calculate_new_window(
    model, 
    frame:Frame, 
    window_size:int,
    mask_token_id:int = None,
    ) -> torch.Tensor:
    masked_positions = frame.generated_tokens == mask_token_id
    # select first window_size masked positions
    masked_indices = [
        torch.nonzero(masked_positions[i], as_tuple=False)[:window_size, 0]
        for i in range(masked_positions.size(0))
    ]
    lengths = torch.tensor([idx.numel() for idx in masked_indices], device=frame.prompts.device)

    row_indices = torch.repeat_interleave(
        torch.arange(frame.prompts.size(0), device=frame.prompts.device), lengths
    )
    col_indices = torch.cat(masked_indices)
    new_masked = torch.zeros_like(masked_positions, dtype=torch.bool)
    new_masked[row_indices.unsqueeze(-1), col_indices] = True

    return new_masked

@prefill_register.gen_strategy("sliding_window")
def sliding_window_prefill(
    model,
    input_ids: torch.Tensor,
    alg: str = "maskgit_plus",
    steps: int = 128,
    window_size: int = 32,
    gen_length: int = 128,
    temperature: float = 0.0,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    debias: bool = False,
    threshold: float | None = None,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache_cls: Type[dCache] | None = None,
    tokenizer=None,
) -> Tuple[dCache|None, Frame, FrameDelta, torch.Tensor|None]: # cache, initial_frame, initial_delta, attention_mask
    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID", None) is None:
        raise ValueError(
            "mask_token_id must be provided either as an argument or an environment variable."
        )
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))  # type: ignore

    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    if pad_token_id is not None:
        attention_mask = torch.cat(
            [
                (input_ids != pad_token_id).long().to(device=model.device),
                torch.ones(input_ids.size(0), gen_length, device=model.device).long(),
            ],
            dim=-1,
        )
    else:
        attention_mask = None

    cache = cache_cls(model.config) if cache_cls is not None else None
    frame = initial_frame

    if cache is not None:
        cache.on_inference_start(frame, window_or_block_size=window_size)

    full_mask = torch.ones(
        (input_ids.size(0), gen_length),
        dtype=torch.bool,
        device=model.device,
    )
    num_transfer_tokens = get_num_transfer_tokens(full_mask, steps)

    new_window_mask = calculate_new_window(model, frame, window_size=window_size, mask_token_id=mask_token_id)
    if cache is not None:
        cache.on_step_start(new_window_mask, frame)
    delta = generate_step(
        model=model,
        frame=frame,
        num_transfer_tokens=num_transfer_tokens[:, 0],
        attention_mask=attention_mask,
        past_key_values=cache,
        window_mask=new_window_mask,
        alg=alg,
        temperature=temperature,
        mask_token_id=mask_token_id,
        top_p=top_p,
        top_k=top_k,
        debias=debias,
        threshold=threshold,
        factor=factor,
        output_hidden_states=output_hidden_states,
        output_probs=output_probs,
    )
    if cache is not None:
        cache.on_step_end(new_window_mask, initial_frame, delta)

    return cache, initial_frame, delta.to("cpu"), attention_mask

@generate_register.gen_strategy("sliding_window")
def sliding_window_generate_(
    model,
    cache: dCache | None,
    initial_frame: Frame,
    initial_delta: FrameDelta,
    attention_mask: torch.Tensor | None = None,
    alg: str = "maskgit_plus",
    steps: int = 128,
    window_size: int = 32,
    gen_length: int = 128,
    temperature: float = 0.0,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    debias: bool = False,
    threshold: float | None = None,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache_cls: Type[dCache] | None = None,
    tokenizer=None,
) -> DecodeRecord:

    frame = initial_frame.apply_delta(initial_delta).to(device=model.device, dtype=model.dtype)
    deltas = [initial_delta.to("cpu")]

    full_mask = torch.ones(
        (frame.generated_tokens.size(0), gen_length),
        dtype=torch.bool,
        device=model.device,
    )
    num_transfer_tokens = get_num_transfer_tokens(full_mask, steps)

    i = 1
    while i < steps:
        
        start_frame = frame.clone()
        new_window_mask = calculate_new_window(model, frame, window_size=window_size, mask_token_id=mask_token_id)
        if cache is not None:
            cache.on_step_start(new_window_mask, frame)
        delta = generate_step(
            model=model,
            frame=frame,
            num_transfer_tokens=num_transfer_tokens[:, i],
            attention_mask=attention_mask,
            past_key_values=cache,
            window_mask=new_window_mask,
            alg=alg,
            temperature=temperature,
            mask_token_id=mask_token_id,
            top_p=top_p,
            top_k=top_k,
            debias=debias,
            threshold=threshold,
            factor=factor,
            output_hidden_states=output_hidden_states,
            output_probs=output_probs,
        )
        if delta is None:
            # if no more mask tokens are left, break the loop

            break
        if cache is not None:
            cache.on_step_end(new_window_mask, start_frame, delta)

        frame = frame.apply_delta(delta)

        deltas.append(delta.to("cpu"))

        # # print intermediate generation result
        # print(
        #     f"Step {i+1}/{steps}, generated tokens: {tokenizer.batch_decode(frame.generated_tokens, skip_special_tokens=True)}"
        # )

        i += 1
    if cache is not None:
        cache.on_inference_end(frame)
    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=window_size,
        recompute_ratio=cache.get_recompute_ratio() if cache is not None else 0.0,
    )




@register.gen_strategy("sliding_window")
def sliding_window_generate(
    model,
    input_ids: torch.Tensor,
    alg: str = "maskgit_plus",
    steps: int = 128,
    window_size: int = 32,
    gen_length: int = 128,
    temperature: float = 0.0,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    debias: bool = False,
    threshold: float | None = None,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache_cls: Type[dCache] | None = None,
    tokenizer=None,
) -> DecodeRecord:
    """
    Vanilla generation for diffusion large language models.
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L).
        steps: Sampling steps, less than or equal to gen_length.
        window_size: The size of the sliding window.
        gen_length: Generated answer length.
        temperature: Categorical distribution sampling temperature.
        mask_token_id: The token id of [MASK]. It can be `None` if "MASK_TOKEN_ID" is specified in the environment variables.
        top_k: The number of highest probability tokens to keep for one generation step.
        top_p: The cumulative probability threshold for nucleus sampling.
        threshold: A threshold for remasking. If provided, all tokens whose confidence is above this threshold will be kept.
        factor: factor-based parallel decoding factor, see https://arxiv.org/pdf/2505.22618.
        output_hidden_states: Whether to return the hidden states of all decoded tokens from layers.
        output_probs: Whether to return the probs of all tokens.
    """

    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID", None) is None:
        raise ValueError(
            "mask_token_id must be provided either as an argument or an environment variable."
        )
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))  # type: ignore

    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)
    if pad_token_id is not None:
        attention_mask = torch.cat(
            [
                (input_ids != pad_token_id).long().to(device=model.device),
                torch.ones(input_ids.size(0), gen_length, device=model.device).long(),
            ],
            dim=-1,
        )
    else:
        attention_mask = None

    cache = cache_cls(model.config) if cache_cls is not None else None
    frame = initial_frame

    if cache is not None:
        cache.on_inference_start(frame, window_or_block_size=window_size)

    deltas = []

    full_mask = torch.ones(
        (input_ids.size(0), gen_length),
        dtype=torch.bool,
        device=model.device,
    )
    num_transfer_tokens = get_num_transfer_tokens(full_mask, steps)

    # print gen length
    # print(f"Generation length: {gen_length}, steps: {steps}, window size: {window_size}")

    i = 0
    while i < steps:

        start_frame = frame.clone()
        new_window_mask = calculate_new_window(model, frame, window_size=window_size, mask_token_id=mask_token_id)
        if cache is not None:
            cache.on_step_start(new_window_mask, frame)
        delta = generate_step(
            model=model,
            frame=frame,
            num_transfer_tokens=num_transfer_tokens[:, i],
            attention_mask=attention_mask,
            past_key_values=cache,
            window_mask=new_window_mask,
            alg=alg,
            temperature=temperature,
            mask_token_id=mask_token_id,
            top_p=top_p,
            top_k=top_k,
            debias=debias,
            threshold=threshold,
            factor=factor,
            output_hidden_states=output_hidden_states,
            output_probs=output_probs,
        )
        if delta is None:
            # if no more mask tokens are left, break the loop
            break
        if cache is not None:
            cache.on_step_end(new_window_mask, start_frame, delta)

        frame = frame.apply_delta(delta)

        deltas.append(delta.to("cpu"))

        # print intermediate generation result
        # print(
        #     f"Step {i+1}/{steps}, generated tokens: {tokenizer.batch_decode(frame.generated_tokens, skip_special_tokens=True)}"
        # )

        i += 1
    
    if cache is not None:
        cache.on_inference_end(frame)

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=window_size,
        recompute_ratio=cache.get_recompute_ratio() if cache is not None else 0.0,
    )
