"""
Training-aligned prefix helpers for the Qwen3 proxy-correlation study.

These utilities mirror the prefix construction and mean-token-entropy
computation used in the training loop
(`verl/verl/trainer/ppo/ray_trainer.py::_update_prefix_memory`), so
that the "MTE-of-prefix" random variable measured in the analysis
scripts is the same one training's guided-resampling code selects on.

Public API:

    strip_thinking(response_ids, tokenizer) -> (post_ids, start_pos, had_close)
        Slice off the <think>...</think> prelude in token-ID space.

    build_newline_token_ids(tokenizer) -> set[int]
        Cache of token IDs that decode to newlines.  Used as the split
        alphabet for prefix boundaries.

    find_prefix_boundaries(response_ids, newline_ids, min_gap=32, cap=15)
        Newline-token positions in `response_ids`, merged so adjacent
        boundaries are >= `min_gap` tokens apart, and evenly
        downsampled to at most `cap` entries.  Matches
        ray_trainer.py:806-836.

    mean_token_entropy_over_response(
        model, tokenizer, formatted_prompt, response_ids,
        max_context_tokens=None,
    ) -> np.ndarray
        One HF forward pass; returns per-token entropy over the
        response tokens (length = len(response_ids)).  NO 2048
        truncation by default -- uses the tokenizer/model's own limit.

Design notes:
- Everything works in token-ID space to stay isomorphic with training.
  Decoding to text happens only at the boundary (for embedding / for
  vLLM continuation prompts).
- Fallbacks are silent-but-flagged: if </think> is not found, we
  return the whole response and set had_close=False so the caller can
  choose to skip prefix mining for that rollout.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Newline / </think> token-ID discovery (cached on the tokenizer object).
# --------------------------------------------------------------------------

_NEWLINE_ATTR = "_qwen3_prefix_newline_ids"
_CLOSE_THINK_ATTR = "_qwen3_prefix_close_think_ids"


def build_newline_token_ids(tokenizer) -> set[int]:
    """Set of token IDs that decode to a bare newline string.

    Same discovery logic as ray_trainer.py:788-804.  Cached on the
    tokenizer instance so repeated calls are O(1).
    """
    cached = getattr(tokenizer, _NEWLINE_ATTR, None)
    if cached is not None:
        return cached

    nl_ids: set[int] = set()
    for text in ("\n", "\n\n"):
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            encoded = []
        nl_ids.update(encoded)

    # Scan the first 1000 vocab entries for single-newline tokens; this
    # catches merged tokens that don't come out of encode("\n").  Same
    # cap as the training code.
    vocab_size = getattr(tokenizer, "vocab_size", 0)
    for tid in range(min(1000, vocab_size)):
        try:
            decoded = tokenizer.decode([tid])
        except Exception:
            continue
        if decoded == "\n":
            nl_ids.add(tid)

    setattr(tokenizer, _NEWLINE_ATTR, nl_ids)
    return nl_ids


def _close_think_token_ids(tokenizer) -> set[int]:
    """Token IDs that emit '</think>'.  Cached on the tokenizer."""
    cached = getattr(tokenizer, _CLOSE_THINK_ATTR, None)
    if cached is not None:
        return cached

    ids: set[int] = set()
    # Qwen3 typically has </think> as a single added special token.
    for text in ("</think>", " </think>", "</think>\n"):
        try:
            enc = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            enc = []
        # If the exact string maps to a single token, use it.
        if len(enc) == 1:
            ids.add(enc[0])

    # Fallback: sweep added_tokens for a </think>-like entry.
    added = getattr(tokenizer, "added_tokens_encoder", {}) or {}
    for tok_str, tid in added.items():
        s = str(tok_str).strip()
        if s == "</think>":
            ids.add(int(tid))

    setattr(tokenizer, _CLOSE_THINK_ATTR, ids)
    return ids


# --------------------------------------------------------------------------
# <think>...</think> stripping.
# --------------------------------------------------------------------------

def strip_thinking(
    response_ids: list[int] | torch.Tensor,
    tokenizer,
) -> tuple[list[int], int, bool]:
    """Return (post_think_ids, start_pos, had_close).

    start_pos is the index in `response_ids` immediately after
    </think> (i.e. the offset such that response_ids[start_pos:] ==
    post_think_ids).  had_close=False signals that no </think> token
    was found; in that case we return the whole response unchanged so
    the caller can choose whether to skip the rollout.
    """
    if isinstance(response_ids, torch.Tensor):
        id_list = response_ids.tolist()
    else:
        id_list = list(response_ids)

    close_ids = _close_think_token_ids(tokenizer)
    if not close_ids:
        # Tokenizer without a </think> special token -- fall back to
        # substring search in the decoded text.  Rare on Qwen3.
        try:
            decoded = tokenizer.decode(id_list, skip_special_tokens=False)
        except Exception:
            return id_list, 0, False
        marker = "</think>"
        pos = decoded.find(marker)
        if pos < 0:
            return id_list, 0, False
        # Approximate the token offset by re-encoding the prefix.  This
        # is a fallback, not the hot path.
        head_text = decoded[: pos + len(marker)]
        try:
            head_ids = tokenizer.encode(head_text, add_special_tokens=False)
        except Exception:
            return id_list, 0, False
        start = min(len(head_ids), len(id_list))
        return id_list[start:], start, True

    # Fast path: find the first close-think token id in the response.
    for i, tid in enumerate(id_list):
        if tid in close_ids:
            start = i + 1
            return id_list[start:], start, True
    return id_list, 0, False


# --------------------------------------------------------------------------
# Prefix boundary extraction (port of ray_trainer.py:806-836).
# --------------------------------------------------------------------------

def find_prefix_boundaries(
    response_ids: list[int] | torch.Tensor,
    newline_ids: set[int],
    min_gap: int = 32,
    cap: int = 15,
) -> list[int]:
    """Prefix cut points, in the SAME index space as `response_ids`.

    Each returned value `b` means "prefix = response_ids[:b]".  Ports
    the training-side logic exactly:

      * scan for newline-bearing tokens, boundary = pos + 1
      * greedily drop boundaries closer than `min_gap` to the last kept
      * drop the trailing boundary (which would give the full response)
      * evenly downsample to `cap` boundaries if there are more.
    """
    if isinstance(response_ids, torch.Tensor):
        id_list = response_ids.tolist()
    else:
        id_list = list(response_ids)

    split_positions: list[int] = []
    for pos, tid in enumerate(id_list):
        if tid in newline_ids:
            split_positions.append(pos + 1)

    if not split_positions:
        return []

    merged = [split_positions[0]]
    for pos in split_positions[1:]:
        if pos - merged[-1] >= min_gap:
            merged.append(pos)
    split_positions = merged

    if len(split_positions) > 1:
        boundaries = split_positions[:-1]
    else:
        boundaries = split_positions

    if len(boundaries) > cap:
        step = len(boundaries) / cap
        boundaries = [boundaries[int(i * step)] for i in range(cap)]

    return boundaries


# --------------------------------------------------------------------------
# Mean token entropy: one forward pass per rollout, no 2048 truncation.
# --------------------------------------------------------------------------

@torch.no_grad()
def mean_token_entropy_over_response(
    model,
    tokenizer,
    formatted_prompt: str,
    response_ids: list[int] | torch.Tensor,
    max_context_tokens: Optional[int] = None,
) -> np.ndarray:
    """Return per-token entropy for `response_ids` under `model`.

    One HF forward pass over `formatted_prompt` concatenated with
    `response_ids`.  Result length = len(response_ids); position i is
    the entropy of the distribution the model puts at the step where
    it predicts response_ids[i].

    This is the analysis analogue of `entropy_mat[i, :valid_len]` in
    ray_trainer.py -- same random variable, same aggregation.

    No `max_length=2048` truncation: we defer to the model's own
    context window via `max_context_tokens` (default: use the
    tokenizer's model_max_length, or the model's config, or 32768).
    """
    if isinstance(response_ids, torch.Tensor):
        resp_list = response_ids.tolist()
    else:
        resp_list = list(response_ids)

    if len(resp_list) == 0:
        return np.zeros(0, dtype=np.float32)

    # Tokenize the prompt in isolation to get its length in token IDs.
    prompt_enc = tokenizer(formatted_prompt, return_tensors="pt",
                           add_special_tokens=False)
    prompt_ids = prompt_enc.input_ids[0].tolist()
    prompt_len = len(prompt_ids)

    # Assemble the full ID tensor.
    full_ids = prompt_ids + resp_list

    # Determine an honest context cap.
    if max_context_tokens is None:
        cap = getattr(tokenizer, "model_max_length", None)
        if cap is None or cap > 10 ** 8:
            cap = getattr(getattr(model, "config", None),
                          "max_position_embeddings", 32768)
    else:
        cap = int(max_context_tokens)

    # If the concatenation blows past the cap, LEFT-truncate the
    # prompt (keep the response tail intact -- that's what we're
    # measuring).  If the response alone exceeds the cap, keep the
    # tail of the response and align entropy indexing accordingly.
    if len(full_ids) > cap:
        overflow = len(full_ids) - cap
        if overflow < prompt_len:
            full_ids = full_ids[overflow:]
            prompt_len -= overflow
        else:
            # Drop the whole prompt and start with a suffix of the response.
            drop_from_resp = overflow - prompt_len
            prompt_len = 0
            full_ids = resp_list[drop_from_resp:]
            resp_list = full_ids[:]

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)

    outputs = model(input_ids)
    # Logits at position t predict token at position t+1, so response
    # entropies live at logits[prompt_len-1 : -1] (inclusive of the
    # position just before the first response token, up to the last
    # response token).  When prompt_len == 0 we start at position 0.
    start = max(0, prompt_len - 1)
    end = start + len(resp_list)
    logits = outputs.logits[0, start:end, :]

    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)

    # Should be length == len(resp_list); if a boundary case shortens
    # it we pad with zeros to keep index alignment for the caller.
    ent_np = entropy.detach().cpu().numpy().astype(np.float32)
    if ent_np.shape[0] < len(resp_list):
        pad = np.zeros(len(resp_list) - ent_np.shape[0], dtype=np.float32)
        ent_np = np.concatenate([ent_np, pad], axis=0)
    return ent_np


# --------------------------------------------------------------------------
# Convenience: chat-template formatting reused by the wrappers.
# --------------------------------------------------------------------------

def format_prompt_qwen3(tokenizer, problem_text: str) -> str:
    """Same chat-template shape as monte_carlo_experiment.format_prompt.

    Kept here so downstream code can import a single canonical name
    without pulling in the whole legacy module.  Preserves the system
    prompt for parity with the Qwen2.5 baseline; drop or override
    upstream if you want to test a system-prompt-free run.
    """
    messages = [
        {"role": "system",
         "content": "Please reason step by step, and put your final answer "
                    "within \\boxed{}."},
        {"role": "user", "content": problem_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
