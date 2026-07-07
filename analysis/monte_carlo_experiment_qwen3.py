#!/usr/bin/env python3
"""
Qwen3-1.7B proxy-correlation study (rebuilt).

This is a wrapper around analysis/monte_carlo_experiment.py that:

  1. Splits prefixes on newline TOKEN IDs (not `process_thoughts`
     text), matching the training pipeline
     `verl/verl/trainer/ppo/ray_trainer.py::_update_prefix_memory`.
  2. Computes mean-token-entropy of each prefix from ONE HF forward
     pass over the rollout, with NO max_length=2048 truncation.  This
     is the same random variable training's guided-resampling selects
     on.
  3. Strips <think>...</think> from each rollout before boundary
     detection when --strip-thinking is on (default).  Fixes the
     Qwen3 answer-leak that inflated GT-injected metrics.

All existing GT-injected metrics are still computed, but they now see
the post-think prefix text and use the full model context window, so
the leaderboard is apples-to-apples with the Qwen2.5-1.5B baseline.

Cache is versioned; the on-disk .pkl written by an earlier version is
refused (delete `qwen3_outputs/2b_prefix_metrics_data.pkl` and rerun).

Usage:
    cd analysis/
    CUDA_VISIBLE_DEVICES=0,1,2,3 python monte_carlo_experiment_qwen3.py \\
        --n_problems 60
    CUDA_VISIBLE_DEVICES=0,1,2,3 python monte_carlo_experiment_qwen3.py \\
        --n_problems 5 --no-strip-thinking   # legacy Qwen2.5 semantics
"""

import argparse
import json
import os
import pickle
import random
import sys
import traceback

import numpy as np

# CUDA_VISIBLE_DEVICES must be set before any cuda init.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Disable vLLM's torch.compile path.
os.environ.setdefault("VLLM_USE_V1", "0")

# Make legacy modules importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Install shims for `simplified_evaluator` and `openrlhf`.
import _qwen3_shims  # noqa: F401
import _qwen3_robust as robust  # noqa: F401, E402
import _qwen3_prefix as qp  # noqa: E402

import torch  # noqa: E402
import monte_carlo_experiment as mc_orig  # noqa: E402
from vllm import LLM as _OrigLLM  # noqa: E402


QWEN3_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_outputs")
os.makedirs(QWEN3_OUT_DIR, exist_ok=True)


# Bump this when the on-disk prefix-metrics schema changes.  We refuse
# to load a cached .pkl written under an older schema, to prevent
# accidentally mixing pre-fix and post-fix numbers.
SCHEMA_VERSION = 2
SCHEMA_MARKER_FILE = os.path.join(QWEN3_OUT_DIR, "2b_prefix_metrics_data.schema")


# Qwen3-1.7B has 16 attention heads.  vLLM requires TP to divide that
# evenly, so valid TP sizes are 1, 2, 4, 8, 16.
def _pick_tp(n_gpus: int) -> int:
    for tp in (n_gpus, 4, 2, 1):
        if tp <= n_gpus and 16 % tp == 0:
            return tp
    return 1


_n_gpus = max(1, torch.cuda.device_count())
TP_SIZE = _pick_tp(_n_gpus)


# ----- Patch vLLM to use all visible GPUs (with valid TP size) -----
class _PatchedLLM(_OrigLLM):
    def __init__(self, *args, **kwargs):
        kwargs["tensor_parallel_size"] = TP_SIZE
        kwargs.setdefault("enforce_eager", True)
        kwargs.setdefault("gpu_memory_utilization", 0.6)
        # Raise vLLM's max_model_len well above the legacy 2048 so
        # Qwen3 thinking traces aren't silently truncated at gen time.
        kwargs.setdefault("max_model_len", 16384)
        super().__init__(*args, **kwargs)


mc_orig.LLM = _PatchedLLM


# ----- Patch HF model loader to use cuda:0 only -----
_orig_load_model = mc_orig.load_model


def _patched_load_model(model_name):
    print(f"[qwen3] loading HF metric model {model_name} on cuda:0 only")
    tokenizer = mc_orig.AutoTokenizer.from_pretrained(model_name)
    model = mc_orig.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


mc_orig.load_model = _patched_load_model


# ----- Mixed-pool problem loader (unchanged from previous version) -----
# Qwen3-1.7B saturates on MATH-500 levels 3-4, and even level-5 has
# ceiling-ish behavior.  For a non-trivial proxy correlation study we
# pull from MATH-500 (level 5) + AMC23 + AIME24 + AIME25 and uniformly
# sample n_problems across them.  Falls back to MATH-500-only if HF
# sources are unavailable.

def _safe_hf_load(name, **kw):
    from datasets import load_dataset
    try:
        return load_dataset(name, **kw)
    except Exception as e:
        print(f"[mixed] HF load failed for {name}: {e}")
        return None


def _build_mixed_pool(math500_levels):
    from datasets import load_dataset

    pool = []

    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        n = 0
        for x in ds:
            if x.get("level") in math500_levels:
                pool.append({"problem": x["problem"],
                             "answer": str(x["answer"]),
                             "level": f"MATH500-L{x.get('level')}",
                             "source": "MATH-500"})
                n += 1
        print(f"[mixed] MATH-500 levels {math500_levels}: {n} problems")
    except Exception as e:
        print(f"[mixed] MATH-500 load failed: {e}")

    for name in ("AI-MO/aimo-validation-amc", "math-ai/amc23"):
        ds = _safe_hf_load(name, split="train")
        if ds is None:
            ds = _safe_hf_load(name, split="test")
        if ds is None:
            continue
        n = 0
        for x in ds:
            prob = x.get("problem") or x.get("Problem") or x.get("question")
            ans = x.get("answer") or x.get("Answer") or x.get("final_answer")
            if prob is None or ans is None:
                continue
            pool.append({"problem": str(prob), "answer": str(ans),
                         "level": "AMC", "source": name})
            n += 1
        print(f"[mixed] {name}: {n} problems")
        break

    for name in ("Maxwell-Jia/AIME_2024", "AI-MO/aimo-validation-aime"):
        ds = _safe_hf_load(name, split="train")
        if ds is None:
            continue
        n = 0
        for x in ds:
            prob = x.get("Problem") or x.get("problem") or x.get("question")
            ans = x.get("Answer") or x.get("answer") or x.get("final_answer")
            if prob is None or ans is None:
                continue
            pool.append({"problem": str(prob), "answer": str(ans),
                         "level": "AIME24", "source": name})
            n += 1
        print(f"[mixed] {name} (AIME24): {n} problems")
        break

    for name in ("opencompass/AIME2025", "yentinglin/aime_2025"):
        ds = _safe_hf_load(name, split="train")
        if ds is None:
            ds = _safe_hf_load(name, split="test")
        if ds is None:
            continue
        n = 0
        for x in ds:
            prob = x.get("problem") or x.get("Problem") or x.get("question")
            ans = x.get("answer") or x.get("Answer") or x.get("final_answer")
            if prob is None or ans is None:
                continue
            pool.append({"problem": str(prob), "answer": str(ans),
                         "level": "AIME25", "source": name})
            n += 1
        print(f"[mixed] {name} (AIME25): {n} problems")
        break

    print(f"[mixed] total combined pool: {len(pool)} problems")
    return pool


def _patched_load_math_problems(target_levels, n=None):
    if not _CLI_OVERRIDES.get("mixed_pool", True):
        return _orig_load_math_problems(target_levels, n=n)

    pool = _build_mixed_pool(target_levels)
    if not pool:
        print("[mixed] empty pool; falling back to MATH-500 only")
        return _orig_load_math_problems(target_levels, n=n)

    random.shuffle(pool)
    if n is not None:
        pool = pool[:n]
    print(f"[mixed] using {len(pool)} problems "
          f"(sources: "
          f"{ {p['source'] for p in pool} })")
    return pool


_orig_load_math_problems = mc_orig.load_math_problems
mc_orig.load_math_problems = _patched_load_math_problems


# ----- Patch Config so model name + save dir + TP size are correct -----
_OrigConfig = mc_orig.Config

# Overrides set from CLI args in main(); kept at module scope so
# _patched_Config (called inside legacy code) can read them.
_CLI_OVERRIDES = {}


def _patched_Config(*args, **kwargs):
    c = _OrigConfig(*args, **kwargs)
    c.model_name = "Qwen/Qwen3-1.7B"
    c.save_dir = QWEN3_OUT_DIR
    c.vllm_tensor_parallel_size = TP_SIZE
    if "n_problems" in _CLI_OVERRIDES:
        c.n_problems = _CLI_OVERRIDES["n_problems"]
    if "target_levels" in _CLI_OVERRIDES:
        c.target_levels = list(_CLI_OVERRIDES["target_levels"])
    return c


# ============================================================
# Training-aligned prefix construction and MTE.
# ============================================================
#
# Replaces monte_carlo_experiment._build_prefix_requests
# (analysis/monte_carlo_experiment.py:335-388).
#
# Per-rollout entropy is cached and reused across all prefixes of that
# rollout, so MTE-of-prefix is exactly `entropy_row[:prefix_len].mean()`
# -- mirroring ray_trainer.py:838-844.
# ============================================================

# Cache the per-rollout entropy row so downstream metric computation
# can retrieve it in O(1) via the trajectory identity.  Populated
# during _patched_build_prefix_requests and consumed inside
# _patched_compute_all_metrics_for_prefix.
_ROLLOUT_ENTROPY_CACHE: dict[int, np.ndarray] = {}

# Also cache per-rollout tokenized response IDs so the metric loop can
# recover the token-space prefix length without re-tokenizing.
_ROLLOUT_TOKENS_CACHE: dict[int, list[int]] = {}


def _cache_key(problem_id: int, resp_index: int) -> int:
    """Stable per-rollout cache key used across the pipeline."""
    return hash((int(problem_id), int(resp_index)))


def _patched_build_prefix_requests(collected_data, tokenizer, config):
    """Newline-token-ID splitter with optional </think> stripping.

    Emits prefix requests in the same schema as the legacy
    _build_prefix_requests, plus these extra fields consumed by the
    patched metric loop:

      rollout_key: int
        Key into _ROLLOUT_ENTROPY_CACHE / _ROLLOUT_TOKENS_CACHE.
      prefix_len_tokens: int
        Length of the prefix in response-token-ID space (used to slice
        the cached entropy row).
      post_think_start: int
        Token offset of the </think> boundary in the response tokens
        (0 if not stripped or not found).
      had_close_think: bool
        Whether </think> was actually found for this rollout.
    """
    strip_thinking = bool(_CLI_OVERRIDES.get("strip_thinking", True))
    newline_ids = qp.build_newline_token_ids(tokenizer)

    all_requests: list[dict] = []
    prompts_for_generation: list[str] = []

    n_no_close = 0
    n_no_boundary = 0

    for item in collected_data:
        prompt = item["problem"]
        gt = item["gt"]
        formatted_prompt = mc_orig.format_prompt(tokenizer, prompt)

        correct_responses = [p for p in item["parsed_responses"] if p["is_correct"]]
        incorrect_responses = [p for p in item["parsed_responses"] if not p["is_correct"]]

        n_each = min(len(correct_responses), len(incorrect_responses), 4)
        random.shuffle(correct_responses)
        random.shuffle(incorrect_responses)
        selected = correct_responses[:n_each] + incorrect_responses[:n_each]

        for resp_index, resp_data in enumerate(selected):
            full_response = resp_data["full_response"]
            resp_ids = tokenizer.encode(full_response, add_special_tokens=False)
            if len(resp_ids) < 2:
                continue

            if strip_thinking:
                post_ids, start_pos, had_close = qp.strip_thinking(resp_ids, tokenizer)
                if not had_close:
                    n_no_close += 1
                    # Rollout never closed thinking: skip prefix mining
                    # (matches the safest training behavior).
                    continue
            else:
                post_ids = resp_ids
                start_pos = 0
                had_close = False

            boundaries = qp.find_prefix_boundaries(
                post_ids, newline_ids, min_gap=32, cap=15,
            )
            if not boundaries:
                n_no_boundary += 1
                continue

            rk = _cache_key(item["id"], resp_index)
            # Store the ORIGINAL (pre-strip) response ids for entropy;
            # the entropy row indexes into the whole response, then we
            # offset by start_pos when slicing per-prefix.
            _ROLLOUT_TOKENS_CACHE[rk] = resp_ids

            for boundary in boundaries:
                # boundary is in POST-think index space; map back to
                # the full response for entropy indexing.
                prefix_ids = post_ids[:boundary]
                prefix_len_in_response = start_pos + boundary
                prefix_text = tokenizer.decode(
                    prefix_ids, skip_special_tokens=True,
                )

                all_requests.append({
                    "problem_id": item["id"],
                    "problem": prompt,
                    "gt": gt,
                    "prefix_text": prefix_text,
                    "prefix_steps": [prefix_text],  # legacy field
                    "remaining_steps": [],
                    # prefix_fraction: fraction of the POST-think response
                    # consumed.  Retained for legacy plotting.
                    "prefix_fraction": (
                        float(boundary) / max(1, len(post_ids))
                    ),
                    "n_prefix_steps": 1,
                    "n_total_steps": len(boundaries) + 1,
                    "full_trajectory_correct": resp_data["is_correct"],
                    "full_response": full_response,
                    "formatted_prompt": formatted_prompt,
                    # ----- training-alignment metadata -----
                    "rollout_key": rk,
                    "prefix_len_tokens": prefix_len_in_response,
                    "post_think_start": start_pos,
                    "had_close_think": had_close,
                })
                prompts_for_generation.append(formatted_prompt + prefix_text)

    print(f"[qwen3-prefix] built {len(all_requests)} prefixes "
          f"(strip_thinking={strip_thinking}, "
          f"skipped no-</think>={n_no_close}, "
          f"skipped no-boundary={n_no_boundary})")
    return all_requests, prompts_for_generation


mc_orig._build_prefix_requests = _patched_build_prefix_requests


# ============================================================
# Metric computation: single-forward-pass MTE + no-truncation patch.
# ============================================================

_orig_compute_all_metrics_for_prefix = mc_orig.compute_all_metrics_for_prefix
_orig_tokenizer_call = None  # set inside _patch_truncation()


class _NoTruncationTokenizer:
    """Context manager: strip `truncation=True, max_length=2048` from
    every tokenizer call for the duration of the metric loop.

    We do this by monkey-patching the tokenizer's __call__ to override
    the two kwargs.  Restores on exit.  Chosen over editing the 10+
    call sites inside monte_carlo_experiment.py so the legacy Qwen2.5
    code path is untouched.
    """

    def __init__(self, tokenizer, max_context_tokens: int | None):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self._orig_call = None

    def __enter__(self):
        tok = self.tokenizer
        # tokenizers are callable via __call__ on the instance's class;
        # patching the bound method on the instance is safest.
        cap = self.max_context_tokens
        if cap is None:
            cap = getattr(tok, "model_max_length", None)
            if cap is None or cap > 10 ** 8:
                cap = 32768
        cap = int(cap)

        self._orig_call = tok.__call__

        def _patched_call(*args, **kwargs):
            # Drop the legacy 2048 cap; raise it to the real ceiling.
            if kwargs.get("truncation", False) and kwargs.get("max_length", None) == 2048:
                kwargs["max_length"] = cap
            return self._orig_call(*args, **kwargs)

        # Rebind on the instance.
        tok.__call__ = _patched_call  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_call is not None:
            try:
                self.tokenizer.__call__ = self._orig_call  # type: ignore[assignment]
            except Exception:
                pass
        return False


def _patched_compute_all_metrics_for_prefix(model, tokenizer, data_point):
    """Legacy metrics + a training-aligned `prefix_mte` column.

    `prefix_mte` (also mirrored as `prefix_entropy_mean` so the legacy
    plot code keeps working) is computed from a SINGLE per-rollout HF
    forward pass, cached in _ROLLOUT_ENTROPY_CACHE.  All other metrics
    fall through to the legacy function, but they run under
    _NoTruncationTokenizer, so `truncation=True, max_length=2048`
    calls silently see the model's true context window.
    """
    rk = data_point.get("rollout_key")
    prefix_len_tokens = int(data_point.get("prefix_len_tokens", 0))

    max_ctx = _CLI_OVERRIDES.get("max_context_tokens")

    metrics: dict[str, float] = {}

    # ------ 1. Legacy metrics under the no-truncation patch. ------
    with _NoTruncationTokenizer(tokenizer, max_ctx):
        try:
            legacy = _orig_compute_all_metrics_for_prefix(model, tokenizer, data_point)
            metrics.update(legacy)
        except Exception as e:
            print(f"[qwen3-metrics] legacy metrics failed: {e}")
            traceback.print_exc(file=sys.stdout)

    # ------ 2. Training-aligned prefix MTE via one forward pass. ------
    if rk is not None and rk in _ROLLOUT_TOKENS_CACHE:
        try:
            if rk not in _ROLLOUT_ENTROPY_CACHE:
                resp_ids = _ROLLOUT_TOKENS_CACHE[rk]
                ent_row = qp.mean_token_entropy_over_response(
                    model, tokenizer,
                    data_point["formatted_prompt"], resp_ids,
                    max_context_tokens=max_ctx,
                )
                _ROLLOUT_ENTROPY_CACHE[rk] = ent_row
            ent_row = _ROLLOUT_ENTROPY_CACHE[rk]
            L = max(1, min(int(prefix_len_tokens), ent_row.shape[0]))
            prefix_mte = float(ent_row[:L].mean())
            metrics["prefix_mte"] = prefix_mte
            # Alias so legacy plot / correlation code that hardcodes
            # `prefix_entropy_mean` keeps rendering the new number.
            metrics["prefix_entropy_mean"] = prefix_mte
        except Exception as e:
            print(f"[qwen3-metrics] prefix_mte failed: {e}")
            traceback.print_exc(file=sys.stdout)

    return metrics


mc_orig.compute_all_metrics_for_prefix = _patched_compute_all_metrics_for_prefix


# ============================================================
# Robustness: dump correlation results immediately (existing pattern).
# ============================================================

_orig_analyze_correlations = mc_orig.analyze_correlations


def _patched_analyze_correlations(prefix_data_points):
    correlation_results = _orig_analyze_correlations(prefix_data_points)

    serializable = {}
    for name, info in correlation_results.items():
        ser = {}
        for k, v in info.items():
            try:
                ser[k] = float(v) if hasattr(v, "__float__") else v
            except Exception:
                ser[k] = v
        serializable[name] = ser

    robust.safe_save_json(
        {
            "model_name": "Qwen/Qwen3-1.7B",
            "schema_version": SCHEMA_VERSION,
            "strip_thinking": _CLI_OVERRIDES.get("strip_thinking", True),
            "n_prefix_data_points": len(prefix_data_points),
            "correlations": serializable,
        },
        os.path.join(QWEN3_OUT_DIR, "prefix_correlations_raw.json"),
    )

    rows = []
    for name, info in correlation_results.items():
        rows.append({
            "metric": name,
            "within_problem_corr": float(info.get("within_problem_corr", float("nan"))),
            "within_problem_p": float(info.get("within_problem_p", float("nan"))),
            "pooled_corr": float(info.get("corr_pass_rate", float("nan"))),
            "pooled_p": float(info.get("p_pass_rate", float("nan"))),
            "binary_corr": float(info.get("corr_binary", float("nan"))),
            "n_problems_used": int(info.get("n_problems_used", 0)),
        })
    rows.sort(key=lambda r: -abs(r["within_problem_corr"]))
    robust.safe_save_json(
        {"rows": rows, "schema_version": SCHEMA_VERSION},
        os.path.join(QWEN3_OUT_DIR, "prefix_correlations_table.json"),
    )

    import csv
    csv_path = os.path.join(QWEN3_OUT_DIR, "prefix_correlations_table.csv")
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[robustness] saved CSV: {csv_path}")

    return correlation_results


mc_orig.analyze_correlations = _patched_analyze_correlations


# ----- Wrap each plotting function so a matplotlib failure is non-fatal -----
def _wrap_plot_fn(fn, label):
    def _safe(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"[robustness] plot {label!r} failed: {e}")
            traceback.print_exc(file=sys.stdout)
            return None
    return _safe


mc_orig.plot_pass_rate_distribution = _wrap_plot_fn(
    mc_orig.plot_pass_rate_distribution, "pass_rate_distribution")
mc_orig.plot_metric_vs_pass_rate = _wrap_plot_fn(
    mc_orig.plot_metric_vs_pass_rate, "metric_vs_pass_rate")
mc_orig.plot_value_trajectories = _wrap_plot_fn(
    mc_orig.plot_value_trajectories, "value_trajectories")


# ============================================================
# Schema-version guard.
# ============================================================

def _refuse_stale_cache() -> None:
    """If the metrics .pkl exists but has no matching schema marker,
    refuse to run and instruct the user to delete it.

    This prevents mixing a stale pre-fix run with post-fix analysis.
    """
    pkl = os.path.join(QWEN3_OUT_DIR, "2b_prefix_metrics_data.pkl")
    if not os.path.exists(pkl):
        return
    ok = False
    if os.path.exists(SCHEMA_MARKER_FILE):
        try:
            with open(SCHEMA_MARKER_FILE, "r") as f:
                marker = json.load(f)
            ok = int(marker.get("schema_version", 0)) == SCHEMA_VERSION
        except Exception:
            ok = False
    if not ok:
        raise SystemExit(
            f"[qwen3] refusing to load {pkl}: schema version mismatch.\n"
            f"  Expected SCHEMA_VERSION={SCHEMA_VERSION}, marker file at\n"
            f"  {SCHEMA_MARKER_FILE} missing or stale.\n"
            f"  Delete the .pkl (and 2b_prefix_value_data.pkl / "
            f"2b_prefix_collected_data.pkl if you want a fully fresh run) "
            f"and rerun."
        )


def _write_schema_marker() -> None:
    try:
        with open(SCHEMA_MARKER_FILE, "w") as f:
            json.dump({
                "schema_version": SCHEMA_VERSION,
                "strip_thinking": _CLI_OVERRIDES.get("strip_thinking", True),
            }, f, indent=2)
    except Exception as e:
        print(f"[qwen3] failed to write schema marker: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_problems", type=int, default=60,
                        help="Number of problems to use for the proxy "
                             "correlation study (default: 60).")
    parser.add_argument("--target_levels", nargs="+", type=int,
                        default=[5],
                        help="MATH-500 difficulty levels included in the "
                             "mixed pool (default: [5]).")
    parser.add_argument("--no_mixed_pool", action="store_true",
                        help="Disable the mixed (MATH-500 + AMC23 + "
                             "AIME24 + AIME25) pool and use MATH-500 only.")
    # New in this rebuild: training-alignment flags.
    strip_group = parser.add_mutually_exclusive_group()
    strip_group.add_argument("--strip-thinking", dest="strip_thinking",
                             action="store_true", default=True,
                             help="(default) Strip <think>...</think> from "
                                  "each rollout before prefix boundary "
                                  "detection.  Matches training-side "
                                  "behavior for Qwen3 thinking models.")
    strip_group.add_argument("--no-strip-thinking", dest="strip_thinking",
                             action="store_false",
                             help="Legacy Qwen2.5 semantics: treat the "
                                  "whole response as prefix material.")
    parser.add_argument("--max-context-tokens", type=int, default=None,
                        help="Cap on the HF forward-pass context length "
                             "used for metric computation.  Defaults to "
                             "the tokenizer's model_max_length (or 32768 "
                             "if the tokenizer reports an unrealistic "
                             "value).  Replaces the legacy 2048 cap.")
    args, _unknown = parser.parse_known_args()

    _CLI_OVERRIDES["n_problems"] = args.n_problems
    _CLI_OVERRIDES["target_levels"] = args.target_levels
    _CLI_OVERRIDES["mixed_pool"] = not args.no_mixed_pool
    _CLI_OVERRIDES["strip_thinking"] = bool(args.strip_thinking)
    _CLI_OVERRIDES["max_context_tokens"] = args.max_context_tokens

    print(f"[qwen3] n_problems={args.n_problems}, "
          f"target_levels={args.target_levels}, "
          f"mixed_pool={_CLI_OVERRIDES['mixed_pool']}, "
          f"strip_thinking={_CLI_OVERRIDES['strip_thinking']}, "
          f"max_context_tokens={_CLI_OVERRIDES['max_context_tokens']}")

    robust.dual_log(QWEN3_OUT_DIR)

    _refuse_stale_cache()

    config = _patched_Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    print(
        f"[qwen3] using {_n_gpus} GPU(s): "
        f"HF metric model on cuda:0, vLLM TP={TP_SIZE} on cuda:1..{_n_gpus - 1}"
    )

    mc_orig.Config = _patched_Config

    try:
        robust.patch_pyplot_savefig(QWEN3_OUT_DIR)
    except Exception as e:
        print(f"[robustness] could not patch pyplot.savefig: {e}")

    try:
        mc_orig.main()
    finally:
        mc_orig.Config = _OrigConfig

    _write_schema_marker()

    print(f"\n[done] Qwen3 monte-carlo outputs written under: {QWEN3_OUT_DIR}")


if __name__ == "__main__":
    main()
