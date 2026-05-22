"""
Lightweight stand-ins for the original `simplified_evaluator` and
`openrlhf.trainer.ppo_utils.score` modules used by the legacy Qwen2.5
prefix-analysis scripts.  These shims let the Qwen3 wrappers run in any
environment where verl is installed, without requiring those external
research repos to be present.

Place this file in `analysis/`. When a script does
    from simplified_evaluator.eval import parse_prediction
or
    from openrlhf.trainer.ppo_utils.score import process_thoughts
the loader walks `sys.path`, which includes the `analysis/` directory,
and finds the shim packages we registered below.

The shims are intentionally minimal:
- parse_prediction: extract a \\boxed{...} answer; fall back to the last
  numeric/short token sequence after "answer is" or similar.
- process_thoughts: split a response into reasoning step strings using
  newline-based segmentation matching the verl implementation.
"""

import importlib
import importlib.util
import re
import sys
import types
from pathlib import Path


# --------------------------------------------------------------------------
# `simplified_evaluator.eval.parse_prediction`
# --------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ANSWER_HINT_RE = re.compile(r"(?:answer\s*is|=\s*)\s*([^\n.]+)", re.IGNORECASE)


def _extract_boxed(text: str) -> str:
    """Return the contents of the LAST \\boxed{...} in the text, or ''."""
    matches = _BOXED_RE.findall(text or "")
    return matches[-1].strip() if matches else ""


def parse_prediction(response: str, gt: str = "", task: str = "math") -> str:
    """
    Best-effort extractor that returns the model's predicted answer.
    Tries `\\boxed{...}` first, then "answer is X" patterns, else returns
    the last non-empty line trimmed.
    """
    if not isinstance(response, str):
        return ""
    boxed = _extract_boxed(response)
    if boxed:
        return boxed
    m = _ANSWER_HINT_RE.search(response)
    if m:
        return m.group(1).strip().strip(".")
    # final-line fallback
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    return lines[-1] if lines else ""


# --------------------------------------------------------------------------
# `openrlhf.trainer.ppo_utils.score.process_thoughts`
# --------------------------------------------------------------------------
# Copy of the implementation in
# verl/verl/trainer/ppo/metric_utils.py::process_thoughts so we don't
# require an importable openrlhf install.

_COLON_END_RE = re.compile(r":\s*$")


def _merge_colon_ended_elements(steps):
    out = []
    buf = ""
    for s in steps:
        if _COLON_END_RE.search(s):
            buf = (buf + " " + s).strip() if buf else s
        else:
            if buf:
                out.append((buf + " " + s).strip())
                buf = ""
            else:
                out.append(s)
    if buf:
        out.append(buf)
    return out


def _merge_steps(steps, target_count: int = 14):
    if len(steps) <= target_count:
        return steps
    bucket = max(1, len(steps) // target_count)
    out = []
    for i in range(0, len(steps), bucket):
        out.append(" ".join(steps[i:i + bucket]).strip())
    return out


def process_thoughts(resp: str):
    if not isinstance(resp, str):
        return []
    thoughts = [line.strip() for line in resp.split("\n") if line.strip()]
    result = []
    merge_mode = False
    temp_merge = []

    for item in thoughts:
        if "\\[" in item and "\\]" not in item:
            merge_mode = True
            temp_merge.append(item)
        elif "\\]" in item and "\\[" not in item:
            merge_mode = False
            temp_merge.append(item)
            if temp_merge:
                result.append("\n".join(temp_merge))
                temp_merge = []
        elif merge_mode:
            temp_merge.append(item)
        else:
            result.append(item)

    if len(temp_merge) > 0:
        result += temp_merge

    if len(result) >= 15:
        result = _merge_colon_ended_elements(result)
    if len(result) >= 15:
        result = _merge_steps(result)

    return result


# --------------------------------------------------------------------------
# Register in-memory packages so the legacy `from simplified_evaluator.eval
# import parse_prediction` and `from openrlhf.trainer.ppo_utils.score import
# process_thoughts` lines resolve.
# --------------------------------------------------------------------------

def _install_shims():
    # simplified_evaluator.eval
    se = types.ModuleType("simplified_evaluator")
    se_eval = types.ModuleType("simplified_evaluator.eval")
    se_eval.parse_prediction = parse_prediction
    se.eval = se_eval
    sys.modules.setdefault("simplified_evaluator", se)
    sys.modules.setdefault("simplified_evaluator.eval", se_eval)

    # openrlhf.trainer.ppo_utils.score
    orh = types.ModuleType("openrlhf")
    orh_t = types.ModuleType("openrlhf.trainer")
    orh_pu = types.ModuleType("openrlhf.trainer.ppo_utils")
    orh_score = types.ModuleType("openrlhf.trainer.ppo_utils.score")
    orh_score.process_thoughts = process_thoughts
    orh_pu.score = orh_score
    orh_t.ppo_utils = orh_pu
    orh.trainer = orh_t
    sys.modules.setdefault("openrlhf", orh)
    sys.modules.setdefault("openrlhf.trainer", orh_t)
    sys.modules.setdefault("openrlhf.trainer.ppo_utils", orh_pu)
    sys.modules.setdefault("openrlhf.trainer.ppo_utils.score", orh_score)


_install_shims()
