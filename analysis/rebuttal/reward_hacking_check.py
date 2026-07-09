#!/usr/bin/env python3
"""
reward_hacking_check.py — EMNLP rebuttal analysis A5.

Compares ExTra vs GRPO eval outputs on three sanity indicators that would
reveal reward-hacking / degenerate-generation behavior. Answers XB9Q Q7.

For each pair (ref_run, cmp_run) and each benchmark, reports:
  * mean response length (tokens if `response_length` present, else chars)
  * malformed-answer rate (no `\\boxed{...}` in response)
  * verifier-accept-but-no-final-answer rate (correct=1 but no boxed answer)

Inputs are the JSONL rollout files under {eval_dir}/{run}/step_{N}/. Any
existing grading field is respected (is_correct, correct, score).

Usage:
  python analysis/rebuttal/reward_hacking_check.py \
    --eval_dir ./eval_outputs_rebuttal \
    --ref_run 01_GRPO_NanoNemotron_8B \
    --cmp_run 02_ExTra_Full_NanoNemotron_8B \
    --step 150 \
    --out reward_hack.md
"""

import argparse
import json
import re
from pathlib import Path


BOXED_RE = re.compile(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")


def load_jsonl(step_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for jsonl in sorted(step_dir.glob("*.jsonl")):
        bench = jsonl.stem.split("_t")[0]
        rows = [json.loads(x) for x in jsonl.read_text().splitlines() if x.strip()]
        out[bench] = rows
    return out


def _resp(row) -> str:
    for k in ("response", "output", "generation", "text"):
        if k in row and isinstance(row[k], str):
            return row[k]
    return ""


def _correct(row) -> int | None:
    for k in ("is_correct", "correct", "score"):
        if k in row:
            return int(bool(row[k]))
    return None


def summarize_run(rows_by_bench: dict[str, list[dict]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for bench, rows in rows_by_bench.items():
        if not rows:
            continue
        lens = []
        malformed = 0
        accept_no_boxed = 0
        n = 0
        for r in rows:
            resp = _resp(r)
            if not resp:
                continue
            n += 1
            # length: prefer explicit length field if present, else chars.
            length = r.get("response_length") or r.get("output_length")
            if length is None:
                length = len(resp)
            lens.append(int(length))
            has_boxed = bool(BOXED_RE.search(resp))
            if not has_boxed:
                malformed += 1
            corr = _correct(r)
            if corr == 1 and not has_boxed:
                accept_no_boxed += 1
        out[bench] = {
            "n": n,
            "mean_len": round(sum(lens) / max(len(lens), 1), 1),
            "malformed_rate": round(malformed / max(n, 1), 4),
            "verifier_accept_no_boxed_rate": round(accept_no_boxed / max(n, 1), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--ref_run", required=True)
    ap.add_argument("--cmp_run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", default="reward_hack.md")
    args = ap.parse_args()

    root = Path(args.eval_dir)
    ref_dir = root / args.ref_run / f"step_{args.step}"
    cmp_dir = root / args.cmp_run / f"step_{args.step}"
    if not ref_dir.exists() or not cmp_dir.exists():
        print(f"[reward_hack] missing dir: {ref_dir} or {cmp_dir}")
        return

    ref = summarize_run(load_jsonl(ref_dir))
    cmp = summarize_run(load_jsonl(cmp_dir))

    lines = [
        "# Reward-hacking sanity check\n",
        f"* Reference (baseline): `{args.ref_run}` @ step {args.step}",
        f"* Compared (method):   `{args.cmp_run}` @ step {args.step}\n",
        "| Benchmark | Mean len (ref → cmp, Δ) | Malformed % (ref → cmp) | "
        "Verifier-accept-w/o-boxed % (ref → cmp) |",
        "|---|---|---|---|",
    ]
    for bench in sorted(set(ref) | set(cmp)):
        r = ref.get(bench, {})
        c = cmp.get(bench, {})
        lines.append(
            "| {b} | {rl:.1f} → {cl:.1f} (Δ={dl:+.1f}) | "
            "{rm:.2%} → {cm:.2%} | {ra:.2%} → {ca:.2%} |".format(
                b=bench,
                rl=r.get("mean_len", 0),
                cl=c.get("mean_len", 0),
                dl=c.get("mean_len", 0) - r.get("mean_len", 0),
                rm=r.get("malformed_rate", 0),
                cm=c.get("malformed_rate", 0),
                ra=r.get("verifier_accept_no_boxed_rate", 0),
                ca=c.get("verifier_accept_no_boxed_rate", 0),
            )
        )
    lines.append(
        "\nInterpretation: if `cmp` mean_len is much larger than `ref`, or "
        "malformed rate rises, or verifier-accept-w/o-boxed rate rises, the "
        "method may be exploiting the verifier. Otherwise, no evidence of "
        "reward hacking.\n"
    )
    Path(args.out).write_text("\n".join(lines))
    print(f"[reward_hack] wrote {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
