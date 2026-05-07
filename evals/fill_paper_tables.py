#!/usr/bin/env python3
"""
Fill paper tables with experimental results.

Reads eval_outputs_final/summary_*.csv and eval_outputs_final/diversity_*.csv,
then writes a *copy* of main.tex with numbers filled in (main_filled.tex) and
a paper_status.md summarizing what's filled vs. placeholder.

Usage:
    python3 evals/fill_paper_tables.py \
        --summary_glob 'eval_outputs_final/summary_*.csv' \
        --diversity_glob 'eval_outputs_final/diversity_*.csv' \
        --paper_dir 'ARR_May___ExTra__Exploratory_Trajectory_Optimization_for_Language_Model_Reinforcement_Learning' \
        --output_tex main_filled.tex --status_md paper_status.md
"""

import argparse
import csv
import glob as glob_mod
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_summaries(pattern: str) -> dict:
    """
    Load all summary CSVs and return a nested dict:
      {experiment_name: {step: {task: {metric: value}}}}
    """
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for path in sorted(glob_mod.glob(pattern)):
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                exp = row.get("experiment", "").strip()
                step = row.get("step", "").strip()
                task = row.get("task", "").strip()
                if not exp or not step or not task:
                    continue
                for k, v in row.items():
                    if k not in ("experiment", "step", "task") and v:
                        try:
                            data[exp][step][task][k] = float(v)
                        except ValueError:
                            data[exp][step][task][k] = v
    return data


def load_diversity(pattern: str) -> list[dict]:
    """Load all diversity CSVs and return list of row dicts."""
    rows = []
    for path in sorted(glob_mod.glob(pattern)):
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Experiment -> table cell mapping
# ---------------------------------------------------------------------------
# Maps experiment name prefixes to their role in the paper tables.

# For tab:main_results (currently commented out)
MAIN_RESULTS_MAP = {
    # "ExTra-Curiosity" row -> 04_ExTra_CuriOnly
    "04_ExTra_CuriOnly": "ExTra-Curiosity",
    # "ExTra-Regen" row -> 03_ExTra_RegenOnly
    "03_ExTra_RegenOnly": "ExTra-Regen",
    # "ExTra-Full" row -> 02_ExTra_Full or 09_ExTra_Tau1
    "02_ExTra_Full": "ExTra-Full",
    "09_ExTra_Tau1": "ExTra-Full",
}

# For tab:ablation
# Row mapping: (curiosity, regen) -> experiment
ABLATION_MAP = {
    # Row 1: no-curi, no-regen -> 01_GRPO (already filled: 47.0)
    "01_GRPO": ("no", "no"),
    # Row 2: yes-curi, no-regen -> 04_ExTra_CuriOnly (already filled: 51.6)
    "04_ExTra_CuriOnly": ("yes", "no"),
    # Row 3: no-curi, yes-regen -> 03_ExTra_RegenOnly
    "03_ExTra_RegenOnly": ("no", "yes"),
    # Row 4: yes-curi, yes-regen -> 02_ExTra_Full or 09
    "02_ExTra_Full": ("yes", "yes"),
    "09_ExTra_Tau1": ("yes", "yes"),
}

# For tab:gamma_sensitivity
# alpha (novelty_reward_scale) -> experiment
GAMMA_MAP = {
    "03_ExTra_RegenOnly": "0.0",      # regen only ≈ alpha=0.0
    "10_ExTra_Alpha005": "0.05",
    "10b_ExTra_Alpha005_Tau1": "0.05",
    "09_ExTra_Tau1": "0.1",
    "11_ExTra_Alpha02": "0.2",
}


def find_experiment(data: dict, prefix: str) -> str | None:
    """Find an experiment key in data that starts with prefix."""
    for key in data:
        if key.startswith(prefix):
            return key
    return None


def get_math500_score(data: dict, exp_prefix: str, step: str = "200") -> float | None:
    """Get MATH-500 mean_score for an experiment at a given step."""
    exp_key = find_experiment(data, exp_prefix)
    if exp_key is None:
        return None
    step_data = data[exp_key].get(step, {})
    # Try different task name formats
    for task_name in ["math-500", "MATH-500", "math", "MATH"]:
        if task_name in step_data and "mean_score" in step_data[task_name]:
            return step_data[task_name]["mean_score"]
    return None


# ---------------------------------------------------------------------------
# LaTeX manipulation
# ---------------------------------------------------------------------------
PLACEHOLDER_MACRO = r"\newcommand{\PLACEHOLDER}[1]{\textcolor{red}{[\textbf{TODO: #1}]}}"


def fill_tables(tex: str, data: dict) -> tuple[str, list[dict]]:
    """
    Fill table cells in the LaTeX source. Returns (modified_tex, status_list).
    status_list entries: {table, row_desc, action, value_or_placeholder}
    """
    status = []

    # --- Ensure \PLACEHOLDER macro exists ---
    if r"\newcommand{\PLACEHOLDER}" not in tex:
        # Add after \usepackage{microtype} or before \begin{document}
        insert_point = tex.find(r"\begin{document}")
        if insert_point == -1:
            insert_point = 0
        tex = tex[:insert_point] + PLACEHOLDER_MACRO + "\n" + tex[insert_point:]

    # --- tab:ablation ---
    # Row 3: \ding{55} & \ding{51} & -- \\
    # Row 4: \ding{51} & \ding{51} & -- \\

    # Row 3 (no-curi, yes-regen) -> 03_ExTra_RegenOnly
    score_03 = get_math500_score(data, "03_ExTra_RegenOnly")
    ablation_r3_pattern = r"(\\ding\{55\}\s*&\s*\\ding\{51\}\s*&\s*)(--)(.*?\\\\)"
    if score_03 is not None:
        val = f"{score_03 * 100:.1f}"
        tex = re.sub(ablation_r3_pattern, rf"\g<1>{val}\g<3>", tex, count=1)
        status.append({"table": "tab:ablation", "row_desc": "no-curi yes-regen (row 3)",
                        "action": "filled", "value": val})
    else:
        placeholder = r"\PLACEHOLDER{03\_ExTra\_RegenOnly}"
        tex = re.sub(ablation_r3_pattern, rf"\g<1>{placeholder}\g<3>", tex, count=1)
        status.append({"table": "tab:ablation", "row_desc": "no-curi yes-regen (row 3)",
                        "action": "placeholder", "value": "03_ExTra_RegenOnly"})

    # Row 4 (yes-curi, yes-regen) -> 02_ExTra_Full or 09_ExTra_Tau1
    score_full = get_math500_score(data, "02_ExTra_Full") or get_math500_score(data, "09_ExTra_Tau1")
    ablation_r4_pattern = r"(\\ding\{51\}\s*&\s*\\ding\{51\}\s*&\s*)(--)(.*?\\\\)"
    if score_full is not None:
        val = f"{score_full * 100:.1f}"
        tex = re.sub(ablation_r4_pattern, rf"\g<1>{val}\g<3>", tex, count=1)
        status.append({"table": "tab:ablation", "row_desc": "yes-curi yes-regen (row 4)",
                        "action": "filled", "value": val})
    else:
        placeholder = r"\PLACEHOLDER{02\_ExTra\_Full or 09\_ExTra\_Tau1}"
        tex = re.sub(ablation_r4_pattern, rf"\g<1>{placeholder}\g<3>", tex, count=1)
        status.append({"table": "tab:ablation", "row_desc": "yes-curi yes-regen (row 4)",
                        "action": "placeholder", "value": "02_ExTra_Full or 09_ExTra_Tau1"})

    # Check existing pre-filled cells (47.0, 51.6) -- warn if inconsistent
    score_01 = get_math500_score(data, "01_GRPO")
    if score_01 is not None:
        existing_val = 47.0
        new_val = score_01 * 100
        if abs(new_val - existing_val) > 1.0:
            status.append({"table": "tab:ablation", "row_desc": "no-curi no-regen (row 1)",
                            "action": "WARNING",
                            "value": f"Existing {existing_val} may be Qwen-Instruct; R1-Distill gives {new_val:.1f}"})

    score_04 = get_math500_score(data, "04_ExTra_CuriOnly")
    if score_04 is not None:
        existing_val = 51.6
        new_val = score_04 * 100
        if abs(new_val - existing_val) > 1.0:
            status.append({"table": "tab:ablation", "row_desc": "yes-curi no-regen (row 2)",
                            "action": "WARNING",
                            "value": f"Existing {existing_val} may be Qwen-Instruct; R1-Distill gives {new_val:.1f}"})

    # --- tab:gamma_sensitivity ---
    gamma_rows = [
        ("0.0 (regen only)", "03_ExTra_RegenOnly"),
        ("0.05", "10_ExTra_Alpha005"),
        ("0.1 (default)", "09_ExTra_Tau1"),
        ("0.2", "11_ExTra_Alpha02"),
    ]

    for gamma_label, exp_prefix in gamma_rows:
        score = get_math500_score(data, exp_prefix)
        # Also try alternate experiments for 0.05
        if score is None and exp_prefix == "10_ExTra_Alpha005":
            score = get_math500_score(data, "10b_ExTra_Alpha005_Tau1")
            if score is not None:
                exp_prefix = "10b_ExTra_Alpha005_Tau1"

        # Match the row pattern: gamma_label & -- \\
        # The LaTeX has patterns like:  0.0 (regen only)  & -- \\
        escaped_label = re.escape(gamma_label)
        gamma_pattern = rf"({escaped_label}\s*&\s*)(--)(.*?\\\\)"

        if score is not None:
            val = f"{score * 100:.1f}"
            tex = re.sub(gamma_pattern, rf"\g<1>{val}\g<3>", tex, count=1)
            status.append({"table": "tab:gamma_sensitivity", "row_desc": f"gamma={gamma_label}",
                            "action": "filled", "value": val})
        else:
            placeholder_exp = exp_prefix.replace("_", r"\_")
            placeholder = rf"\PLACEHOLDER{{{placeholder_exp}}}"
            tex = re.sub(gamma_pattern, rf"\g<1>{placeholder}\g<3>", tex, count=1)
            status.append({"table": "tab:gamma_sensitivity", "row_desc": f"gamma={gamma_label}",
                            "action": "placeholder", "value": exp_prefix})

    # --- tab:main_results (commented out section) ---
    # Uncomment the table and fill what we can
    main_table_start = "% Table 1 commented out"
    if main_table_start in tex:
        # Find the commented block
        block_start = tex.find(main_table_start)
        # Find the end of the commented block (% \end{table})
        block_end = tex.find("% \\end{table}", block_start)
        if block_end != -1:
            block_end += len("% \\end{table}")
            commented_block = tex[block_start:block_end]

            # Uncomment: remove leading "% " from each line
            uncommented_lines = []
            for line in commented_block.split("\n"):
                if line.startswith("% "):
                    uncommented_lines.append(line[2:])
                elif line.startswith("%"):
                    uncommented_lines.append(line[1:])
                else:
                    uncommented_lines.append(line)
            uncommented = "\n".join(uncommented_lines)

            # Remove the "Table 1 commented out" comment line
            uncommented = uncommented.replace("Table 1 commented out -- using step-200 comparison instead\n", "")

            # Fill ExTra-Curiosity row
            score_curi = get_math500_score(data, "04_ExTra_CuriOnly")
            if score_curi is not None:
                val = f"{score_curi * 100:.1f}"
                # The existing row has 53.0 as Peak and \textbf{51.6} as Step 200
                # These are Qwen-Instruct numbers; update Step 200 column
                status.append({"table": "tab:main_results", "row_desc": "ExTra-Curiosity Step 200",
                                "action": "filled", "value": val})
            else:
                status.append({"table": "tab:main_results", "row_desc": "ExTra-Curiosity Step 200",
                                "action": "placeholder", "value": "04_ExTra_CuriOnly"})

            # Fill ExTra-Regen row
            score_regen = get_math500_score(data, "03_ExTra_RegenOnly")
            if score_regen is not None:
                val = f"{score_regen * 100:.1f}"
                uncommented = uncommented.replace(
                    "ExTra-Regen            & -- & -- & -- \\\\",
                    f"ExTra-Regen            & -- & -- & {val} \\\\"
                )
                status.append({"table": "tab:main_results", "row_desc": "ExTra-Regen Step 200",
                                "action": "filled", "value": val})
            else:
                uncommented = uncommented.replace(
                    "ExTra-Regen            & -- & -- & -- \\\\",
                    "ExTra-Regen            & -- & -- & \\PLACEHOLDER{03\\_ExTra\\_RegenOnly} \\\\"
                )
                status.append({"table": "tab:main_results", "row_desc": "ExTra-Regen Step 200",
                                "action": "placeholder", "value": "03_ExTra_RegenOnly"})

            # Fill ExTra-Full row
            if score_full is not None:
                val = f"{score_full * 100:.1f}"
                uncommented = uncommented.replace(
                    "ExTra-Full             & -- & -- & -- \\\\",
                    f"ExTra-Full             & -- & -- & {val} \\\\"
                )
                status.append({"table": "tab:main_results", "row_desc": "ExTra-Full Step 200",
                                "action": "filled", "value": val})
            else:
                uncommented = uncommented.replace(
                    "ExTra-Full             & -- & -- & -- \\\\",
                    "ExTra-Full             & -- & -- & \\PLACEHOLDER{02\\_ExTra\\_Full} \\\\"
                )
                status.append({"table": "tab:main_results", "row_desc": "ExTra-Full Step 200",
                                "action": "placeholder", "value": "02_ExTra_Full or 09_ExTra_Tau1"})

            # Fill GRPO row Step 200 if we have R1-Distill data
            if score_01 is not None:
                val_01 = f"{score_01 * 100:.1f}"
                uncommented = uncommented.replace(
                    "GRPO Baseline          & 53.8 & -- & 47.0 \\\\",
                    f"GRPO Baseline          & -- & -- & {val_01} \\\\"
                )
                status.append({"table": "tab:main_results", "row_desc": "GRPO Step 200",
                                "action": "filled", "value": val_01})

            tex = tex[:block_start] + uncommented + tex[block_end:]

    return tex, status


def write_status_md(status: list[dict], diversity_rows: list[dict], out_path: str):
    """Write a paper_status.md summarizing filled vs. placeholder cells."""
    with open(out_path, "w") as f:
        f.write("# Paper Table Status\n\n")
        f.write(f"Generated by `fill_paper_tables.py`\n\n")

        # Filled cells
        filled = [s for s in status if s["action"] == "filled"]
        if filled:
            f.write("## Filled Cells\n\n")
            for s in filled:
                f.write(f"- **{s['table']}** {s['row_desc']}: **{s['value']}**\n")
            f.write("\n")

        # Placeholder cells
        placeholders = [s for s in status if s["action"] == "placeholder"]
        if placeholders:
            f.write("## Placeholder Cells (need data from other server)\n\n")
            for s in placeholders:
                f.write(f"- **{s['table']}** {s['row_desc']} → run **{s['value']}**\n")
            f.write("\n")

        # Warnings
        warnings = [s for s in status if s["action"] == "WARNING"]
        if warnings:
            f.write("## Warnings\n\n")
            for s in warnings:
                f.write(f"- **{s['table']}** {s['row_desc']}: {s['value']}\n")
            f.write("\n")

        # Other-server runs needed summary
        if placeholders:
            f.write("## Other-server runs needed for paper tables\n\n")
            seen = set()
            for s in placeholders:
                key = f"{s['table']}: {s['row_desc']} → {s['value']}"
                if key not in seen:
                    f.write(f"- {s['table']}, {s['row_desc']} → run **{s['value']}**\n")
                    seen.add(key)
            f.write("\n")

        # Diversity summary
        if diversity_rows:
            f.write("## Diversity Comparison Summary\n\n")
            delta_rows = [r for r in diversity_rows if r.get("method", "").startswith("delta")]
            if delta_rows:
                f.write("| Task | Δ cosine_dist | Δ logdet_vol | Δ pass@1 |\n")
                f.write("|------|--------------|-------------|----------|\n")
                for r in delta_rows:
                    f.write(f"| {r.get('task', '')} | {r.get('avg_cosine_distance', '')} | "
                            f"{r.get('avg_logdet_volume', '')} | {r.get('pass@1', '')} |\n")
            f.write("\n")

    print(f"Wrote status to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fill paper tables with eval results")
    parser.add_argument("--summary_glob", default="eval_outputs_final/summary_*.csv",
                        help="Glob pattern for summary CSVs")
    parser.add_argument("--diversity_glob", default="eval_outputs_final/diversity_*.csv",
                        help="Glob pattern for diversity CSVs")
    parser.add_argument("--paper_dir",
                        default="ARR_May___ExTra__Exploratory_Trajectory_Optimization_for_Language_Model_Reinforcement_Learning",
                        help="Directory containing main.tex")
    parser.add_argument("--output_tex", default="main_filled.tex",
                        help="Output filename for filled tex (written in paper_dir)")
    parser.add_argument("--status_md", default="paper_status.md",
                        help="Output filename for status markdown (written in paper_dir)")
    parser.add_argument("--overwrite_existing", action="store_true",
                        help="Overwrite existing pre-filled numbers (e.g. Qwen-Instruct 47.0, 51.6)")
    args = parser.parse_args()

    # Load data
    data = load_summaries(args.summary_glob)
    diversity_rows = load_diversity(args.diversity_glob)

    if not data:
        print("WARNING: No summary data found. Tables will use placeholders for all cells.",
              file=sys.stderr)

    # Read original tex
    tex_path = os.path.join(args.paper_dir, "main.tex")
    if not os.path.exists(tex_path):
        print(f"ERROR: {tex_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(tex_path) as f:
        tex = f.read()

    # Fill tables
    filled_tex, status = fill_tables(tex, data)

    # Write output
    out_tex_path = os.path.join(args.paper_dir, args.output_tex)
    with open(out_tex_path, "w") as f:
        f.write(filled_tex)
    print(f"Wrote filled tex to {out_tex_path}")

    out_status_path = os.path.join(args.paper_dir, args.status_md)
    write_status_md(status, diversity_rows, out_status_path)

    # Print summary to stdout
    print("\n--- Summary ---")
    filled_count = sum(1 for s in status if s["action"] == "filled")
    placeholder_count = sum(1 for s in status if s["action"] == "placeholder")
    warning_count = sum(1 for s in status if s["action"] == "WARNING")
    print(f"Filled: {filled_count} cells")
    print(f"Placeholders: {placeholder_count} cells")
    print(f"Warnings: {warning_count}")


if __name__ == "__main__":
    main()
