#!/usr/bin/env python3
"""Plot training accuracy and MATH-500 test accuracy for GRPO Baseline vs NoRegen (Curiosity Only).

Usage:
    python analysis/plot_training_curves.py
"""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths to wandb output logs ───────────────────────────────────────────────
# Best complete single runs (300 steps each)
BASELINE_LOG = "verl/wandb/run-20260321_171335-lnaead62/files/output.log"
NOREGEN_LOG = "verl/wandb/run-20260405_213706-b4nj48mr/files/output.log"


def parse_log(path):
    """Return (steps, train_accs, val_steps, val_accs) from a wandb output.log."""
    steps, train_accs, val_steps, val_accs = [], [], [], []
    with open(path) as f:
        for line in f:
            m_step = re.search(r"training/global_step:(\d+)", line)
            m_score = re.search(r"critic/score/mean:([\d.eE+-]+)", line)
            m_val = re.search(r"val-core.*?reward/mean@1:([\d.eE+-]+)", line)
            if m_step and m_score:
                s = int(m_step.group(1))
                score = float(m_score.group(1))
                train_acc = (1.0 + score) / 2.0  # map [-1, 1] -> [0, 1]
                steps.append(s)
                train_accs.append(train_acc)
                if m_val:
                    val_steps.append(s)
                    val_accs.append(float(m_val.group(1)))
    return steps, train_accs, val_steps, val_accs


def smooth(values, window=5):
    """Simple moving average for noisy training curves."""
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ── Parse data ───────────────────────────────────────────────────────────────
MAX_STEP = 200  # Truncate to 200 steps

def truncate(steps, vals, max_step):
    """Keep only data up to max_step."""
    out_s, out_v = [], []
    for s, v in zip(steps, vals):
        if s <= max_step:
            out_s.append(s)
            out_v.append(v)
    return out_s, out_v

bl_train_steps, bl_train_accs, bl_val_steps, bl_val_accs = parse_log(BASELINE_LOG)
nr_train_steps, nr_train_accs, nr_val_steps, nr_val_accs = parse_log(NOREGEN_LOG)

bl_train_steps, bl_train_accs = truncate(bl_train_steps, bl_train_accs, MAX_STEP)
nr_train_steps, nr_train_accs = truncate(nr_train_steps, nr_train_accs, MAX_STEP)
bl_val_steps, bl_val_accs = truncate(bl_val_steps, bl_val_accs, MAX_STEP)
nr_val_steps, nr_val_accs = truncate(nr_val_steps, nr_val_accs, MAX_STEP)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

# Color palette (colorblind-friendly)
C_BL = "#4878D0"   # blue
C_NR = "#D65F5F"   # red

SMOOTH_TRAIN = 9
SMOOTH_VAL = 3

# ── Left: Training Accuracy ──────────────────────────────────────────────────
# Faint raw data + bold smoothed line
ax1.plot(bl_train_steps, bl_train_accs, color=C_BL, alpha=0.12, linewidth=0.8)
ax1.plot(nr_train_steps, nr_train_accs, color=C_NR, alpha=0.12, linewidth=0.8)
ax1.plot(bl_train_steps, smooth(bl_train_accs, SMOOTH_TRAIN),
         color=C_BL, alpha=0.9, linewidth=2, label="GRPO Baseline")
ax1.plot(nr_train_steps, smooth(nr_train_accs, SMOOTH_TRAIN),
         color=C_NR, alpha=0.9, linewidth=2, label="ExTra-Curiosity")

ax1.set_xlabel("Training Step", fontsize=11)
ax1.set_ylabel("Training Accuracy", fontsize=11)
ax1.set_title("Training Accuracy (MATH-DAPO)", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax1.set_xlim(0, max(max(bl_train_steps), max(nr_train_steps)))
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.25, linestyle="--")
ax1.tick_params(labelsize=9)

# ── Right: Test Accuracy (MATH-500) ─────────────────────────────────────────
# Faint raw data + bold smoothed line
ax2.plot(bl_val_steps, bl_val_accs, color=C_BL, alpha=0.2, linewidth=0.8,
         marker="o", markersize=2.5)
ax2.plot(nr_val_steps, nr_val_accs, color=C_NR, alpha=0.2, linewidth=0.8,
         marker="s", markersize=2.5)
ax2.plot(bl_val_steps, smooth(bl_val_accs, SMOOTH_VAL),
         color=C_BL, alpha=0.9, linewidth=2.2, label="GRPO Baseline")
ax2.plot(nr_val_steps, smooth(nr_val_accs, SMOOTH_VAL),
         color=C_NR, alpha=0.9, linewidth=2.2, label="ExTra-Curiosity")

ax2.set_xlabel("Training Step", fontsize=11)
ax2.set_ylabel("Pass@1 Accuracy", fontsize=11)
ax2.set_title("Test Accuracy (MATH-500)", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax2.set_xlim(0, max(max(bl_val_steps), max(nr_val_steps)))
ax2.set_ylim(0.30, 0.58)
ax2.grid(True, alpha=0.25, linestyle="--")
ax2.tick_params(labelsize=9)

plt.tight_layout()
out_path = "analysis/training_curves.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved to {out_path}")

# Also save a PNG for quick viewing
out_png = "analysis/training_curves.png"
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved to {out_png}")
