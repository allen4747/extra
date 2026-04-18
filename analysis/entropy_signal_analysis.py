#!/usr/bin/env python3
"""
Entropy Signal Analysis for ExTra — strengthens the case for entropy-guided resampling.

Experiments:
  A. Smoothed vs raw entropy correlation with pass rate
  B. Top-k selection accuracy (practical task-relevant metric)
  C. End-to-end resampling pass@k lift (via simulation on collected data)
  D. Hybrid signal exploration (entropy + layer consistency)

Usage:
    python analysis/entropy_signal_analysis.py
"""

import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "prefix_metrics_data.pkl"
OUT_DIR = Path(__file__).parent
SMOOTHING_TAU = 0.1


def load_data():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} prefix entries across "
          f"{len(set(d['problem_id'] for d in data))} problems")
    return data


def group_by_problem(data):
    """Group prefix entries by problem_id."""
    groups = defaultdict(list)
    for d in data:
        groups[d["problem_id"]].append(d)
    return dict(groups)


# ═════════════════════════════════════════════════════════════════════════════
#  Smoothing (same as select_best_prefix in ray_trainer.py)
# ═════════════════════════════════════════════════════════════════════════════

def compute_smoothed_entropy(entries, tau=SMOOTHING_TAU):
    """Compute embedding-smoothed entropy for a list of prefix entries.

    Mimics CuriosityMemory.select_best_prefix: uses cosine similarity of
    semantic embeddings to weight-average raw entropy scores.
    """
    # Use semantic_sim_gt embedding if available, else fall back to raw entropy
    # Since we don't have raw embeddings stored, we approximate smoothing
    # using the semantic similarity between prefixes within the same problem.
    # We use the prefix_entropy_mean as the raw score.
    raw = np.array([e["metrics"]["prefix_entropy_mean"] for e in entries])
    n = len(entries)

    if n <= 1:
        return raw

    # Build similarity matrix from prefix text overlap (token Jaccard)
    # This approximates the embedding-based smoothing in the trainer
    from sentence_transformers import SentenceTransformer
    model = _get_emb_model()
    texts = [e["prefix_text"] for e in entries]
    embs = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    sim = embs @ embs.T

    weights = _softmax_2d(sim / max(tau, 1e-6))
    smoothed = weights @ raw
    return smoothed


_EMB_MODEL = None
def _get_emb_model():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _EMB_MODEL


def _softmax_2d(x):
    """Row-wise softmax for a 2D numpy array."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Correlation Analysis
# ═════════════════════════════════════════════════════════════════════════════

def fisher_z(rho):
    """Fisher z-transform for averaging correlations."""
    rho = np.clip(rho, -0.9999, 0.9999)
    return np.arctanh(rho)


def inv_fisher_z(z):
    return np.tanh(z)


def within_problem_correlation(groups, metric_fn):
    """Compute within-problem Spearman ρ (Fisher z-averaged)."""
    z_vals = []
    for pid, entries in groups.items():
        if len(entries) < 4:
            continue
        scores = np.array([metric_fn(e) for e in entries])
        pass_rates = np.array([e["pass_rate"] for e in entries])
        if np.std(scores) < 1e-10 or np.std(pass_rates) < 1e-10:
            continue
        rho, _ = stats.spearmanr(scores, pass_rates)
        if not np.isnan(rho):
            z_vals.append(fisher_z(rho))
    if not z_vals:
        return 0.0, 0
    return inv_fisher_z(np.mean(z_vals)), len(z_vals)


def pooled_correlation(data, metric_fn):
    """Compute pooled (global) Spearman ρ."""
    scores = np.array([metric_fn(e) for e in data])
    pass_rates = np.array([e["pass_rate"] for e in data])
    rho, p = stats.spearmanr(scores, pass_rates)
    return rho, p


# ═════════════════════════════════════════════════════════════════════════════
#  Experiment A: Smoothed vs Raw Correlation
# ═════════════════════════════════════════════════════════════════════════════

def experiment_a_smoothed_correlation(data, groups):
    print("\n" + "=" * 70)
    print("  EXPERIMENT A: Raw vs Smoothed Entropy Correlation")
    print("=" * 70)

    # Compute smoothed entropy for each entry
    smoothed_map = {}  # (problem_id, prefix_idx) -> smoothed_entropy
    for pid, entries in groups.items():
        smoothed = compute_smoothed_entropy(entries)
        for i, e in enumerate(entries):
            smoothed_map[id(e)] = smoothed[i]

    # Metric functions
    def raw_entropy(e):
        return e["metrics"]["prefix_entropy_mean"]

    def smoothed_entropy(e):
        return smoothed_map[id(e)]

    def layer_consistency(e):
        return e["metrics"]["layer_consistency"]

    def hybrid_entropy_layer(e):
        return e["metrics"]["prefix_entropy_mean"] + 0.3 * e["metrics"]["layer_consistency"]

    metrics = [
        ("Prefix Entropy (Raw)", raw_entropy),
        ("Prefix Entropy (Smoothed)", smoothed_entropy),
        ("Layer Consistency", layer_consistency),
        ("Hybrid (Entropy + 0.3*Layer)", hybrid_entropy_layer),
    ]

    print(f"\n{'Metric':<35} {'Within-ρ':>10} {'Pooled-ρ':>10} {'n_problems':>12}")
    print("-" * 70)
    results = {}
    for name, fn in metrics:
        wp_rho, n_prob = within_problem_correlation(groups, fn)
        p_rho, p_val = pooled_correlation(data, fn)
        print(f"{name:<35} {wp_rho:>+10.4f} {p_rho:>+10.4f} {n_prob:>12}")
        results[name] = {"within_rho": wp_rho, "pooled_rho": p_rho}

    return results, smoothed_map


# ═════════════════════════════════════════════════════════════════════════════
#  Experiment B: Top-k Selection Accuracy
# ═════════════════════════════════════════════════════════════════════════════

def experiment_b_selection_accuracy(groups, smoothed_map):
    print("\n" + "=" * 70)
    print("  EXPERIMENT B: Top-k Selection Accuracy")
    print("=" * 70)
    print("  (When selecting the prefix with lowest score, what fraction")
    print("   of the time is it in the top-Q% by pass rate?)\n")

    strategies = {
        "Random": lambda e: np.random.random(),
        "Raw Entropy (min)": lambda e: e["metrics"]["prefix_entropy_mean"],
        "Smoothed Entropy (min)": lambda e: smoothed_map[id(e)],
        "Layer Consistency (min)": lambda e: e["metrics"]["layer_consistency"],
        "Hybrid (min)": lambda e: e["metrics"]["prefix_entropy_mean"] + 0.3 * e["metrics"]["layer_consistency"],
    }

    quartiles = [25, 50]  # top-25%, top-50%
    n_random_trials = 1000

    print(f"{'Strategy':<30}", end="")
    for q in quartiles:
        print(f"  {'Top-' + str(q) + '%':>8}", end="")
    print(f"  {'Avg PR of selected':>18}")
    print("-" * 80)

    all_results = {}
    for name, score_fn in strategies.items():
        hits = {q: [] for q in quartiles}
        selected_pass_rates = []

        trials = n_random_trials if name == "Random" else 1
        for _ in range(trials):
            for pid, entries in groups.items():
                if len(entries) < 4:
                    continue
                pass_rates = np.array([e["pass_rate"] for e in entries])
                scores = np.array([score_fn(e) for e in entries])
                best_idx = np.argmin(scores)
                selected_pr = pass_rates[best_idx]
                selected_pass_rates.append(selected_pr)

                for q in quartiles:
                    threshold = np.percentile(pass_rates, 100 - q)
                    hits[q].append(1.0 if selected_pr >= threshold else 0.0)

        row = {}
        print(f"{name:<30}", end="")
        for q in quartiles:
            acc = np.mean(hits[q])
            print(f"  {acc:>7.1%}", end="")
            row[f"top_{q}"] = acc
        avg_pr = np.mean(selected_pass_rates)
        print(f"  {avg_pr:>18.4f}")
        row["avg_pass_rate"] = avg_pr
        all_results[name] = row

    # Expected random baseline
    print(f"\n  (Random baseline: Top-25% = 25.0%, Top-50% = 50.0%)")

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
#  Experiment C: Simulated Resampling Pass@k Lift
# ═════════════════════════════════════════════════════════════════════════════

def experiment_c_resampling_lift(groups, smoothed_map):
    print("\n" + "=" * 70)
    print("  EXPERIMENT C: Resampling Lift (simulated from prefix pass rates)")
    print("=" * 70)
    print("  For each problem, we compare:")
    print("    Random:     pick a random prefix, its pass@k = prefix pass_rate")
    print("    Entropy:    pick the lowest-entropy prefix")
    print("    Smoothed:   pick the lowest smoothed-entropy prefix")
    print("    Oracle:     pick the highest pass_rate prefix (upper bound)\n")

    strategies = {
        "Random prefix": lambda entries: entries[np.random.randint(len(entries))],
        "Raw Entropy (min)": lambda entries: min(entries, key=lambda e: e["metrics"]["prefix_entropy_mean"]),
        "Smoothed Entropy (min)": lambda entries: min(entries, key=lambda e: smoothed_map[id(e)]),
        "Hybrid (min)": lambda entries: min(entries, key=lambda e: e["metrics"]["prefix_entropy_mean"] + 0.3 * e["metrics"]["layer_consistency"]),
        "Oracle (best pass_rate)": lambda entries: max(entries, key=lambda e: e["pass_rate"]),
    }

    K_values = [1, 4, 8, 16]
    n_random_trials = 1000

    print(f"{'Strategy':<30}", end="")
    for k in K_values:
        print(f"  {'pass@' + str(k):>8}", end="")
    print()
    print("-" * 70)

    all_results = {}
    for name, select_fn in strategies.items():
        trials = n_random_trials if "Random" in name else 1
        pass_at_k = {k: [] for k in K_values}

        for _ in range(trials):
            for pid, entries in groups.items():
                if len(entries) < 4:
                    continue
                selected = select_fn(entries)
                p = selected["pass_rate"]
                for k in K_values:
                    # pass@k = 1 - (1-p)^k
                    pass_at_k[k].append(1.0 - (1.0 - p) ** k)

        row = {}
        print(f"{name:<30}", end="")
        for k in K_values:
            val = np.mean(pass_at_k[k])
            print(f"  {val:>8.4f}", end="")
            row[f"pass@{k}"] = val
        print()
        all_results[name] = row

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
#  Experiment D: Hybrid Signal Grid Search
# ═════════════════════════════════════════════════════════════════════════════

def experiment_d_hybrid_search(data, groups):
    print("\n" + "=" * 70)
    print("  EXPERIMENT D: Hybrid Signal Grid Search")
    print("=" * 70)
    print("  Testing: score = α * entropy_mean + β * metric_2\n")

    second_metrics = [
        ("layer_consistency", "Layer Consistency"),
        ("prefix_entropy_max", "Entropy Max"),
        ("prefix_entropy_std", "Entropy Std"),
    ]

    best_overall = None
    for metric_key, metric_name in second_metrics:
        print(f"  Entropy + {metric_name}:")
        best_for_metric = None
        for beta in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
            def hybrid_fn(e, _beta=beta, _key=metric_key):
                return e["metrics"]["prefix_entropy_mean"] + _beta * e["metrics"][_key]

            wp_rho, n = within_problem_correlation(groups, hybrid_fn)
            marker = ""
            if best_for_metric is None or abs(wp_rho) > abs(best_for_metric[1]):
                best_for_metric = (beta, wp_rho, metric_name)
                marker = " <--"
            if best_overall is None or abs(wp_rho) > abs(best_overall[1]):
                best_overall = (beta, wp_rho, metric_name)
            print(f"    β={beta:.1f}  within-ρ={wp_rho:+.4f}{marker}")
        print()

    if best_overall:
        print(f"  Best hybrid: Entropy + {best_overall[0]:.1f}*{best_overall[2]}"
              f"  (ρ={best_overall[1]:+.4f})")

    return best_overall


# ═════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═════════════════════════════════════════════════════════════════════════════

def plot_results(data, groups, smoothed_map, selection_results, resampling_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    # ── Panel 1: Scatter — Smoothed Entropy vs Pass Rate (clean) ─────────
    raw_ent = np.array([d["metrics"]["prefix_entropy_mean"] for d in data])
    smoothed_ent = np.array([smoothed_map[id(d)] for d in data])
    pass_rates = np.array([d["pass_rate"] for d in data])

    # Bin the data for a cleaner visualization
    n_bins = 8
    bin_edges = np.linspace(raw_ent.min(), raw_ent.max(), n_bins + 1)
    bin_centers, bin_means, bin_stds = [], [], []
    for i in range(n_bins):
        mask = (raw_ent >= bin_edges[i]) & (raw_ent < bin_edges[i + 1])
        if mask.sum() > 2:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_means.append(pass_rates[mask].mean())
            bin_stds.append(pass_rates[mask].std() / np.sqrt(mask.sum()))

    ax1.scatter(raw_ent, pass_rates, c="#4878D0", alpha=0.15, s=12, zorder=1,
                edgecolors="none")
    ax1.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="o-",
                 color="#D65F5F", markersize=6, linewidth=2, capsize=3,
                 label="Binned Mean ± SE", zorder=3)

    # Trend line
    z = np.polyfit(raw_ent, pass_rates, 1)
    x_range = np.linspace(raw_ent.min(), raw_ent.max(), 100)
    ax1.plot(x_range, np.polyval(z, x_range), "#333333", linewidth=1.2,
             linestyle="--", alpha=0.6, zorder=2)

    rho, p = stats.spearmanr(raw_ent, pass_rates)
    ax1.set_xlabel("Mean Token Entropy", fontsize=11)
    ax1.set_ylabel("Prefix Pass Rate", fontsize=11)
    ax1.set_title("Entropy vs Pass Rate", fontsize=12, fontweight="bold")
    ax1.text(0.97, 0.97, f"ρ = {rho:.3f}\np = {p:.4f}",
             transform=ax1.transAxes, fontsize=9, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.tick_params(labelsize=9)

    # ── Panel 2: Grouped bar — Resampling Pass@k Lift ─────────────────────
    rs = resampling_results
    strats = ["Random prefix", "Raw Entropy (min)", "Smoothed Entropy (min)", "Oracle (best pass_rate)"]
    strat_labels = ["Random", "Raw Entropy", "Smoothed\nEntropy", "Oracle"]
    colors = ["#AAAAAA", "#4878D0", "#D65F5F", "#EE854A"]

    k_vals = [1, 4, 16]
    x = np.arange(len(k_vals))
    n_strats = len(strats)
    w = 0.18
    offset = np.arange(n_strats) * w - (n_strats - 1) * w / 2

    for i, (strat, label, c) in enumerate(zip(strats, strat_labels, colors)):
        vals = [rs[strat][f"pass@{k}"] * 100 for k in k_vals]
        bars = ax2.bar(x + offset[i], vals, w, label=label, color=c,
                       edgecolor="white", linewidth=0.5)
        # Add value labels on top of bars
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.0f}", ha="center", va="bottom", fontsize=6.5,
                     fontweight="bold", color=c)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"pass@{k}" for k in k_vals], fontsize=11)
    ax2.set_ylabel("Success Rate (%)", fontsize=11)
    ax2.set_title("Resampling Pass@k by Prefix Selection", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.9, ncol=2)
    ax2.grid(True, alpha=0.25, linestyle="--", axis="y")
    ax2.tick_params(labelsize=9)
    ax2.set_ylim(0, max(rs["Oracle (best pass_rate)"][f"pass@16"] * 100 + 10, 100))

    plt.tight_layout()
    out_pdf = OUT_DIR / "entropy_signal_analysis.pdf"
    out_png = OUT_DIR / "entropy_signal_analysis.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"\nPlots saved to {out_pdf} and {out_png}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)

    data = load_data()
    groups = group_by_problem(data)

    # A: Correlation analysis (raw vs smoothed)
    corr_results, smoothed_map = experiment_a_smoothed_correlation(data, groups)

    # B: Selection accuracy
    selection_results = experiment_b_selection_accuracy(groups, smoothed_map)

    # C: Resampling pass@k lift
    resampling_results = experiment_c_resampling_lift(groups, smoothed_map)

    # D: Hybrid signal search
    experiment_d_hybrid_search(data, groups)

    # Plot
    plot_results(data, groups, smoothed_map, selection_results, resampling_results)

    # Summary for paper
    print("\n" + "=" * 70)
    print("  SUMMARY FOR PAPER")
    print("=" * 70)
    raw_wp = corr_results["Prefix Entropy (Raw)"]["within_rho"]
    sm_wp = corr_results["Prefix Entropy (Smoothed)"]["within_rho"]
    print(f"  Raw entropy within-problem ρ:       {raw_wp:+.4f}")
    print(f"  Smoothed entropy within-problem ρ:  {sm_wp:+.4f}")
    improvement = (abs(sm_wp) - abs(raw_wp)) / abs(raw_wp) * 100
    print(f"  Smoothing improvement:              {improvement:+.1f}%")
    print()
    sel = selection_results
    print(f"  Top-25% selection accuracy:")
    print(f"    Random:   {sel['Random']['top_25']:.1%}")
    print(f"    Smoothed: {sel['Smoothed Entropy (min)']['top_25']:.1%}")
    print()
    rs = resampling_results
    for k in [1, 16]:
        rand_v = rs["Random prefix"][f"pass@{k}"]
        sm_v = rs["Smoothed Entropy (min)"][f"pass@{k}"]
        oracle_v = rs["Oracle (best pass_rate)"][f"pass@{k}"]
        lift = (sm_v - rand_v) / rand_v * 100
        print(f"  pass@{k}:  Random={rand_v:.4f}  Smoothed={sm_v:.4f}  "
              f"Oracle={oracle_v:.4f}  (Lift: {lift:+.1f}%)")


if __name__ == "__main__":
    main()
