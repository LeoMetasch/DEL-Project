"""Generate figures for DEL-Bench reports.

Self-contained: reads the results JSONL directly (no del_bench import needed).
Outputs vector PDF (for LaTeX \\includegraphics) + PNG preview for every figure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DEFAULT_RESULTS = Path("results/final_5provider_mixed_3seed.jsonl")
DEFAULT_OUT = Path("reports/plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DEL-Bench paper plots.")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"input JSONL results file (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output directory for PDF/PNG figures (default: {DEFAULT_OUT})",
    )
    return parser.parse_args()


ARGS = parse_args()
RESULTS = ARGS.results_file
OUT = ARGS.out_dir
if not RESULTS.exists():
    raise SystemExit(f"results file not found: {RESULTS}")
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Global style: clean, restrained, serif (Times-metric) to match a Springer doc
# ----------------------------------------------------------------------------
INK = "#1a1a1a"
GRID = "#d9d9d9"
MUTE = "#6b6b6b"
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 9,
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#4d4d4d",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "text.color": INK,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "legend.fontsize": 9,
        "legend.frameon": False,
    }
)

# Okabe-Ito colorblind-safe palette, fixed model -> colour mapping everywhere.
MODEL_NAME = {
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "qwen/qwen3-32b": "Qwen3 32B",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "mistralai/mistral-small-3.2-24b-instruct": "Mistral Small 3.2",
    "openai/gpt-4o-mini": "GPT-4o mini",
}
MODEL_COLOR = {
    "Gemini 2.5 Flash": "#0072B2",
    "Qwen3 32B": "#009E73",
    "Claude Haiku 4.5": "#D55E00",
    "Mistral Small 3.2": "#CC79A7",
    "GPT-4o mini": "#E69F00",
}
# Difficulty-axis colours for scenario figure
AXIS_COLOR = {
    "Observation": "#0072B2",
    "Deception": "#D55E00",
    "Trust": "#009E73",
    "Asymmetry": "#CC79A7",
    "Composite": "#8a8a8a",
}
SCEN = {  # short label, axis
    "coin_public_peek_v1": ("Public peek", "Observation"),
    "coin_private_peek_v1": ("Private peek", "Observation"),
    "coin_double_peek_v1": ("Double peek", "Observation"),
    "coin_truthful_announcement_v1": ("Truthful announcement", "Deception"),
    "coin_successful_lie_v1": ("Successful lie", "Deception"),
    "coin_lie_then_peek_v1": ("Lie then peek", "Deception"),
    "coin_unreliable_announcement_v1": ("Unreliable announcement", "Trust"),
    "coin_soft_announcement_v1": ("Soft announcement", "Trust"),
    "coin_conflicting_claims_v1": ("Conflicting claims", "Trust"),
    "coin_successful_lie_3agent_v1": ("Successful lie (3 agents)", "Asymmetry"),
    "coin_witness_deception_v1": ("Witness deception", "Asymmetry"),
    "coin_trusted_lie_v1": ("Trusted lie", "Composite"),
    "coin_peek_then_soft_v1": ("Peek then soft", "Composite"),
    "coin_private_peek_3agent_v1": ("Private peek (3 agents)", "Composite"),
    "coin_public_peek_tails_v1": ("Public peek, tails", "Composite"),
    "coin_peek_then_announce_v1": ("Peek then announce", "Composite"),
    "coin_moore_sentence_v1": ("Moore sentence", "Composite"),
    "coin_conditional_belief_v1": ("Conditional belief", "Composite"),
}
PROBE_SHORT = {
    "q6_bob_about_alice_about_bob": "Bob>Alice>Bob",
    "q5_bob_about_alice_about_bob": "Bob>Alice>Bob",
    "q6_bob_about_alice_about_bob_knows": "Bob>Alice>K(Bob)",
    "q8_depth4_bob_about_alice_about_bob_about_alice_knows": "Bob>Alice>Bob>K(Alice)",
    "q8_depth4_bob_about_alice_about_bob_about_alice": "Bob>Alice>Bob>Alice",
    "q11_charles_about_bob_about_charles_about_bob": "Chas>Bob>Chas>Bob",
    "q12_charles_about_alice_about_bob_about_alice": "Chas>Alice>Bob>Alice",
    "q6_depth4": "depth-4 chain",
    "q10_alice_about_bob_about_alice_about_charles": "Alice>Bob>Alice>Chas",
    "q2_alice_belief": "Alice belief",
    "q3_bob_belief": "Bob belief",
    "q5_alice_about_bob": "Alice>Bob",
    "q10_charles_about_bob_belief": "Chas>Bob",
    "q4_bob_believes_full_announcement": "Bob>full announcement",
    "q3_moore_punchline": "Moore punchline",
}


def wilson(num, den, z=1.96):
    if den == 0:
        return (np.nan, np.nan)
    p = num / den
    d = 1 + z * z / den
    c = (p + z * z / (2 * den)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * den)) / den) / d
    return c - m, c + m


def acc_ci(frame, groups):
    g = frame.groupby(groups, dropna=False).correct.agg(["sum", "count"]).reset_index()
    g["acc"] = g["sum"] / g["count"]
    lo, hi = zip(*[wilson(int(r["sum"]), int(r["count"])) for _, r in g.iterrows()])
    g["lo"], g["hi"] = lo, hi
    return g


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
rows = []
with RESULTS.open() as f:
    for line in f:
        t = json.loads(line)
        for p in t["probes"]:
            rows.append(
                {
                    "scenario": t["scenario_id"],
                    "model": MODEL_NAME.get(t["model"], t["model"]),
                    "strategy": t["strategy"],
                    "seed": t["seed"],
                    "probe_id": p["probe_id"],
                    "depth": p["depth"],
                    "correct": bool(p["correct"]),
                    "syntactic": bool(p["is_syntactic_control"]),
                    "gold": p["gold_label"],
                    "chosen": p["parsed_label"],
                }
            )
df = pd.DataFrame(rows)
lg = df[~df.syntactic].copy()
order = lg.groupby("model").correct.mean().sort_values(ascending=False).index.tolist()

# ============================================================================
# FIG 1 — Model overview: overall logical accuracy, direct vs CoT, Wilson CIs
# ============================================================================
ov = acc_ci(lg, ["model", "strategy"])
fig, ax = plt.subplots(figsize=(5.4, 3.3))
y = np.arange(len(order))[::-1]  # best at top
STRAT = {
    "direct": ("Direct", "#264653", "o"),
    "cot": ("Chain-of-thought", "#bb6b2c", "D"),
}
off = {"direct": 0.16, "cot": -0.16}
for s, (lab, col, mk) in STRAT.items():
    sub = ov[ov.strategy == s].set_index("model").loc[order].reset_index()
    yy = y + off[s]
    xerr = np.vstack([(sub.acc - sub.lo) * 100, (sub.hi - sub.acc) * 100])
    ax.errorbar(
        sub.acc * 100,
        yy,
        xerr=xerr,
        fmt=mk,
        ms=6,
        color=col,
        ecolor=col,
        elinewidth=1.3,
        capsize=2.5,
        mfc=col,
        mec="white",
        mew=0.6,
        label=lab,
        zorder=3,
    )
ax.set_yticks(y)
ax.set_yticklabels(order)
for spine_y in y[:-1]:
    ax.axhline(spine_y - 0.5, color=GRID, lw=0.6, zorder=0)
ax.set_xlim(60, 95)
ax.set_xlabel("Logical accuracy (%)")
ax.set_title("Overall higher-order belief tracking")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.legend(loc="lower right", handletextpad=0.4, borderaxespad=0.3)
save(fig, "fig1_model_overview")

# ============================================================================
# FIG 2 — HEADLINE: syntactic vs logical depth-3 (dumbbell)
# ============================================================================
d3 = df[df.depth == 3]
syn = acc_ci(d3[d3.syntactic], ["model"]).set_index("model")
log = acc_ci(d3[~d3.syntactic], ["model"]).set_index("model")
fig, ax = plt.subplots(figsize=(5.6, 3.3))
y = np.arange(len(order))[::-1]
for i, m in zip(y, order):
    lx, sx = log.loc[m, "acc"] * 100, syn.loc[m, "acc"] * 100
    ax.plot([lx, sx], [i, i], color="#bdbdbd", lw=2.2, zorder=1, solid_capstyle="round")
    ax.annotate(
        f"{sx-lx:+.0f} pp",
        ((lx + sx) / 2, i + 0.18),
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTE,
    )
ax.scatter(
    [log.loc[m, "acc"] * 100 for m in order],
    y,
    s=70,
    color="#c1432e",
    zorder=3,
    ec="white",
    lw=0.8,
    label="Logical depth 3",
)
ax.scatter(
    [syn.loc[m, "acc"] * 100 for m in order],
    y,
    s=70,
    color="#2a7f5e",
    zorder=3,
    ec="white",
    lw=0.8,
    label="Syntactic depth 3 (control)",
)
ax.set_yticks(y)
ax.set_yticklabels(order)
ax.set_xlim(35, 105)
ax.set_xlabel("Accuracy (%)")
ax.set_title("Syntactic depth is not logical depth")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.5)
save(fig, "fig2_syntactic_vs_logical")

# ============================================================================
# FIG 3 — Depth degradation: aggregate band + per-model thin lines
# ============================================================================
depths = sorted(lg.depth.unique())
agg = acc_ci(lg, ["depth"]).sort_values("depth")
fig, ax = plt.subplots(figsize=(5.6, 3.5))
for m in order:
    sub = lg[lg.model == m].groupby("depth").correct.mean().reindex(depths)
    ax.plot(
        depths,
        sub * 100,
        color=MODEL_COLOR[m],
        lw=1.1,
        alpha=0.55,
        zorder=2,
        marker="o",
        ms=3,
        mec="none",
    )
ax.fill_between(
    agg.depth, agg.lo * 100, agg.hi * 100, color="#333333", alpha=0.12, zorder=1
)
ax.plot(
    agg.depth,
    agg.acc * 100,
    color="#1a1a1a",
    lw=2.6,
    marker="o",
    ms=6,
    mfc="#1a1a1a",
    mec="white",
    mew=0.8,
    zorder=4,
    label="Pooled (95% CI)",
)
ax.axvspan(2.5, 3.5, color="#c1432e", alpha=0.05, zorder=0)
ax.annotate(
    "hardest probes\nconcentrated here",
    (3.0, 47),
    ha="center",
    fontsize=8,
    color="#c1432e",
)
ax.annotate(
    "depth-4 mix draws\nfrom easier scenarios",
    (3.55, 86),
    ha="center",
    fontsize=7.5,
    color=MUTE,
)
ax.set_xticks(depths)
ax.set_xlabel("Modal nesting depth")
ax.set_ylabel("Logical accuracy (%)")
ax.set_ylim(38, 103)
ax.set_title("Accuracy degrades with belief nesting")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
handles = [Line2D([], [], color="#1a1a1a", lw=2.6, marker="o", label="Pooled (95% CI)")]
handles += [Line2D([], [], color=MODEL_COLOR[m], lw=1.4, label=m) for m in order]
ax.legend(
    handles=handles,
    loc="lower left",
    ncol=1,
    handletextpad=0.5,
    labelspacing=0.25,
    borderaxespad=0.3,
)
save(fig, "fig3_depth_profile")

# ============================================================================
# FIG 4 — Scenario difficulty, coloured by difficulty axis
# ============================================================================
sc = acc_ci(lg, ["scenario"]).copy()
sc["label"] = sc.scenario.map(lambda s: SCEN[s][0])
sc["axis"] = sc.scenario.map(lambda s: SCEN[s][1])
sc = sc.sort_values("acc")
fig, ax = plt.subplots(figsize=(5.8, 5.2))
y = np.arange(len(sc))
cols = [AXIS_COLOR[a] for a in sc.axis]
xerr = np.vstack([(sc.acc - sc.lo) * 100, (sc.hi - sc.acc) * 100])
ax.barh(y, sc.acc * 100, color=cols, edgecolor="white", height=0.72, zorder=2)
ax.errorbar(
    sc.acc * 100,
    y,
    xerr=xerr,
    fmt="none",
    ecolor="#3a3a3a",
    elinewidth=0.9,
    capsize=2,
    zorder=3,
)
overall = lg.correct.mean() * 100
ax.axvline(overall, color="#1a1a1a", ls=(0, (5, 2)), lw=1.6, zorder=5)
ax.text(
    overall,
    len(sc) - 0.30,
    f"pooled {overall:.0f}%",
    fontsize=8.5,
    ha="center",
    va="bottom",
    color="#1a1a1a",
    zorder=6,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1a1a1a", lw=0.7),
)
for yy, (a, h) in zip(y, zip(sc.acc * 100, sc.hi * 100)):
    ax.text(h + 1.2, yy, f"{a:.0f}", va="center", fontsize=8, color=INK)
ax.set_yticks(y)
ax.set_yticklabels(sc.label)
ax.set_xlim(0, 112)
ax.set_ylim(-0.7, len(sc) - 1 + 1.05)
ax.set_xlabel("Logical accuracy (%)")
ax.set_title("Scenario difficulty by epistemic structure")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
leg = [
    Patch(facecolor=AXIS_COLOR[a], label=a)
    for a in ["Observation", "Deception", "Trust", "Asymmetry", "Composite"]
]
ax.legend(
    handles=leg,
    loc="lower right",
    title="Difficulty axis",
    title_fontsize=9,
    ncol=1,
    handlelength=1.2,
)
save(fig, "fig4_scenario_difficulty")

# ============================================================================
# FIG 5 — Model x depth heatmap
# ============================================================================
piv = (lg.groupby(["model", "depth"]).correct.mean().unstack() * 100).loc[order, depths]
fig, ax = plt.subplots(figsize=(5.4, 3.2))
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    " delb", ["#7a1f1f", "#c1432e", "#e8b04b", "#7cae8f", "#2a7f5e"]
)
im = ax.imshow(piv.values, cmap=cmap, vmin=40, vmax=100, aspect="auto")
ax.set_xticks(range(len(depths)))
ax.set_xticklabels(depths)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.set_xlabel("Modal nesting depth")
ax.set_title("Per-model accuracy by depth")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(
            j,
            i,
            f"{v:.0f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if (v < 62 or v > 92) else "#1a1a1a",
            fontweight="bold",
        )
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("Accuracy (%)", fontsize=9)
cb.ax.tick_params(labelsize=8)
ax.set_xticks(np.arange(-0.5, len(depths), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
ax.grid(which="minor", color="white", lw=1.5)
ax.tick_params(which="minor", length=0)
save(fig, "fig5_model_depth_heatmap")

# ============================================================================
# FIG 6 — Robustly hardest individual probes (pooled over all runs)
# ============================================================================
pr = acc_ci(lg, ["scenario", "probe_id", "depth"]).copy()
pr["label"] = (
    pr.scenario.map(lambda s: SCEN[s][0])
    + "  ·  "
    + pr.probe_id.map(lambda p: PROBE_SHORT.get(p, p))
    + " (d"
    + pr.depth.astype(str)
    + ")"
)
pr = pr.sort_values("acc").head(12)
fig, ax = plt.subplots(figsize=(6.2, 4.4))
y = np.arange(len(pr))[::-1]
dcol = {0: "#6c8ebf", 1: "#7aa974", 2: "#e0a23c", 3: "#c1432e", 4: "#8b6bb1"}
cols = [dcol[int(d)] for d in pr.depth]
ax.barh(y, pr.acc * 100, color=cols, edgecolor="white", height=0.74, zorder=2)
for yy, a in zip(y, pr.acc * 100):
    ax.text(a + 0.8, yy, f"{a:.0f}", va="center", fontsize=8, color=INK)
ax.set_yticks(y)
ax.set_yticklabels(pr.label, fontsize=8.5)
ax.set_xlim(0, 72)
ax.set_xlabel("Accuracy across all model / strategy / seed runs (%)")
ax.set_title("Robustly hard probes")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
leg = [
    Patch(facecolor=dcol[d], label=f"depth {d}")
    for d in sorted(dcol)
    if d in pr.depth.values
]
ax.legend(handles=leg, loc="lower right", ncol=1, handlelength=1.2)
save(fig, "fig6_hardest_probes")

# ============================================================================
# FIG 7 — Matched-pair fact-matching diagnostic (stacked composition)
# For each diagnostic pair, on shared probes whose gold differs between the two
# members, each (model,strategy,seed,probe) instance is classified as:
#   both_correct | same answer to both (fact-matching failure) | other error.
# ============================================================================
PAIRS = [
    ("coin_public_peek_v1", "coin_private_peek_v1"),
    ("coin_truthful_announcement_v1", "coin_successful_lie_v1"),
    ("coin_unreliable_announcement_v1", "coin_soft_announcement_v1"),
    ("coin_successful_lie_3agent_v1", "coin_witness_deception_v1"),
]
recs = []
for a, b in PAIRS:
    pa = lg[lg.scenario == a]
    pb = lg[lg.scenario == b]
    shared = set(pa.probe_id) & set(pb.probe_id)
    diff = [
        pid
        for pid in shared
        if pa[pa.probe_id == pid].gold.iloc[0] != pb[pb.probe_id == pid].gold.iloc[0]
    ]
    ma = pa[pa.probe_id.isin(diff)][
        ["model", "strategy", "seed", "probe_id", "correct", "chosen"]
    ]
    mb = pb[pb.probe_id.isin(diff)][
        ["model", "strategy", "seed", "probe_id", "correct", "chosen"]
    ]
    m = ma.merge(
        mb, on=["model", "strategy", "seed", "probe_id"], suffixes=("_a", "_b")
    )
    for _, r in m.iterrows():
        if r.correct_a and r.correct_b:
            cat = "both_correct"
        elif r.chosen_a == r.chosen_b:
            cat = "fact_match"
        else:
            cat = "other"
        recs.append(dict(model=r.model, cat=cat))
mp = pd.DataFrame(recs)
comp = mp.groupby(["model", "cat"]).size().unstack(fill_value=0)
comp = comp.div(comp.sum(axis=1), axis=0).loc[order]
CATS = [
    ("both_correct", "Tracks structure (both correct)", "#2a7f5e"),
    ("fact_match", "Fact-matching failure (same answer to both)", "#c1432e"),
    ("other", "Other error", "#bdbdbd"),
]
fig, ax = plt.subplots(figsize=(6.6, 3.3))
y = np.arange(len(order))[::-1]
left = np.zeros(len(order))
for key, lab, col in CATS:
    vals = comp[key].values * 100
    ax.barh(y, vals, left=left, color=col, edgecolor="white", height=0.66, label=lab)
    for yy, v, l in zip(y, vals, left):
        if v > 6:
            ax.text(
                l + v / 2,
                yy,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if col != "#bdbdbd" else INK,
                fontweight="bold",
            )
    left += vals
ax.set_yticks(y)
ax.set_yticklabels(order)
ax.set_xlim(0, 100)
ax.set_xlabel("Share of matched-pair instances (%)")
ax.set_title("Matched-pair sensitivity: structure vs. fact-matching")
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=1,
    handlelength=1.2,
    labelspacing=0.3,
)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
save(fig, "fig7_matched_pair_diagnostic")

# ============================================================================
# FIG 8 — Paraphrase robustness across the three narrative seeds
# ============================================================================
seed_acc = lg.groupby(["model", "seed"]).correct.mean().unstack() * 100
seed_acc = seed_acc.loc[order]
fig, ax = plt.subplots(figsize=(5.6, 3.1))
y = np.arange(len(order))[::-1]
for i, m in zip(y, order):
    vals = seed_acc.loc[m].values
    ax.plot(
        [vals.min(), vals.max()],
        [i, i],
        color="#cfcfcf",
        lw=3,
        solid_capstyle="round",
        zorder=1,
    )
    ax.scatter(
        vals, [i] * len(vals), color=MODEL_COLOR[m], s=42, zorder=3, ec="white", lw=0.7
    )
    ax.scatter(vals.mean(), i, marker="|", color="#1a1a1a", s=260, zorder=4, lw=1.6)
    ax.text(
        vals.max() + 0.8,
        i,
        f"\u0394={vals.max()-vals.min():.1f}",
        va="center",
        fontsize=8,
        color=MUTE,
    )
ax.set_yticks(y)
ax.set_yticklabels(order)
ax.set_xlim(63, 95)
ax.set_xlabel("Logical accuracy per narrative seed (%)")
ax.set_title("Robustness across paraphrase seeds")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
hd = [
    plt.Line2D(
        [], [], marker="o", ls="", color="#888", mec="white", label="Seed 0 / 1 / 2"
    ),
    plt.Line2D([], [], marker="|", ls="", color="#1a1a1a", ms=12, label="Mean"),
]
ax.legend(handles=hd, loc="lower right", handletextpad=0.4)
save(fig, "fig8_paraphrase_robustness")


# ============================================================================
# FIG 9 — Error-mode decomposition (heuristic) among logical errors
# ============================================================================
def norm_ht(s):
    return (
        str(s)
        .lower()
        .replace("heads", "@")
        .replace("tails", "@")
        .replace("h)", "@")
        .replace("t)", "@")
    )


NB = ("no belief either way",)


def classify(gold, chosen):
    g, c = str(gold).lower(), str(chosen).lower()
    g_nb = any(k in g for k in NB)
    c_nb = any(k in c for k in NB)
    if g_nb != c_nb:
        return "No-belief confusion"
    if norm_ht(gold) == norm_ht(chosen) and gold != chosen:
        return "Belief polarity flip"
    return "Other structural error"


err = lg[~lg.correct].copy()
err["mode"] = [classify(g, c) for g, c in zip(err.gold, err.chosen)]
emc = err.groupby(["model", "mode"]).size().unstack(fill_value=0)
emc = emc.div(emc.sum(axis=1), axis=0).loc[order]
MODES = [
    ("Belief polarity flip", "#c45a2d"),
    ("No-belief confusion", "#e0a23c"),
    ("Other structural error", "#7d8aa0"),
]
fig, ax = plt.subplots(figsize=(6.6, 3.1))
y = np.arange(len(order))[::-1]
left = np.zeros(len(order))
for mode, col in MODES:
    vals = (emc[mode].values if mode in emc else np.zeros(len(order))) * 100
    ax.barh(y, vals, left=left, color=col, edgecolor="white", height=0.66, label=mode)
    for yy, v, l in zip(y, vals, left):
        if v > 7:
            ax.text(
                l + v / 2,
                yy,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white",
                fontweight="bold",
            )
    left += vals
ax.set_yticks(y)
ax.set_yticklabels(order)
ax.set_xlim(0, 100)
ax.set_xlabel("Share of logical errors (%)")
ax.set_title("How models fail: error-mode decomposition")
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=3,
    handlelength=1.2,
    columnspacing=1.0,
)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
save(fig, "fig9_error_modes")


print("Saved figures to", OUT)
for p in sorted(OUT.glob("*.pdf")):
    print(" ", p.name)
