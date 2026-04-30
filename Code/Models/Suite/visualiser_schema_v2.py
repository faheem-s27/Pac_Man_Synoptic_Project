"""
Suite Training Visualiser (Schema V2)
====================================
Generates five dashboard PNGs for train_suite schema-v2 CSV files.

Themes:
1) training_progression
2) outcome_analysis
3) fixed_vs_random
4) test_performance
5) efficiency

Usage:
    python visualiser_schema_v2.py
    python visualiser_schema_v2.py --file path/to/train_suite_*.csv
    python visualiser_schema_v2.py --show
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator


# -- Theme -------------------------------------------------------------------
plt.style.use("dark_background")
BG = "#0A0A0A"
PANEL_BG = "#111111"
GRID = "#2A2A2A"
TXT = "#BBBBBB"
DQN_COL = "#4EA3FF"
NEAT_COL = "#C678DD"
FIXED_COL = "#FFB347"
RANDOM_COL = "#00D4FF"
WIN_COL = "#00FF88"
GHOST_COL = "#FF6B6B"
STARVE_COL = "#FFD166"
MAX_COL = "#9B5DE5"
OTHER_COL = "#888888"

ROLL_WIN = 100

# -- Paths -------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(_HERE, "CSV_History_SchemaV2")
OUT_DIR = os.path.join(_HERE, "Visualiser_Output_V2")


def _style_ax(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, color="white", pad=6)
    ax.set_xlabel(xlabel, fontsize=9, color=TXT)
    ax.set_ylabel(ylabel, fontsize=9, color=TXT)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID, alpha=0.25, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def _roll(s: pd.Series, w: int = ROLL_WIN) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()


def _to_bool(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["1", "true", "t", "yes", "y"])


def _latest_csv() -> str:
    files = glob.glob(os.path.join(CSV_DIR, "train_suite_*.csv"))
    if not files:
        raise FileNotFoundError(f"No train_suite_*.csv found in {CSV_DIR}")
    return max(files, key=os.path.getmtime)


def _short_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].replace("train_suite_", "")


def _seed_regime_group(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("fixed"):
        return "fixed"
    return "random"


def _outcome_group(value: str) -> str:
    text = str(value).strip().upper()
    if text == "WIN":
        return "WIN"
    if "STARV" in text:
        return "STARVATION"
    if "MAX" in text:
        return "MAX_STEPS"
    if "GHOST" in text or text in {"BLINKY", "PINKY", "INKY", "CLYDE"}:
        return "GHOST"
    if text in {"NONE", ""}:
        return "OTHER"
    return "OTHER"


def _add_stage_aware_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Build a stage-normalized x-axis so each stage gets equal visual width."""
    out = df.copy()
    out["_StageAwareX"] = np.nan

    required = {"Algorithm", "Seed_Regime_Group", "Is_Test", "Stage", "Episode"}
    if not required.issubset(out.columns):
        return out

    work = out.sort_values(["Algorithm", "Seed_Regime_Group", "Is_Test", "Episode"]).copy()
    grp_cols = ["Algorithm", "Seed_Regime_Group", "Is_Test", "Stage"]
    within = work.groupby(grp_cols).cumcount() + 1
    size = work.groupby(grp_cols)["Episode"].transform("size").clip(lower=1)

    # Value range per stage is [stage, stage+1), independent of episode count at that stage.
    work["_StageAwareX"] = work["Stage"].astype(float) + (within - 0.5) / size
    out.loc[work.index, "_StageAwareX"] = work["_StageAwareX"]
    return out


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "Episode", "Stage", "Maze_Seed", "Reward", "Macro_Steps", "Micro_Ticks",
        "Win", "Epsilon", "Pellets", "Power_Pellets", "Ghosts", "Explore_Rate", "Avg_Loss",
        "Generation", "Best_Fitness", "Avg_Fitness", "Species_Count", "Eval_Seeds_Per_Genome",
        "Max_Episode_Steps", "Episode_Duration_Sec", "Pipeline_Elapsed_Sec", "Test_Run_Elapsed_Sec",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Algorithm" in df.columns:
        df["Algorithm"] = df["Algorithm"].astype(str).str.strip().str.upper()
    if "Outcome" in df.columns:
        df["Outcome"] = df["Outcome"].astype(str).str.strip().str.upper()
    else:
        df["Outcome"] = "OTHER"

    if "Is_Test" in df.columns:
        df["Is_Test"] = _to_bool(df["Is_Test"])
    else:
        df["Is_Test"] = False

    if "Seed_Regime" not in df.columns:
        df["Seed_Regime"] = "random"
    df["Seed_Regime"] = df["Seed_Regime"].astype(str)
    df["Seed_Regime_Group"] = df["Seed_Regime"].map(_seed_regime_group)

    if "Test_Mode" not in df.columns:
        df["Test_Mode"] = "unknown"
    df["Test_Mode"] = df["Test_Mode"].astype(str)

    df["Outcome_Group"] = df["Outcome"].map(_outcome_group)
    df["Win"] = df.get("Win", 0).fillna(0).astype(int)

    df = _add_stage_aware_axis(df)
    return df


def _plot_stage_transition_lines(ax, x: pd.Series, stage: pd.Series, color: str) -> None:
    if len(x) <= 1:
        return
    stage_change = stage.ne(stage.shift(1)).fillna(False)
    change_x = x[stage_change]
    for xv in change_x.iloc[1:]:
        ax.axvline(float(xv), color=color, alpha=0.12, linewidth=0.8)


def fig_training_progression(df: pd.DataFrame, label: str) -> plt.Figure:
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Training Progression Dashboard - {label}", fontsize=16, color="white", y=0.99)

    train = df[df["Is_Test"] == False].copy()
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(
        ax1,
        f"Rolling Win Rate (window={ROLL_WIN}) + Stage Transitions",
        "Curriculum Stage (normalized within stage)",
        "Win Rate",
    )
    ax1.yaxis.set_major_formatter(PercentFormatter(100))
    for algo in ["DQN", "NEAT"]:
        algo_col = DQN_COL if algo == "DQN" else NEAT_COL
        for regime in ["random", "fixed"]:
            d = train[(train["Algorithm"] == algo) & (train["Seed_Regime_Group"] == regime)].sort_values("Episode")
            if d.empty:
                continue
            xvals = d["_StageAwareX"] if "_StageAwareX" in d.columns else d["Episode"]
            wr = _roll(d["Win"].astype(float)) * 100
            ls = "-" if regime == "random" else "--"
            ax1.plot(xvals, wr, color=algo_col, linestyle=ls, linewidth=1.6,
                     label=f"{algo}-{regime}")
            _plot_stage_transition_lines(ax1, xvals, d["Stage"], algo_col)
    if "Stage" in train.columns:
        smax = int(train["Stage"].max()) if len(train) else 0
        ax1.set_xticks(np.arange(0, smax + 1, 1))
    ax1.legend(fontsize=8, ncol=2)

    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, f"DQN Epsilon vs Rolling Win Rate (window={ROLL_WIN})", "Curriculum Stage (normalized)", "Epsilon")
    ax2r = ax2.twinx()
    ax2r.set_ylabel("Win Rate", color=TXT, fontsize=9)
    ax2r.tick_params(colors=TXT, labelsize=8)
    ax2r.yaxis.set_major_formatter(PercentFormatter(100))
    dqn_train = train[train["Algorithm"] == "DQN"].sort_values("Episode")
    if not dqn_train.empty:
        x_col = "_StageAwareX" if "_StageAwareX" in dqn_train.columns else "Episode"
        merged = dqn_train.groupby(x_col, as_index=True).agg(Epsilon=("Epsilon", "mean"), Win=("Win", "mean"))
        x = merged.index.values
        ax2.plot(x, merged["Epsilon"].values, color=FIXED_COL, linewidth=1.7, label="Epsilon")
        ax2r.plot(x, _roll(merged["Win"]).values * 100, color=DQN_COL, linewidth=1.7, label="Win Rate")
    if "Stage" in train.columns:
        smax = int(train["Stage"].max()) if len(train) else 0
        ax2.set_xticks(np.arange(0, smax + 1, 1))
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, f"DQN Avg Loss (rolling={ROLL_WIN})", "Episode", "Loss")
    for regime in ["random", "fixed"]:
        d = dqn_train[dqn_train["Seed_Regime_Group"] == regime]
        d = d[d["Avg_Loss"] > 0]
        if d.empty:
            continue
        x_col = "_StageAwareX" if "_StageAwareX" in d.columns else "Episode"
        grouped = d.groupby(x_col, as_index=False)["Avg_Loss"].mean()
        ax3.plot(grouped[x_col], _roll(grouped["Avg_Loss"]),
                 color=(RANDOM_COL if regime == "random" else FIXED_COL),
                 linewidth=1.8, label=f"{regime}")
    ax3.legend(fontsize=8)
    if "Stage" in train.columns:
        smax = int(train["Stage"].max()) if len(train) else 0
        ax3.set_xticks(np.arange(0, smax + 1, 1))

    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "NEAT Best vs Avg Fitness by Generation", "Generation", "Fitness")
    neat_train = train[train["Algorithm"] == "NEAT"].copy()
    for regime in ["random", "fixed"]:
        d = neat_train[neat_train["Seed_Regime_Group"] == regime]
        if d.empty:
            continue
        by_gen = d.groupby("Generation", as_index=False).agg(
            Best_Fitness=("Best_Fitness", "max"),
            Avg_Fitness=("Avg_Fitness", "mean"),
        ).sort_values("Generation")
        col = RANDOM_COL if regime == "random" else FIXED_COL
        ax4.plot(by_gen["Generation"], by_gen["Best_Fitness"], color=col, linewidth=2.0,
                 label=f"Best ({regime})")
        ax4.plot(by_gen["Generation"], by_gen["Avg_Fitness"], color=col, linewidth=1.3,
                 linestyle="--", label=f"Avg ({regime})")
    ax4.legend(fontsize=8, ncol=2)

    ax5 = fig.add_subplot(gs[2, 0])
    _style_ax(ax5, "NEAT Species Count by Generation", "Generation", "Species")
    for regime in ["random", "fixed"]:
        d = neat_train[neat_train["Seed_Regime_Group"] == regime]
        if d.empty:
            continue
        species = d.groupby("Generation", as_index=False)["Species_Count"].max().sort_values("Generation")
        col = RANDOM_COL if regime == "random" else FIXED_COL
        ax5.step(species["Generation"], species["Species_Count"], where="post",
                 color=col, linewidth=1.8, label=regime)
    ax5.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[2, 1])
    _style_ax(ax6, "Curriculum Stage Progression (DQN vs NEAT)", "Curriculum Stage (normalized)", "Stage")
    for algo in ["DQN", "NEAT"]:
        d = train[train["Algorithm"] == algo]
        if d.empty:
            continue
        x_col = "_StageAwareX" if "_StageAwareX" in d.columns else "Episode"
        grouped = d.groupby(x_col, as_index=False)["Stage"].mean().sort_values(x_col)
        ax6.step(grouped[x_col], grouped["Stage"], where="post",
                 color=(DQN_COL if algo == "DQN" else NEAT_COL), linewidth=2.0, label=algo)
    ax6.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax6.legend(fontsize=8)
    if "Stage" in train.columns:
        smax = int(train["Stage"].max()) if len(train) else 0
        ax6.set_xticks(np.arange(0, smax + 1, 1))

    ax7 = fig.add_subplot(gs[3, :])
    _style_ax(ax7, "Ghost Interaction Signals Over Training", "Curriculum Stage (normalized)", "Rolling Mean")
    for algo in ["DQN", "NEAT"]:
        d = train[train["Algorithm"] == algo].sort_values("Episode")
        if d.empty:
            continue
        x_col = "_StageAwareX" if "_StageAwareX" in d.columns else "Episode"
        g = d.groupby(x_col, as_index=False).agg(
            Ghosts=("Ghosts", "mean"),
            Power_Pellets=("Power_Pellets", "mean"),
        )
        col = DQN_COL if algo == "DQN" else NEAT_COL
        ax7.plot(g[x_col], _roll(g["Ghosts"]), color=col, linewidth=1.8,
                 label=f"{algo} ghosts eaten")
        ax7.plot(g[x_col], _roll(g["Power_Pellets"]), color=col, linewidth=1.8,
                 linestyle="--", label=f"{algo} power pellets")
    ax7.legend(fontsize=8, ncol=2)
    if "Stage" in train.columns:
        smax = int(train["Stage"].max()) if len(train) else 0
        ax7.set_xticks(np.arange(0, smax + 1, 1))

    return fig


def _stacked_stage_outcomes(ax, stage_df: pd.DataFrame, title: str) -> None:
    _style_ax(ax, title, "Stage", "Proportion")
    if stage_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=TXT, transform=ax.transAxes)
        return

    order = ["WIN", "GHOST", "STARVATION", "MAX_STEPS", "OTHER"]
    colors = {
        "WIN": WIN_COL,
        "GHOST": GHOST_COL,
        "STARVATION": STARVE_COL,
        "MAX_STEPS": MAX_COL,
        "OTHER": OTHER_COL,
    }
    counts = stage_df.groupby(["Stage", "Outcome_Group"]).size().unstack(fill_value=0)
    for cat in order:
        if cat not in counts.columns:
            counts[cat] = 0
    counts = counts[order].sort_index()

    x = counts.index.values
    totals = counts.sum(axis=1).replace(0, np.nan)
    bottom = np.zeros(len(counts))
    for cat in order:
        vals = (counts[cat] / totals).fillna(0).values
        ax.bar(x, vals, bottom=bottom, color=colors[cat], edgecolor=BG, linewidth=0.8, label=cat)
        bottom += vals

    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def fig_outcome_analysis(df: pd.DataFrame, label: str) -> plt.Figure:
    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Outcome Analysis Dashboard - {label}", fontsize=16, color="white", y=0.99)

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    train = df[df["Is_Test"] == False]
    test = df[df["Is_Test"] == True]

    ax1 = fig.add_subplot(gs[0, 0])
    _stacked_stage_outcomes(ax1, train[train["Algorithm"] == "DQN"], "Training Outcome Distribution by Stage (DQN)")
    ax1.legend(fontsize=8, ncol=3)

    ax2 = fig.add_subplot(gs[0, 1])
    _stacked_stage_outcomes(ax2, train[train["Algorithm"] == "NEAT"], "Training Outcome Distribution by Stage (NEAT)")
    ax2.legend(fontsize=8, ncol=3)

    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "Test Outcome Distribution (DQN vs NEAT)", "Outcome", "Count")
    order = ["WIN", "GHOST", "STARVATION", "MAX_STEPS", "OTHER"]
    x = np.arange(len(order))
    width = 0.38
    for i, algo in enumerate(["DQN", "NEAT"]):
        d = test[test["Algorithm"] == algo]
        counts = d["Outcome_Group"].value_counts()
        vals = np.array([int(counts.get(o, 0)) for o in order])
        ax3.bar(x + (i * width), vals, width=width,
                color=(DQN_COL if algo == "DQN" else NEAT_COL), alpha=0.9, label=algo)
    ax3.set_xticks(x + width / 2)
    ax3.set_xticklabels(order, rotation=15)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "Outcome Cause by Regime (Test)", "Outcome", "Proportion")
    regimes = ["random", "fixed"]
    bar_w = 0.35
    for ridx, regime in enumerate(regimes):
        d = test[test["Seed_Regime_Group"] == regime]
        counts = d["Outcome_Group"].value_counts()
        totals = max(1, int(counts.sum()))
        vals = np.array([float(counts.get(o, 0)) / totals for o in order])
        ax4.bar(np.arange(len(order)) + ridx * bar_w, vals, width=bar_w,
                color=(RANDOM_COL if regime == "random" else FIXED_COL), alpha=0.9, label=regime)
    ax4.set_xticks(np.arange(len(order)) + bar_w / 2)
    ax4.set_xticklabels(order, rotation=15)
    ax4.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax4.legend(fontsize=8)

    return fig


def _metric_by_algo_regime(test_df: pd.DataFrame, metric: str, use_reached_stage: bool = True) -> pd.DataFrame:
    d = test_df.copy()
    if use_reached_stage and "Test_Mode" in d.columns:
        subset = d[d["Test_Mode"].str.lower() == "reached_stage"]
        if not subset.empty:
            d = subset
    out = d.groupby(["Algorithm", "Seed_Regime_Group"], as_index=False)[metric].mean()
    return out


def fig_fixed_vs_random(df: pd.DataFrame, label: str) -> plt.Figure:
    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Fixed vs Random Seed Comparison - {label}", fontsize=16, color="white", y=0.99)

    test = df[df["Is_Test"] == True].copy()
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    groups = [("DQN", "fixed"), ("DQN", "random"), ("NEAT", "fixed"), ("NEAT", "random")]
    labels = [f"{a}-{r}" for a, r in groups]

    def _vals(metric: str, scale: float = 1.0) -> list[float]:
        m = _metric_by_algo_regime(test, metric, use_reached_stage=True)
        vals = []
        for algo, regime in groups:
            row = m[(m["Algorithm"] == algo) & (m["Seed_Regime_Group"] == regime)]
            vals.append(float(row[metric].iloc[0] * scale) if not row.empty else 0.0)
        return vals

    cols = [FIXED_COL, RANDOM_COL, FIXED_COL, RANDOM_COL]

    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "Win Rate (%) on Reached-Stage Test", "Group", "Win Rate")
    vals = _vals("Win", scale=100.0)
    bars = ax1.bar(labels, vals, color=cols, edgecolor=BG, linewidth=1.0)
    ax1.yaxis.set_major_formatter(PercentFormatter(100))
    ax1.set_ylim(0, max(10.0, max(vals) * 1.2 if vals else 10.0))
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "Average Reward on Reached-Stage Test", "Group", "Reward")
    vals = _vals("Reward")
    ax2.bar(labels, vals, color=cols, edgecolor=BG, linewidth=1.0)

    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "Average Pellets Collected on Reached-Stage Test", "Group", "Pellets")
    vals = _vals("Pellets")
    ax3.bar(labels, vals, color=cols, edgecolor=BG, linewidth=1.0)

    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "Sample Count per Group (Reached-Stage Test)", "Group", "Episodes")
    reached = test[test["Test_Mode"].str.lower() == "reached_stage"]
    cnt = reached.groupby(["Algorithm", "Seed_Regime_Group"]).size()
    vals = [int(cnt.get((a, r), 0)) for a, r in groups]
    ax4.bar(labels, vals, color=cols, edgecolor=BG, linewidth=1.0)

    return fig


def fig_test_performance(df: pd.DataFrame, label: str) -> plt.Figure:
    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Test Performance Deep Dive - {label}", fontsize=16, color="white", y=0.99)

    test = df[df["Is_Test"] == True].copy()
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    def _boxplot(ax, metric: str, title: str, scale: float = 1.0) -> None:
        _style_ax(ax, title, "Algorithm", metric)
        data = []
        labels = []
        colors = []
        for algo, col in [("DQN", DQN_COL), ("NEAT", NEAT_COL)]:
            d = test[test["Algorithm"] == algo][metric].dropna()
            if d.empty:
                continue
            data.append(d.values * scale)
            labels.append(algo)
            colors.append(col)
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=TXT, transform=ax.transAxes)
            return
        bp = ax.boxplot(
            data,
            patch_artist=True,
            labels=labels,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=TXT),
            capprops=dict(color=TXT),
            flierprops=dict(marker=".", color="#555555", markersize=2),
        )
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.8)

    ax1 = fig.add_subplot(gs[0, 0])
    _boxplot(ax1, "Reward", "Reward Distribution (Test)")

    ax2 = fig.add_subplot(gs[0, 1])
    _boxplot(ax2, "Pellets", "Pellets Collected Distribution (Test)")

    ax3 = fig.add_subplot(gs[1, 0])
    _boxplot(ax3, "Explore_Rate", "Explore Rate Distribution (Test)", scale=100.0)
    ax3.yaxis.set_major_formatter(PercentFormatter(100))

    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "Ghosts Eaten Distribution (Test)", "Algorithm", "Ghosts")
    _boxplot(ax4, "Ghosts", "Ghosts Eaten Distribution (Test)")

    return fig


def _episodes_to_stage(train_df: pd.DataFrame, algo: str, regime: str, stages: list[int]) -> list[float]:
    d = train_df[(train_df["Algorithm"] == algo) & (train_df["Seed_Regime_Group"] == regime)].sort_values("Episode")
    if d.empty:
        return [np.nan for _ in stages]
    out = []
    for stg in stages:
        hit = d[d["Stage"] >= stg]
        out.append(float(hit["Episode"].iloc[0]) if not hit.empty else np.nan)
    return out


def fig_efficiency(df: pd.DataFrame, label: str) -> plt.Figure:
    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Efficiency Dashboard - {label}", fontsize=16, color="white", y=0.99)

    train = df[df["Is_Test"] == False].copy()
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[:, 0])
    _style_ax(ax1, "Episodes to Reach Each Curriculum Stage", "Episodes", "Stage")
    max_stage = int(train["Stage"].max()) if not train.empty else 0
    stages = list(range(max_stage + 1))
    y = np.arange(len(stages))
    h = 0.18
    profiles = [
        ("DQN", "random", DQN_COL),
        ("DQN", "fixed", FIXED_COL),
        ("NEAT", "random", NEAT_COL),
        ("NEAT", "fixed", STARVE_COL),
    ]
    for i, (algo, regime, col) in enumerate(profiles):
        vals = _episodes_to_stage(train, algo, regime, stages)
        yy = y + (i - 1.5) * h
        ax1.barh(yy, vals, height=h, color=col, alpha=0.9, label=f"{algo}-{regime}")
    ax1.set_yticks(y)
    ax1.set_yticklabels(stages)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "Wall-Clock Time per Episode (Train)", "Group", "Seconds")
    grp = train.groupby(["Algorithm", "Seed_Regime_Group"], as_index=False)["Episode_Duration_Sec"].mean()
    labels = []
    vals = []
    cols = []
    for algo, regime, col in profiles:
        row = grp[(grp["Algorithm"] == algo) & (grp["Seed_Regime_Group"] == regime)]
        labels.append(f"{algo}-{regime}")
        vals.append(float(row["Episode_Duration_Sec"].iloc[0]) if not row.empty else 0.0)
        cols.append(col)
    ax2.bar(labels, vals, color=cols, edgecolor=BG, linewidth=1.0)

    ax3 = fig.add_subplot(gs[1, 1])
    _style_ax(ax3, "Total Pipeline Duration (Train)", "Group", "Seconds")
    elapsed = train.groupby(["Algorithm", "Seed_Regime_Group"], as_index=False)["Pipeline_Elapsed_Sec"].max()
    labels = []
    vals = []
    cols = []
    for algo, regime, col in profiles:
        row = elapsed[(elapsed["Algorithm"] == algo) & (elapsed["Seed_Regime_Group"] == regime)]
        labels.append(f"{algo}-{regime}")
        vals.append(float(row["Pipeline_Elapsed_Sec"].iloc[0]) if not row.empty else 0.0)
        cols.append(col)
    ax3.bar(labels, vals, color=cols, edgecolor=BG, linewidth=1.0)

    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Suite Visualiser for schema v2 CSV")
    p.add_argument("--file", "-f", type=str, default=None,
                   help="Path to a specific train_suite CSV (default: latest in CSV_History_SchemaV2).")
    p.add_argument("--show", action="store_true", help="Display figures after saving.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = args.file if args.file else _latest_csv()
    run_label = _short_label(csv_path)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[suite-v2-visualiser] Loading: {csv_path}")
    df = load_csv(csv_path)
    print(f"[suite-v2-visualiser] Rows: {len(df):,}")

    figs = [
        ("training_progression", fig_training_progression(df, run_label)),
        ("outcome_analysis", fig_outcome_analysis(df, run_label)),
        ("fixed_vs_random", fig_fixed_vs_random(df, run_label)),
        ("test_performance", fig_test_performance(df, run_label)),
        ("efficiency", fig_efficiency(df, run_label)),
    ]

    for tag, fig in figs:
        out = os.path.join(OUT_DIR, f"{run_label}_{tag}.png")
        fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[suite-v2-visualiser] Saved: {out}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()

