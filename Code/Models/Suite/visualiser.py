"""
Suite Training Visualiser
========================
Visualises train_suite_*.csv logs produced by train_suite.py.

Usage:
    python visualiser.py                # latest CSV, display windows
    python visualiser.py -f path.csv    # specific CSV
    python visualiser.py --all          # include all-runs overlay figure
    python visualiser.py --save         # save PNGs to Visualiser_Output/
    python visualiser.py --all --save   # both
"""

import os
import sys
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# -- Dark theme ---------------------------------------------------------------
plt.style.use("dark_background")
BG = "#0A0A0A"
PANEL_BG = "#111111"
ACCENT = "#00D4FF"
WIN_COL = "#00FF88"
LOSE_COL = "#FF4444"
WARN_COL = "#FFB347"
DQN_COL = "#4EA3FF"
NEAT_COL = "#C678DD"
GREY = "#888888"

ROLL_WIN = 100

# -- Paths --------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(_HERE, "CSV_History")

OUTCOME_COLORS = {
    "WIN": WIN_COL,
    "STARVATION": WARN_COL,
    "GHOST": NEAT_COL,
    "MAX_STEPS": "#FFD166",
    "NONE": GREY,
}

ALGO_COLORS = {
    "DQN": DQN_COL,
    "NEAT": NEAT_COL,
}


# -- Helpers ------------------------------------------------------------------
def latest_csv() -> str:
    files = glob.glob(os.path.join(CSV_DIR, "train_suite_*.csv"))
    if not files:
        sys.exit(f"[suite-visualiser] No train_suite_*.csv found in: {CSV_DIR}")
    return max(files, key=os.path.getmtime)


def short_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].replace("train_suite_", "")


def style_ax(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, color="white", pad=6)
    ax.set_xlabel(xlabel, fontsize=9, color=GREY)
    ax.set_ylabel(ylabel, fontsize=9, color=GREY)
    ax.tick_params(colors=GREY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_facecolor(PANEL_BG)


def roll(series: pd.Series, w: int = ROLL_WIN) -> pd.Series:
    return series.rolling(w, min_periods=1).mean()


def _to_bool(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["1", "true", "t", "yes", "y"])


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric = [
        "Episode", "Stage", "Maze_Seed", "Reward", "Macro_Steps", "Micro_Ticks",
        "Win", "Epsilon", "Pellets", "Power_Pellets", "Ghosts", "Explore_Rate", "Avg_Loss",
        "Generation", "Best_Fitness", "Avg_Fitness", "Species_Count", "Eval_Seeds_Per_Genome",
        "Max_Episode_Steps", "Episode_Duration_Sec", "Pipeline_Elapsed_Sec", "Test_Run_Elapsed_Sec",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Win" in df.columns:
        df["Win"] = df["Win"].fillna(0).astype(int)

    if "Algorithm" in df.columns:
        df["Algorithm"] = df["Algorithm"].astype(str).str.strip().str.upper()

    if "Outcome" in df.columns:
        df["Outcome"] = df["Outcome"].astype(str).str.strip().str.upper()

    if "Is_Test" in df.columns:
        df["Is_Test"] = _to_bool(df["Is_Test"])
    else:
        df["Is_Test"] = False

    return df


def _algo_splits(df: pd.DataFrame, is_test: bool) -> dict[str, pd.DataFrame]:
    subset = df[df["Is_Test"] == is_test].copy()
    splits = {}
    for algo in ["DQN", "NEAT"]:
        d = subset[subset["Algorithm"] == algo].copy()
        if len(d):
            splits[algo] = d.sort_values("Episode")
    return splits


# ============================================================================
# Figure 1 - Training Overview (Is_Test == False)
# ============================================================================
def fig_training_overview(df: pd.DataFrame, label: str) -> plt.Figure:
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Suite Training Overview - {label}", fontsize=14, color="white", y=0.99)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.28)
    train = _algo_splits(df, is_test=False)

    # 1) Rolling reward by algorithm
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, f"Rolling Reward (window={ROLL_WIN})", "Episode", "Reward")
    for algo, d in train.items():
        ax1.plot(d["Episode"], roll(d["Reward"]), linewidth=1.7,
                 color=ALGO_COLORS.get(algo, ACCENT), label=f"{algo} train")
    ax1.legend(fontsize=8)

    # 2) Rolling win rate by algorithm
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, f"Rolling Win Rate (window={ROLL_WIN})", "Episode", "Win Rate")
    for algo, d in train.items():
        wr = roll(d["Win"].astype(float)) * 100
        ax2.plot(d["Episode"], wr, linewidth=1.7,
                 color=ALGO_COLORS.get(algo, ACCENT), label=f"{algo} train")
    ax2.axhline(50, color=GREY, linestyle="--", linewidth=0.8)
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.legend(fontsize=8)

    # 3) Stage progression by algorithm
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "Curriculum Stage Progression", "Episode", "Stage")
    for algo, d in train.items():
        ax3.step(d["Episode"], d["Stage"], where="post", linewidth=1.8,
                 color=ALGO_COLORS.get(algo, ACCENT), label=f"{algo} train")
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.legend(fontsize=8)

    # 4) Outcome mix (train only)
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Outcome Distribution (Train)", "Outcome", "Count")
    train_df = df[df["Is_Test"] == False]
    if len(train_df):
        outcome_counts = train_df.groupby(["Algorithm", "Outcome"]).size().unstack(fill_value=0)
        outcomes = list(outcome_counts.columns)
        x = np.arange(len(outcome_counts.index))
        bottom = np.zeros(len(outcome_counts.index))
        for out in outcomes:
            vals = outcome_counts[out].values
            ax4.bar(
                x,
                vals,
                bottom=bottom,
                color=OUTCOME_COLORS.get(out, ACCENT),
                edgecolor=BG,
                linewidth=0.8,
                label=out,
            )
            bottom += vals
        ax4.set_xticks(x)
        ax4.set_xticklabels(outcome_counts.index)
        ax4.legend(fontsize=7, ncol=2)

    fig.text(0.5, 0.005, f"Source: {os.path.basename(label)}.csv",
             ha="center", fontsize=8, color="#444444")
    return fig


# ============================================================================
# Figure 2 - Test Comparison (Is_Test == True)
# ============================================================================
def fig_test_comparison(df: pd.DataFrame, label: str) -> plt.Figure:
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Suite Zero-Shot Test Comparison - {label}", fontsize=14, color="white", y=0.99)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)
    test = _algo_splits(df, is_test=True)

    # 1) Test reward distribution by algorithm
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "Test Reward Distribution", "Algorithm", "Reward")
    labels = []
    data = []
    colors = []
    for algo in ["DQN", "NEAT"]:
        d = test.get(algo)
        if d is not None and len(d):
            labels.append(algo)
            data.append(d["Reward"].dropna().values)
            colors.append(ALGO_COLORS.get(algo, ACCENT))
    if data:
        bp = ax1.boxplot(
            data,
            patch_artist=True,
            labels=labels,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=GREY),
            capprops=dict(color=GREY),
            flierprops=dict(marker=".", color="#555555", markersize=2),
        )
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)

    # 2) Test win rate by algorithm
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Test Win Rate", "Algorithm", "Win Rate")
    bars = []
    bar_vals = []
    bar_cols = []
    for algo in ["DQN", "NEAT"]:
        d = test.get(algo)
        if d is not None and len(d):
            bars.append(algo)
            bar_vals.append(float(d["Win"].mean() * 100))
            bar_cols.append(ALGO_COLORS.get(algo, ACCENT))
    if bar_vals:
        rects = ax2.bar(bars, bar_vals, color=bar_cols, edgecolor=BG, linewidth=1.0)
        for rect, val in zip(rects, bar_vals):
            ax2.text(rect.get_x() + rect.get_width() / 2, val + 1,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="white")
    ax2.axhline(50, color=GREY, linestyle="--", linewidth=0.8)
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())

    # 3) Test outcomes side-by-side
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "Test Outcome Counts", "Outcome", "Count")
    outcomes = sorted(df["Outcome"].dropna().unique().tolist())
    x = np.arange(len(outcomes))
    width = 0.38
    offset = 0
    for algo in ["DQN", "NEAT"]:
        d = test.get(algo)
        if d is None or not len(d):
            continue
        counts = d["Outcome"].value_counts()
        vals = [int(counts.get(o, 0)) for o in outcomes]
        ax3.bar(x + (offset * width), vals, width=width,
                color=ALGO_COLORS.get(algo, ACCENT), alpha=0.85, label=algo)
        offset += 1
    ax3.set_xticks(x + (width / 2 if offset > 1 else 0.0))
    ax3.set_xticklabels(outcomes, rotation=15)
    ax3.legend(fontsize=8)

    # 4) Test pellets and ghosts summary
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Test Resource Summary", "Metric", "Mean per Episode")
    metrics = ["Pellets", "Power_Pellets", "Ghosts", "Explore_Rate"]
    available = [m for m in metrics if m in df.columns]
    if available and len(test):
        x = np.arange(len(available))
        width = 0.38
        offset = 0
        for algo in ["DQN", "NEAT"]:
            d = test.get(algo)
            if d is None or not len(d):
                continue
            vals = [float(d[m].mean()) if m != "Explore_Rate" else float(d[m].mean() * 100) for m in available]
            ax4.bar(x + (offset * width), vals, width=width,
                    color=ALGO_COLORS.get(algo, ACCENT), alpha=0.85, label=algo)
            offset += 1
        labels_fmt = [m if m != "Explore_Rate" else "Explore_Rate (%)" for m in available]
        ax4.set_xticks(x + (width / 2 if offset > 1 else 0.0))
        ax4.set_xticklabels(labels_fmt, rotation=20)
        ax4.legend(fontsize=8)

    fig.text(0.5, 0.005, f"Source: {os.path.basename(label)}.csv",
             ha="center", fontsize=8, color="#444444")
    return fig


# ============================================================================
# Figure 3 - Algorithm Diagnostics
# ============================================================================
def fig_algorithm_diagnostics(df: pd.DataFrame, label: str) -> plt.Figure:
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Suite Algorithm Diagnostics - {label}", fontsize=14, color="white", y=0.99)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.30)
    train = _algo_splits(df, is_test=False)

    dqn = train.get("DQN", pd.DataFrame())
    neat = train.get("NEAT", pd.DataFrame())

    # 1) DQN epsilon decay
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "DQN Epsilon", "Episode", "Epsilon")
    if len(dqn) and "Epsilon" in dqn.columns:
        ax1.plot(dqn["Episode"], dqn["Epsilon"], color=DQN_COL, linewidth=1.6)
        ax1.set_ylim(0, 1.05)

    # 2) DQN average loss
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, f"DQN Avg Loss (rolling={ROLL_WIN})", "Episode", "Loss")
    if len(dqn) and "Avg_Loss" in dqn.columns:
        loss = dqn[dqn["Avg_Loss"] > 0]
        if len(loss):
            ax2.plot(loss["Episode"], loss["Avg_Loss"], color="#7755AA", linewidth=0.6, alpha=0.5)
            ax2.plot(loss["Episode"], roll(loss["Avg_Loss"]), color="#FF69B4", linewidth=1.8)

    # 3) DQN explore rate
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, "DQN Explore Rate", "Episode", "Explore Rate")
    if len(dqn) and "Explore_Rate" in dqn.columns:
        ax3.plot(dqn["Episode"], roll(dqn["Explore_Rate"]) * 100, color=ACCENT, linewidth=1.8)
        ax3.yaxis.set_major_formatter(mticker.PercentFormatter())

    # 4) NEAT best/avg fitness by generation
    ax4 = fig.add_subplot(gs[1, :2])
    style_ax(ax4, "NEAT Fitness by Generation", "Generation", "Fitness")
    if len(neat) and "Generation" in neat.columns:
        by_gen = neat.groupby("Generation", as_index=False).agg(
            Best_Fitness=("Best_Fitness", "max"),
            Avg_Fitness=("Avg_Fitness", "mean"),
        )
        by_gen = by_gen.sort_values("Generation")
        ax4.plot(by_gen["Generation"], by_gen["Best_Fitness"], color=NEAT_COL, linewidth=2.0, label="Best")
        ax4.plot(by_gen["Generation"], by_gen["Avg_Fitness"], color=ACCENT, linewidth=1.8, label="Average")
        ax4.legend(fontsize=8)

    # 5) NEAT species count by generation
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5, "NEAT Species Count", "Generation", "Species")
    if len(neat) and "Species_Count" in neat.columns:
        species = neat.groupby("Generation", as_index=False)["Species_Count"].max().sort_values("Generation")
        ax5.step(species["Generation"], species["Species_Count"], where="post", color=NEAT_COL, linewidth=1.8)
        ax5.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.text(0.5, 0.005, f"Source: {os.path.basename(label)}.csv",
             ha="center", fontsize=8, color="#444444")
    return fig


# ============================================================================
# Figure 4 - Summary Table
# ============================================================================
def fig_summary(df: pd.DataFrame, label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    fig.suptitle(f"Suite Summary - {label}", fontsize=14, color="white", y=0.97)

    rows = []
    for algo in ["DQN", "NEAT"]:
        for is_test in [False, True]:
            part = df[(df["Algorithm"] == algo) & (df["Is_Test"] == is_test)]
            if not len(part):
                continue
            tag = "TEST" if is_test else "TRAIN"
            rows.append({
                "Group": f"{algo}-{tag}",
                "Episodes": int(len(part)),
                "Win_%": float(part["Win"].mean() * 100),
                "Reward_Mean": float(part["Reward"].mean()),
                "Reward_Median": float(part["Reward"].median()),
                "Macro_Mean": float(part["Macro_Steps"].mean()),
                "Micro_Mean": float(part["Micro_Ticks"].mean()),
                "Pellets_Mean": float(part["Pellets"].mean()),
                "Ghosts_Mean": float(part["Ghosts"].mean()),
                "Ep_Duration_Mean_Sec": float(part.get("Episode_Duration_Sec", pd.Series(dtype=float)).mean()),
                "Phase_Elapsed_Max_Sec": float(part.get("Test_Run_Elapsed_Sec", part.get("Pipeline_Elapsed_Sec", pd.Series(dtype=float))).max()),
            })

    if not rows:
        return fig

    summary = pd.DataFrame(rows)
    summary = summary.round(3)

    tab = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        loc="center",
        cellLoc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.1, 1.8)

    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#2A2A2A")
        if r == 0:
            cell.set_facecolor("#1A1A2E")
            cell.set_text_props(color=ACCENT, fontweight="bold")
        else:
            cell.set_facecolor("#0F0F1A")
            cell.set_text_props(color="white")

    fig.text(0.5, 0.08, "Groups include train and zero-shot test segments per algorithm.",
             ha="center", fontsize=9, color="#BBBBBB")
    return fig


# ============================================================================
# Figure 5 - All suite runs overlay (--all)
# ============================================================================
def fig_all_runs(csv_dir: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.patch.set_facecolor(BG)
    fig.suptitle("All Suite Runs - Train Comparison", fontsize=14, color="white", y=0.99)

    ax_r, ax_w = axes
    style_ax(ax_r, f"Rolling Train Reward (window={ROLL_WIN})", "Episode", "Reward")
    style_ax(ax_w, f"Rolling Train Win Rate (window={ROLL_WIN})", "Episode", "Win Rate")
    ax_w.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax_w.set_ylim(0, 105)
    ax_w.axhline(50, color=GREY, linestyle="--", linewidth=0.8)

    all_csvs = sorted(glob.glob(os.path.join(csv_dir, "train_suite_*.csv")), key=os.path.getmtime)

    for i, path in enumerate(all_csvs):
        try:
            d = load_csv(path)
            train = d[d["Is_Test"] == False]
            if not len(train):
                continue

            lbl = short_label(path)
            col = plt.cm.tab20(i % 20)
            rr = roll(train["Reward"])
            wr = roll(train["Win"].astype(float)) * 100

            ax_r.plot(train["Episode"], rr, color=col, linewidth=1.2, alpha=0.9, label=lbl)
            ax_w.plot(train["Episode"], wr, color=col, linewidth=1.2, alpha=0.9, label=lbl)
        except Exception as exc:
            print(f"[suite-visualiser] Skipped {path}: {exc}")

    ax_r.legend(fontsize=7, loc="upper left", ncol=2,
                facecolor="#1A1A1A", edgecolor="#333333", labelcolor="white")
    ax_w.legend(fontsize=7, loc="upper left", ncol=2,
                facecolor="#1A1A1A", edgecolor="#333333", labelcolor="white")
    plt.tight_layout()
    return fig


# ============================================================================
# Entry point
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Suite Training Visualiser")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--file", "-f", metavar="CSV",
                   help="Path to a specific suite CSV file (default: latest)")
    g.add_argument("--all", "-a", action="store_true",
                   help="Also show a 5th figure overlaying all suite runs")
    p.add_argument("--save", "-s", action="store_false",
                   help="Save figures as PNGs to Visualiser_Output/ instead of displaying")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = args.file if args.file else latest_csv()
    run_label = short_label(csv_path)

    print(f"[suite-visualiser] Loading: {csv_path}")
    df = load_csv(csv_path)
    print(f"[suite-visualiser] Rows: {len(df):,} | Columns: {list(df.columns)}")

    figs: list[tuple[str, plt.Figure]] = [
        ("1_training_overview", fig_training_overview(df, run_label)),
        ("2_test_comparison", fig_test_comparison(df, run_label)),
        ("3_algorithm_diagnostics", fig_algorithm_diagnostics(df, run_label)),
        ("4_summary", fig_summary(df, run_label)),
    ]

    if args.all:
        figs.append(("5_all_runs_overlay", fig_all_runs(CSV_DIR)))

    if args.save:
        out_dir = os.path.join(_HERE, "Visualiser_Output")
        os.makedirs(out_dir, exist_ok=True)
        for name, fig in figs:
            out = os.path.join(out_dir, f"{run_label}_{name}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"[suite-visualiser] Saved: {out}")
        print("[suite-visualiser] Done.")
    else:
        plt.show()


if __name__ == "__main__":
    main()

