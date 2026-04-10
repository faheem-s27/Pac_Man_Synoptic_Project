"""
DQN Training Visualiser
=======================
Loads the latest training CSV from csv_history/ and renders four
figures of charts covering every column.

Usage:
    python visualiser.py                # latest CSV, display windows
    python visualiser.py -f path.csv    # specific CSV
    python visualiser.py --all          # overlay all runs on a 5th figure
    python visualiser.py --save         # write PNGs to visualiser_output/
    python visualiser.py --all --save   # both
"""

import os
import sys
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# ── Dark theme ───────────────────────────────────────────────────────────────
plt.style.use("dark_background")
BG       = "#0A0A0A"
PANEL_BG = "#111111"
ACCENT   = "#00D4FF"
WIN_COL  = "#00FF88"
LOSE_COL = "#FF4444"
WARN_COL = "#FFB347"
GHOST_COL= "#CC66FF"
PINK     = "#FF69B4"
GREY     = "#888888"
PALETTE  = [ACCENT, WIN_COL, LOSE_COL, WARN_COL, GHOST_COL, PINK, "#90EE90", "#FFA07A"]

ROLL_WIN = 50   # episodes for rolling average

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(_HERE, "CSV_History")

OUTCOME_COLORS = {
    "WIN":        WIN_COL,
    "STARVATION": WARN_COL,
    "GHOST":      GHOST_COL,
    "NONE":       GREY,
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def latest_csv() -> str:
    files = glob.glob(os.path.join(CSV_DIR, "training_log_*.csv"))
    if not files:
        sys.exit(f"[visualiser] No training_log_*.csv found in: {CSV_DIR}")
    return max(files, key=os.path.getmtime)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "Episode", "Stage", "Maze_Seed", "Reward", "Macro_Steps",
        "Micro_Ticks", "Win", "Epsilon", "Pellets", "Power_Pellets",
        "Ghosts", "Explore_Rate", "Avg_Loss",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Win" in df.columns:
        df["Win"] = df["Win"].fillna(0).astype(int)
    return df


def roll(series: pd.Series, w: int = ROLL_WIN) -> pd.Series:
    return series.rolling(w, min_periods=1).mean()


def style_ax(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, color="white", pad=6)
    ax.set_xlabel(xlabel, fontsize=9, color=GREY)
    ax.set_ylabel(ylabel, fontsize=9, color=GREY)
    ax.tick_params(colors=GREY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_facecolor(PANEL_BG)


def short_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].replace("training_log_", "")


# ════════════════════════════════════════════════════════════════════════════════
# Figure 1 — Training Progress
# ════════════════════════════════════════════════════════════════════════════════
def fig_training_progress(df: pd.DataFrame, label: str) -> plt.Figure:
    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"DQN Training Progress   ·   {label}   ·   {len(df):,} episodes",
        fontsize=14, color="white", y=0.99,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)
    ep = df["Episode"]

    # ── 1. Reward + rolling (wide) ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    style_ax(ax, "Episode Reward", "Episode", "Reward")
    wins  = df["Win"] == 1
    ax.scatter(ep[wins],  df["Reward"][wins],  s=2, c=WIN_COL,  alpha=0.15, linewidths=0)
    ax.scatter(ep[~wins], df["Reward"][~wins], s=2, c=LOSE_COL, alpha=0.10, linewidths=0)
    ax.plot(ep, df["Reward"], color="#333355", linewidth=0.4, alpha=0.7)
    ax.plot(ep, roll(df["Reward"]), color=ACCENT, linewidth=2.0,
            label=f"Rolling {ROLL_WIN}-ep avg")
    ax.legend(fontsize=8, loc="upper left")

    # ── 2. Rolling win rate ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2, f"Rolling Win Rate  (window={ROLL_WIN})", "Episode", "Win Rate")
    wr = roll(df["Win"].astype(float)) * 100
    ax2.plot(ep, wr, color=WIN_COL, linewidth=1.8)
    ax2.fill_between(ep, wr, alpha=0.15, color=WIN_COL)
    ax2.axhline(50, color=GREY, linestyle="--", linewidth=0.8, label="50 %")
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.legend(fontsize=8)

    # ── 3. Epsilon decay ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "Epsilon (Exploration) Decay", "Episode", "ε")
    ax3.plot(ep, df["Epsilon"], color=WARN_COL, linewidth=1.6)
    ax3.set_ylim(0, 1.05)

    # ── 4. Average Loss ───────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Average Loss", "Episode", "Loss")
    loss = df[df["Avg_Loss"] > 0]
    if len(loss):
        ax4.plot(loss["Episode"], loss["Avg_Loss"],
                 color=PINK, linewidth=0.6, alpha=0.45)
        ax4.plot(loss["Episode"], roll(loss["Avg_Loss"]),
                 color=PINK, linewidth=1.8, label="Rolling avg")
        ax4.legend(fontsize=8)

    # ── 5. Macro Steps ────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5, "Episode Length (Macro Steps)", "Episode", "Steps")
    ax5.plot(ep, df["Macro_Steps"], color="#444466", linewidth=0.4, alpha=0.6)
    ax5.plot(ep, roll(df["Macro_Steps"]), color=ACCENT, linewidth=1.8)

    # ── 6. Pellets eaten ──────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    style_ax(ax6, "Pellets Eaten per Episode", "Episode", "Pellets")
    ax6.plot(ep, df["Pellets"], color="#333333", linewidth=0.4, alpha=0.5)
    ax6.plot(ep, roll(df["Pellets"]), color=WIN_COL, linewidth=1.8)

    # ── 7. Explore Rate ───────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    style_ax(ax7, "Explore Rate (% novel states)", "Episode", "Explore Rate")
    ax7.plot(ep, df["Explore_Rate"] * 100, color="#333333", linewidth=0.4, alpha=0.5)
    ax7.plot(ep, roll(df["Explore_Rate"]) * 100, color=WARN_COL, linewidth=1.8)
    ax7.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax7.set_ylim(0)

    # ── 8. Stage progression ─────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    style_ax(ax8, "Curriculum Stage Over Time", "Episode", "Stage")
    ax8.step(ep, df["Stage"], where="post", color=GHOST_COL, linewidth=2.0)
    ax8.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax8.fill_between(ep, df["Stage"], step="post", alpha=0.15, color=GHOST_COL)

    fig.text(0.5, 0.005, f"Source: {os.path.basename(label)}.csv",
             ha="center", fontsize=8, color="#444444")
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# Figure 2 — Distributions & Breakdowns
# ════════════════════════════════════════════════════════════════════════════════
def fig_distributions(df: pd.DataFrame, label: str) -> plt.Figure:
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"DQN Distributions & Breakdowns   ·   {label}",
        fontsize=14, color="white", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)

    # ── 1. Outcome pie ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Outcome Distribution", fontsize=11, color="white", pad=6)
    counts = df["Outcome"].value_counts()
    colors = [OUTCOME_COLORS.get(o, ACCENT) for o in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=colors, startangle=90,
        wedgeprops=dict(edgecolor=BG, linewidth=2),
    )
    for t in texts:     t.set_color("white"); t.set_fontsize(9)
    for t in autotexts: t.set_color("#111111"); t.set_fontsize(8); t.set_fontweight("bold")

    # ── 2. Win rate by stage bar chart ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Win Rate by Stage", "Stage", "Win Rate")
    stage_wr = df.groupby("Stage")["Win"].mean() * 100
    bar_colors = [WIN_COL if v >= 50 else LOSE_COL for v in stage_wr.values]
    bars = ax2.bar(
        stage_wr.index.astype(str), stage_wr.values,
        color=bar_colors, edgecolor=BG, linewidth=1.2,
    )
    ax2.axhline(50, color=GREY, linestyle="--", linewidth=0.9)
    ax2.set_ylim(0, 112)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    for bar, v in zip(bars, stage_wr.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 2,
                 f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color="white")

    # ── 3. Reward histogram ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, "Reward Distribution (Histogram)", "Reward", "Count")
    ax3.hist(df["Reward"].dropna(), bins=70, color=ACCENT,
             edgecolor=BG, linewidth=0.3, alpha=0.85)
    med = df["Reward"].median()
    mu  = df["Reward"].mean()
    ax3.axvline(med, color=WIN_COL,  linestyle="--", linewidth=1.4,
                label=f"Median {med:.0f}")
    ax3.axvline(mu,  color=WARN_COL, linestyle="--", linewidth=1.4,
                label=f"Mean {mu:.0f}")
    ax3.legend(fontsize=8)

    # ── 4. Reward boxplot by outcome ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    style_ax(ax4, "Reward by Outcome", "Outcome", "Reward")
    outcomes = list(df["Outcome"].unique())
    data_by_outcome = [df[df["Outcome"] == o]["Reward"].dropna().values for o in outcomes]
    bp = ax4.boxplot(
        data_by_outcome, patch_artist=True, labels=outcomes,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=GREY), capprops=dict(color=GREY),
        flierprops=dict(marker=".", color="#555555", markersize=2),
    )
    for patch, o in zip(bp["boxes"], outcomes):
        patch.set_facecolor(OUTCOME_COLORS.get(o, ACCENT))
        patch.set_alpha(0.75)
    ax4.tick_params(axis="x", colors="white", labelsize=9)

    # ── 5. Ghosts eaten histogram ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5, "Ghosts Eaten per Episode", "Ghosts", "Count")
    ghost_vals = df["Ghosts"].dropna().astype(int)
    max_g = max(int(ghost_vals.max()), 1)
    bins  = np.arange(-0.5, max_g + 1.5, 1)
    ax5.hist(ghost_vals, bins=bins, color=GHOST_COL, edgecolor=BG, linewidth=0.5)
    ax5.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 6. Pellets vs Reward scatter ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6, "Pellets Eaten vs Reward", "Pellets", "Reward")
    wm = df["Win"] == 1
    ax6.scatter(df[~wm]["Pellets"], df[~wm]["Reward"],
                s=3, c=LOSE_COL, alpha=0.20, linewidths=0, label="Loss")
    ax6.scatter(df[wm]["Pellets"],  df[wm]["Reward"],
                s=3, c=WIN_COL,  alpha=0.30, linewidths=0, label="Win")
    ax6.legend(fontsize=8, markerscale=5)

    fig.text(0.5, 0.005, f"Source: {os.path.basename(label)}.csv",
             ha="center", fontsize=8, color="#444444")
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# Figure 3 — Correlations & Stage Deep Dive
# ════════════════════════════════════════════════════════════════════════════════
def fig_correlation(df: pd.DataFrame, label: str) -> plt.Figure:
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"DQN Correlation & Stage Analysis   ·   {label}",
        fontsize=14, color="white", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40)

    keep = ["Reward", "Macro_Steps", "Micro_Ticks", "Epsilon",
            "Pellets", "Power_Pellets", "Ghosts", "Explore_Rate", "Avg_Loss", "Win"]
    num_df = df[[c for c in keep if c in df.columns]].dropna()

    # ── 1. Correlation heatmap (wide) ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Pearson Correlation Heatmap", fontsize=11, color="white", pad=6)
    corr = num_df.corr()
    im   = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=GREY, labelsize=8)
    n = len(corr)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8, color="white")
    ax.set_yticklabels(corr.columns, fontsize=8, color="white")
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="#000000" if abs(val) > 0.45 else "white")

    # ── 2. Cumulative win rate ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2, "Cumulative Win Rate", "Episode", "Win Rate")
    cum_wr = df["Win"].expanding().mean() * 100
    ax2.plot(df["Episode"], cum_wr, color=WIN_COL, linewidth=1.6)
    ax2.fill_between(df["Episode"], cum_wr, alpha=0.12, color=WIN_COL)
    ax2.axhline(50, color=GREY, linestyle="--", linewidth=0.8)
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())

    # ── 3. Reward boxplot by stage ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "Reward Distribution by Stage", "Stage", "Reward")
    stages     = sorted(df["Stage"].dropna().unique())
    stage_data = [df[df["Stage"] == s]["Reward"].dropna().values for s in stages]
    bp = ax3.boxplot(
        stage_data, patch_artist=True, labels=[str(int(s)) for s in stages],
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=GREY), capprops=dict(color=GREY),
        flierprops=dict(marker=".", color="#555555", markersize=1.5),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.75)
    ax3.tick_params(axis="x", colors="white")

    # ── 4. Episodes per stage bar ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Episode Count per Stage", "Stage", "Episodes")
    sc = df["Stage"].value_counts().sort_index()
    bars = ax4.bar(
        sc.index.astype(str), sc.values,
        color=[PALETTE[i % len(PALETTE)] for i in range(len(sc))],
        edgecolor=BG, linewidth=1,
    )
    for bar, v in zip(bars, sc.values):
        ax4.text(bar.get_x() + bar.get_width() / 2, v + max(sc.values) * 0.01,
                 f"{v:,}", ha="center", va="bottom", fontsize=8, color="white")

    # ── 5. Micro Ticks vs Macro Steps scatter ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5, "Micro Ticks vs Macro Steps", "Macro Steps", "Micro Ticks")
    wm = df["Win"] == 1
    ax5.scatter(df[~wm]["Macro_Steps"], df[~wm]["Micro_Ticks"],
                s=3, c=LOSE_COL, alpha=0.18, linewidths=0, label="Loss")
    ax5.scatter(df[wm]["Macro_Steps"],  df[wm]["Micro_Ticks"],
                s=3, c=WIN_COL,  alpha=0.25, linewidths=0, label="Win")
    # regression line
    try:
        clean = df[["Macro_Steps", "Micro_Ticks"]].dropna()
        m, b  = np.polyfit(clean["Macro_Steps"], clean["Micro_Ticks"], 1)
        xs    = np.linspace(clean["Macro_Steps"].min(), clean["Macro_Steps"].max(), 200)
        ax5.plot(xs, m * xs + b, color=ACCENT, linewidth=1.4, linestyle="--",
                 label=f"fit: {m:.1f}x + {b:.0f}")
    except Exception:
        pass
    ax5.legend(fontsize=7, markerscale=4)

    fig.text(0.5, 0.005, f"Source: {os.path.basename(label)}.csv",
             ha="center", fontsize=8, color="#444444")
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# Figure 4 — Summary Stats Table
# ════════════════════════════════════════════════════════════════════════════════
def fig_summary(df: pd.DataFrame, label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    fig.suptitle(f"DQN Summary Statistics   ·   {label}", fontsize=14,
                 color="white", y=0.97)

    total_eps  = len(df)
    total_wins = int(df["Win"].sum())
    win_pct    = total_wins / total_eps * 100 if total_eps else 0
    stages     = sorted(df["Stage"].dropna().unique().astype(int).tolist())
    outcome_ct = df["Outcome"].value_counts().to_dict()

    info = (
        f"Total Episodes: {total_eps:,}   |   Wins: {total_wins:,} ({win_pct:.1f}%)   |   "
        f"Stages: {stages}   |   Outcomes: {outcome_ct}"
    )
    fig.text(0.5, 0.88, info, ha="center", fontsize=9, color="#BBBBBB")

    keep = ["Reward", "Macro_Steps", "Micro_Ticks", "Epsilon",
            "Pellets", "Power_Pellets", "Ghosts", "Explore_Rate", "Avg_Loss"]
    cols    = [c for c in keep if c in df.columns]
    summary = df[cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    summary = summary.round(4)

    tab = ax.table(
        cellText=summary.values,
        rowLabels=summary.index,
        colLabels=summary.columns,
        loc="center",
        cellLoc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.3, 1.7)

    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#2A2A2A")
        if r == 0:
            cell.set_facecolor("#1A1A2E")
            cell.set_text_props(color=ACCENT, fontweight="bold")
        elif c == -1:
            cell.set_facecolor("#1A1A2E")
            cell.set_text_props(color=WARN_COL, fontweight="bold")
        else:
            cell.set_facecolor("#0F0F1A")
            cell.set_text_props(color="white")

    return fig


# ════════════════════════════════════════════════════════════════════════════════
# Figure 5 — All Runs Overlay (--all flag)
# ════════════════════════════════════════════════════════════════════════════════
def fig_all_runs(csv_dir: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.patch.set_facecolor(BG)
    fig.suptitle("All Training Runs — Comparison", fontsize=14, color="white", y=0.99)

    ax_r, ax_w = axes
    style_ax(ax_r, "Rolling Reward — All Runs", "Episode", f"Reward (rolling {ROLL_WIN})")
    style_ax(ax_w, "Rolling Win Rate — All Runs", "Episode", "Win Rate")
    ax_w.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax_w.set_ylim(0, 105)
    ax_w.axhline(50, color=GREY, linestyle="--", linewidth=0.8)

    all_csvs = sorted(
        glob.glob(os.path.join(csv_dir, "training_log_*.csv")),
        key=os.path.getmtime,
    )
    for i, path in enumerate(all_csvs):
        try:
            d   = load_csv(path)
            lbl = short_label(path)
            col = PALETTE[i % len(PALETTE)]
            ax_r.plot(d["Episode"], roll(d["Reward"]),
                      color=col, linewidth=1.3, label=lbl, alpha=0.9)
            ax_w.plot(d["Episode"], roll(d["Win"].astype(float)) * 100,
                      color=col, linewidth=1.3, label=lbl, alpha=0.9)
        except Exception as exc:
            print(f"[visualiser] Skipped {path}: {exc}")

    ax_r.legend(fontsize=7, loc="upper left", ncol=2,
                facecolor="#1A1A1A", edgecolor="#333333", labelcolor="white")
    ax_w.legend(fontsize=7, loc="upper left", ncol=2,
                facecolor="#1A1A1A", edgecolor="#333333", labelcolor="white")
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DQN Training Visualiser")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--file", "-f", metavar="CSV",
                   help="Path to a specific CSV file (default: latest)")
    g.add_argument("--all",  "-a", action="store_true",
                   help="Also show a 5th figure overlaying all runs")
    p.add_argument("--save", "-s", action="store_true",
                   help="Save figures as PNGs to visualiser_output/ instead of displaying")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path  = args.file if args.file else latest_csv()
    run_label = short_label(csv_path)

    print(f"[visualiser] Loading: {csv_path}")
    df = load_csv(csv_path)
    print(f"[visualiser] {len(df):,} episodes  |  columns: {list(df.columns)}")

    figs: list[tuple[str, plt.Figure]] = [
        ("1_training_progress",  fig_training_progress(df, run_label)),
        ("2_distributions",      fig_distributions(df, run_label)),
        ("3_correlation_stages", fig_correlation(df, run_label)),
        ("4_summary_stats",      fig_summary(df, run_label)),
    ]

    if args.all:
        figs.append(("5_all_runs_overlay", fig_all_runs(CSV_DIR)))

    if args.save:
        out_dir = os.path.join(_HERE, "Visualiser_Output")
        os.makedirs(out_dir, exist_ok=True)
        for name, fig in figs:
            out = os.path.join(out_dir, f"{run_label}_{name}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[visualiser] Saved: {out}")
        print("[visualiser] Done.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
