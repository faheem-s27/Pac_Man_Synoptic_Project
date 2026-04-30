"""
statistical_analysis.py
========================
Comprehensive statistical analysis for the Nature vs Nurture Pac-Man dissertation.
Run this against the merged final CSV to produce all results needed for Chapters 4 and 5.

Usage:
    python statistical_analysis.py --csv train_suite_merged_final.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings('ignore')

# ── Helpers ───────────────────────────────────────────────────────────────────

def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def section(title):
    print(f"\n--- {title} ---")

def get_train(df, alg, regime):
    return df[(df['Algorithm']==alg) &
              (df['Seed_Regime']==regime) &
              (df['Is_Test']==False)].copy()

def get_test(df, alg, regime, test_mode):
    return df[(df['Algorithm']==alg) &
              (df['Seed_Regime']==regime) &
              (df['Is_Test']==True) &
              (df['Test_Mode']==test_mode)].copy()

def ttest_report(name_a, vals_a, name_b, vals_b):
    if len(vals_a) < 2 or len(vals_b) < 2:
        print(f"  Insufficient data: {name_a} n={len(vals_a)}, {name_b} n={len(vals_b)}")
        return None, None
    levene_stat, levene_p = stats.levene(vals_a, vals_b)
    equal_var = levene_p > 0.05
    t_stat, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=equal_var)
    df_val = len(vals_a) + len(vals_b) - 2
    print(f"  Levene's test: W={levene_stat:.4f}, p={levene_p:.4f} -> equal_var={equal_var}")
    print(f"  {name_a}: M={np.mean(vals_a):.2f}, SD={np.std(vals_a):.2f}, n={len(vals_a)}")
    print(f"  {name_b}: M={np.mean(vals_b):.2f}, SD={np.std(vals_b):.2f}, n={len(vals_b)}")
    test_type = "Independent samples t-test" if equal_var else "Welch's t-test"
    print(f"  {test_type}: t({df_val})={t_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.001:
        print(f"  Result: SIGNIFICANT (p < 0.001) ***")
    elif p_val < 0.01:
        print(f"  Result: SIGNIFICANT (p < 0.01) **")
    elif p_val < 0.05:
        print(f"  Result: SIGNIFICANT (p < 0.05) *")
    else:
        print(f"  Result: NOT significant (p = {p_val:.4f})")
    return t_stat, p_val

def proportions_report(name_a, wins_a, n_a, name_b, wins_b, n_b):
    if n_a < 2 or n_b < 2:
        print(f"  Insufficient data for proportions test")
        return
    count = np.array([wins_a, wins_b])
    nobs = np.array([n_a, n_b])
    z_stat, p_val = proportions_ztest(count, nobs)
    print(f"  {name_a}: {wins_a}/{n_a} = {wins_a/n_a*100:.1f}%")
    print(f"  {name_b}: {wins_b}/{n_b} = {wins_b/n_b*100:.1f}%")
    print(f"  Proportions z-test: z={z_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        print(f"  Result: SIGNIFICANT (p < 0.05) *")
    else:
        print(f"  Result: NOT significant (p = {p_val:.4f})")

def anova_report(name, groups_dict):
    valid = {k: v for k, v in groups_dict.items() if len(v) > 1}
    if len(valid) < 2:
        print(f"  Insufficient groups for ANOVA (need at least 2 with n > 1)")
        return
    f_stat, p_val = stats.f_oneway(*valid.values())
    print(f"  Groups: {list(valid.keys())} (n={[len(v) for v in valid.values()]})")
    print(f"  One-way ANOVA: F={f_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        print(f"  Result: SIGNIFICANT — mean reward differs significantly across groups *")
    else:
        print(f"  Result: NOT significant — no significant difference across groups")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(csv_path):
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    df['Win'] = df['Win'].astype(int)
    df['Reward'] = df['Reward'].astype(float)
    df['Stage'] = df['Stage'].astype(int)
    df['Is_Test'] = df['Is_Test'].astype(str).str.lower().isin(['true', '1'])

    print(f"Total rows: {len(df)}")

    # Pull training and test groups
    dqn_rand_train  = get_train(df, 'DQN',  'random')
    dqn_fixed_train = get_train(df, 'DQN',  'fixed_22459265')
    neat_rand_train = get_train(df, 'NEAT', 'random')
    neat_fixed_train= get_train(df, 'NEAT', 'fixed_22459265')

    dqn_rand_test   = get_test(df, 'DQN',  'random',         'reached_stage')
    neat_rand_test  = get_test(df, 'NEAT', 'random',         'reached_stage')
    dqn_fixed_test  = get_test(df, 'DQN',  'fixed_22459265', 'reached_stage')
    neat_fixed_test = get_test(df, 'NEAT', 'fixed_22459265', 'reached_stage')

    dqn_rand_stage7_test   = get_test(df, 'DQN',  'random',         'fixed_stage7')
    neat_rand_stage7_test  = get_test(df, 'NEAT', 'random',         'fixed_stage7')
    dqn_fixed_stage7_test  = get_test(df, 'DQN',  'fixed_22459265', 'fixed_stage7')
    neat_fixed_stage7_test = get_test(df, 'NEAT', 'fixed_22459265', 'fixed_stage7')

    # ── SECTION 1: PRIMARY GENERALISATION TEST ────────────────────────────────
    separator("1. PRIMARY GENERALISATION TEST (Random Seed, Reached Stage)")

    section("1.1 T-Test: DQN vs NEAT — Test Reward")
    ttest_report(
        "DQN-random",  dqn_rand_test['Reward'].values,
        "NEAT-random", neat_rand_test['Reward'].values
    )

    section("1.2 Proportions Z-Test: DQN vs NEAT — Win Rate")
    proportions_report(
        "DQN-random",  int(dqn_rand_test['Win'].sum()),  len(dqn_rand_test),
        "NEAT-random", int(neat_rand_test['Win'].sum()), len(neat_rand_test)
    )

    section("1.3 T-Test: DQN vs NEAT — Pellets Collected")
    ttest_report(
        "DQN-random",  dqn_rand_test['Pellets'].values,
        "NEAT-random", neat_rand_test['Pellets'].values
    )

    section("1.4 T-Test: DQN vs NEAT — Explore Rate")
    ttest_report(
        "DQN-random",  dqn_rand_test['Explore_Rate'].values,
        "NEAT-random", neat_rand_test['Explore_Rate'].values
    )

    # ── SECTION 2: FIXED SEED TEST ────────────────────────────────────────────
    separator("2. FIXED SEED TEST (Fixed Seed 22459265, Reached Stage)")

    section("2.1 T-Test: DQN vs NEAT — Test Reward (Fixed)")
    ttest_report(
        "DQN-fixed",  dqn_fixed_test['Reward'].values,
        "NEAT-fixed", neat_fixed_test['Reward'].values
    )

    section("2.2 Proportions Z-Test: DQN vs NEAT — Win Rate (Fixed)")
    proportions_report(
        "DQN-fixed",  int(dqn_fixed_test['Win'].sum()),  len(dqn_fixed_test),
        "NEAT-fixed", int(neat_fixed_test['Win'].sum()), len(neat_fixed_test)
    )

    # ── SECTION 3: MEMORISATION EFFECT ───────────────────────────────────────
    separator("3. MEMORISATION EFFECT (Fixed vs Random, Same Algorithm)")

    section("3.1 DQN: Fixed vs Random — Test Reward")
    ttest_report(
        "DQN-fixed",  dqn_fixed_test['Reward'].values,
        "DQN-random", dqn_rand_test['Reward'].values
    )

    section("3.2 NEAT: Fixed vs Random — Test Reward")
    ttest_report(
        "NEAT-fixed",  neat_fixed_test['Reward'].values,
        "NEAT-random", neat_rand_test['Reward'].values
    )

    section("3.3 DQN: Fixed vs Random — Win Rate")
    proportions_report(
        "DQN-fixed",  int(dqn_fixed_test['Win'].sum()),  len(dqn_fixed_test),
        "DQN-random", int(dqn_rand_test['Win'].sum()), len(dqn_rand_test)
    )

    section("3.4 NEAT: Fixed vs Random — Win Rate")
    proportions_report(
        "NEAT-fixed",  int(neat_fixed_test['Win'].sum()),  len(neat_fixed_test),
        "NEAT-random", int(neat_rand_test['Win'].sum()), len(neat_rand_test)
    )

    # ── SECTION 4: FULL GAME TEST (Stage 7) ───────────────────────────────────
    separator("4. FULL GAME TEST (Fixed Stage 7, Both Regimes)")

    section("4.1 T-Test: DQN vs NEAT — Full Game Reward (Random Regime)")
    ttest_report(
        "DQN-random-stage7",  dqn_rand_stage7_test['Reward'].values,
        "NEAT-random-stage7", neat_rand_stage7_test['Reward'].values
    )

    section("4.2 T-Test: DQN vs NEAT — Full Game Reward (Fixed Regime)")
    ttest_report(
        "DQN-fixed-stage7",  dqn_fixed_stage7_test['Reward'].values,
        "NEAT-fixed-stage7", neat_fixed_stage7_test['Reward'].values
    )

    # ── SECTION 5: CURRICULUM EFFECTIVENESS (ANOVA) ───────────────────────────
    separator("5. CURRICULUM EFFECTIVENESS (ANOVA Across Stages)")

    section("5.1 DQN Random — Reward by Stage")
    dqn_stage_groups = {
        f"Stage {s}": dqn_rand_train[dqn_rand_train['Stage']==s]['Reward'].values
        for s in sorted(dqn_rand_train['Stage'].unique())
        if len(dqn_rand_train[dqn_rand_train['Stage']==s]) > 30
    }
    anova_report("DQN Random", dqn_stage_groups)

    section("5.2 NEAT Random — Reward by Stage")
    neat_stage_groups = {
        f"Stage {s}": neat_rand_train[neat_rand_train['Stage']==s]['Reward'].values
        for s in sorted(neat_rand_train['Stage'].unique())
        if len(neat_rand_train[neat_rand_train['Stage']==s]) > 30
    }
    anova_report("NEAT Random", neat_stage_groups)

    section("5.3 DQN Fixed — Reward by Stage")
    dqn_fixed_stage_groups = {
        f"Stage {s}": dqn_fixed_train[dqn_fixed_train['Stage']==s]['Reward'].values
        for s in sorted(dqn_fixed_train['Stage'].unique())
        if len(dqn_fixed_train[dqn_fixed_train['Stage']==s]) > 30
    }
    anova_report("DQN Fixed", dqn_fixed_stage_groups)

    section("5.4 NEAT Fixed — Reward by Stage")
    neat_fixed_stage_groups = {
        f"Stage {s}": neat_fixed_train[neat_fixed_train['Stage']==s]['Reward'].values
        for s in sorted(neat_fixed_train['Stage'].unique())
        if len(neat_fixed_train[neat_fixed_train['Stage']==s]) > 30
    }
    anova_report("NEAT Fixed", neat_fixed_stage_groups)

    # ── SECTION 6: TRAINING EFFICIENCY ────────────────────────────────────────
    separator("6. TRAINING EFFICIENCY")

    section("6.1 T-Test: DQN vs NEAT — Episode Duration")
    dqn_dur  = dqn_rand_train['Episode_Duration_Sec'].dropna().values
    neat_dur = neat_rand_train['Episode_Duration_Sec'].dropna().values
    ttest_report("DQN-random", dqn_dur, "NEAT-random", neat_dur)

    section("6.2 Pipeline Duration Summary")
    for name, grp in [
        ("DQN-random",  dqn_rand_train),
        ("DQN-fixed",   dqn_fixed_train),
        ("NEAT-random", neat_rand_train),
        ("NEAT-fixed",  neat_fixed_train),
    ]:
        elapsed = grp['Pipeline_Elapsed_Sec'].dropna()
        if len(elapsed):
            print(f"  {name}: {max(elapsed)/3600:.2f} hrs | avg_ep={grp['Episode_Duration_Sec'].dropna().mean():.3f}s")

    # ── SECTION 7: GHOST INTERACTION ──────────────────────────────────────────
    separator("7. GHOST INTERACTION — DQN vs NEAT (Random Training)")

    section("7.1 T-Test: Ghosts Eaten per Episode (Training, Ghost Stages Only)")
    dqn_ghost_stages  = dqn_rand_train[dqn_rand_train['Stage'] >= 2]['Ghosts'].values
    neat_ghost_stages = neat_rand_train[neat_rand_train['Stage'] >= 2]['Ghosts'].values
    ttest_report("DQN-random (stages 2+)", dqn_ghost_stages,
                 "NEAT-random (stages 2+)", neat_ghost_stages)

    section("7.2 T-Test: Ghosts Eaten per Episode (Test)")
    ttest_report(
        "DQN-random-test",  dqn_rand_test['Ghosts'].values,
        "NEAT-random-test", neat_rand_test['Ghosts'].values
    )

    # ── SECTION 8: SUMMARY TABLE ──────────────────────────────────────────────
    separator("8. SUMMARY TABLE — All Test Conditions")
    print(f"\n{'Condition':<35} {'n':>5} {'Wins':>6} {'Win%':>7} {'M Reward':>10} {'SD Reward':>10}")
    print("-" * 80)
    for label, grp in [
        ("DQN random (reached stage)",   dqn_rand_test),
        ("DQN fixed  (reached stage)",   dqn_fixed_test),
        ("NEAT random (reached stage)",  neat_rand_test),
        ("NEAT fixed  (reached stage)",  neat_fixed_test),
        ("DQN random (stage 7)",         dqn_rand_stage7_test),
        ("DQN fixed  (stage 7)",         dqn_fixed_stage7_test),
        ("NEAT random (stage 7)",        neat_rand_stage7_test),
        ("NEAT fixed  (stage 7)",        neat_fixed_stage7_test),
    ]:
        if len(grp) == 0:
            continue
        n = len(grp)
        wins = int(grp['Win'].sum())
        win_pct = wins/n*100
        m = grp['Reward'].mean()
        sd = grp['Reward'].std()
        print(f"  {label:<33} {n:>5} {wins:>6} {win_pct:>6.1f}% {m:>10.1f} {sd:>10.1f}")

    print("\nDone. Use these results in Chapters 4 and 5.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="train_suite_merged_final.csv")
    args = parser.parse_args()
    main(args.csv)