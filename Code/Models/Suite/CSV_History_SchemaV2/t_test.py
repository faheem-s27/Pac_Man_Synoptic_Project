import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('train_suite_merged_final.csv')

# Filter to reached_stage test only (random regime, n=100 each)
dqn_test = df[(df['Algorithm']=='DQN') &
              (df['Seed_Regime']=='random') &
              (df['Is_Test']==True) &
              (df['Test_Mode']=='reached_stage')]

neat_test = df[(df['Algorithm']=='NEAT') &
               (df['Seed_Regime']=='random') &
               (df['Is_Test']==True) &
               (df['Test_Mode']=='reached_stage')]

dqn_rewards = dqn_test['Reward'].values
neat_rewards = neat_test['Reward'].values

# Step 1 — Levene's test (checks if variances are equal)
levene_stat, levene_p = stats.levene(dqn_rewards, neat_rewards)
print(f"Levene's test: W={levene_stat:.4f}, p={levene_p:.4f}")
equal_var = levene_p > 0.05
print(f"Equal variance assumption: {equal_var}")

# Step 2 — Independent samples t-test
t_stat, p_val = stats.ttest_ind(dqn_rewards, neat_rewards, equal_var=equal_var)
print(f"\nT-test: t={t_stat:.4f}, p={p_val:.4f}")
print(f"DQN: M={np.mean(dqn_rewards):.2f}, SD={np.std(dqn_rewards):.2f}, n={len(dqn_rewards)}")
print(f"NEAT: M={np.mean(neat_rewards):.2f}, SD={np.std(neat_rewards):.2f}, n={len(neat_rewards)}")

if p_val < 0.05:
    print("Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print("Result: NOT statistically significant (p >= 0.05)")

# Step 3 — ANOVA across maze seeds
# Tests whether some seeds were systematically easier/harder
dqn_by_seed = [dqn_test[dqn_test['Maze_Seed']==s]['Reward'].values
               for s in dqn_test['Maze_Seed'].unique()
               if len(dqn_test[dqn_test['Maze_Seed']==s]) > 0]

f_stat, anova_p = stats.f_oneway(*dqn_by_seed)
print(f"\nDQN ANOVA (seed variability): F={f_stat:.4f}, p={anova_p:.4f}")

neat_by_seed = [neat_test[neat_test['Maze_Seed']==s]['Reward'].values
                for s in neat_test['Maze_Seed'].unique()
                if len(neat_test[neat_test['Maze_Seed']==s]) > 0]

f_stat, anova_p = stats.f_oneway(*neat_by_seed)
print(f"NEAT ANOVA (seed variability): F={f_stat:.4f}, p={anova_p:.4f}")

# Step 4 — Win rate comparison (proportions test)
from statsmodels.stats.proportion import proportions_ztest

dqn_wins = int(dqn_test['Win'].sum())
neat_wins = int(neat_test['Win'].sum())
count = np.array([dqn_wins, neat_wins])
nobs = np.array([len(dqn_rewards), len(neat_rewards)])
z_stat, z_p = proportions_ztest(count, nobs)
print(f"\nProportions z-test (win rate): z={z_stat:.4f}, p={z_p:.4f}")
print(f"DQN win rate: {dqn_wins/len(dqn_rewards)*100:.1f}%")
print(f"NEAT win rate: {neat_wins/len(neat_rewards)*100:.1f}%")