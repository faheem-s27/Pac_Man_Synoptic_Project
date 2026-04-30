import pandas as pd

# Load both files
main = pd.read_csv('train_suite_21-04_04-23-22.csv')
retest = pd.read_csv('train_suite_fixed_retest_23-04_04-26-56.csv')

# Remove the old n=1 fixed seed test rows from main
main_cleaned = main[~(
    (main['Seed_Regime'] == 'fixed_22459265') &
    (main['Is_Test'] == True)
)]

print(f"Main before: {len(main)} rows")
print(f"Main after removing old fixed tests: {len(main_cleaned)} rows")
print(f"Retest rows to add: {len(retest)} rows")

# Append new retest rows
merged = pd.concat([main_cleaned, retest], ignore_index=True)
print(f"Merged total: {len(merged)} rows")

# Save
merged.to_csv('train_suite_merged_final.csv', index=False)
print("Saved to train_suite_merged_final.csv")