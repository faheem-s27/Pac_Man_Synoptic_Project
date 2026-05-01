import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

MENU = """
+==========================================+
|      Pac-Man AI Project  -  Launcher     |
+==========================================+
|  PLAY                                    |
|  1.  Play Pac-Man (manual)               |
|                                          |
|  TRAIN                                   |
|  2.  Train DQN  (headless, fast)         |
|  3.  Train DQN  (visual, with dashboard) |
|  4.  Train NEAT (headless, fast)         |
|  5.  Train NEAT (visual, watch genomes)  |
|  6.  Full training suite (DQN + NEAT)    |
|                                          |
|  WATCH AI                                |
|  7.  Watch trained DQN play              |
|  8.  Watch trained NEAT play             |
|  8r. Watch NEAT on random (unseen) mazes |
|                                          |
|  ANALYSIS                                |
|  9.  Benchmark tests (fixed maze)        |
|  10. Generate result charts              |
|  11. Statistical analysis                |
|                                          |
|  0.  Exit                                |
+==========================================+
"""

SCRIPTS = {
    "1":  ("Code/main.py", []),
    "2":  ("Code/Models/DQN/Training/dqn_train_headless.py", []),
    "3":  ("Code/Models/DQN/Training/dqn_train_visual.py", []),
    "4":  ("Code/Models/NEAT/Training/neat_train_headless.py", []),
    "5":  ("Code/Models/NEAT/Training/neat_train_visual.py", []),
    "6":  ("Code/train_suite.py", []),
    "7":  ("Code/Models/DQN/Testing/eval_dqn.py", []),
    "8":  ("Code/Models/NEAT/Testing/neat_replay.py", []),
    "8r": ("Code/Models/NEAT/Testing/neat_replay.py", ["--random"]),
    "9":  ("Code/Tools/run_fixed_seed_tests_only.py", []),
    "10": ("Code/Models/Suite/visualiser_schema_v2.py", []),
    "11": ("Code/Models/Suite/CSV_History_SchemaV2/statistical_analysis.py", []),
}

def run(choice):
    entry = SCRIPTS.get(choice)
    if not entry:
        print("  Invalid choice — try again.")
        return
    rel_path, extra_args = entry
    abs_path = os.path.join(ROOT, rel_path)
    if not os.path.exists(abs_path):
        print(f"  Script not found: {rel_path}")
        return
    print(f"\n  Launching: {rel_path}\n  {'='*42}")
    subprocess.run([sys.executable, abs_path] + extra_args, cwd=ROOT)

def main():
    print("\n  Python", sys.version.split()[0])
    while True:
        print(MENU)
        choice = input("  Enter choice: ").strip().lower()
        if choice == "0":
            print("  Goodbye.\n")
            break
        run(choice)
        input("\n  Press Enter to return to menu...")

if __name__ == "__main__":
    main()
