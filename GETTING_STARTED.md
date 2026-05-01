# Getting Started

## Requirements

Python 3.11 is required. Download from [python.org](https://www.python.org/downloads/) — tick **"Add Python to PATH"** during install.

---

## Setup (first time only)

Double-click **`setup.bat`** at the project root. This installs all required packages automatically.

If you have an NVIDIA GPU and want faster DQN training, run this once after setup:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

CPU-only also works — the DQN will just train slower.

---

## Running the project

Double-click **`run.bat`**. This opens the launcher menu:

```
  1.  Play Pac-Man (manual)

  2.  Train DQN  (headless, fast)
  3.  Train DQN  (visual, with dashboard)
  4.  Train NEAT (headless, fast)
  5.  Train NEAT (visual, watch genomes)
  6.  Full training suite (DQN + NEAT)

  7.  Watch trained DQN play
  8.  Watch trained NEAT play
  8r. Watch NEAT on random (unseen) mazes

  9.  Benchmark tests (fixed maze)
  10. Generate result charts
  11. Statistical analysis
```

Type a number and press Enter. Press Enter again when done to return to the menu.

---

## What each option does

| Option | Description |
|---|---|
| 1 | Play Pac-Man yourself. Arrow keys to move. New procedural maze each time. |
| 2 | Train DQN with no window — fastest, recommended for long runs. Saves checkpoints to `Code/Models/DQN/Checkpoints/`. |
| 3 | Train DQN with a live dashboard showing the game and training metrics. |
| 4 | Train NEAT with no window — uses multiprocessing, resumes from `--checkpoint` if interrupted. |
| 5 | Train NEAT with all genomes rendered in a live grid. Press ESC to stop and save. |
| 6 | Full dissertation pipeline — trains DQN and NEAT across all seed regimes sequentially. Takes several hours. Logs to `Code/Models/Suite/CSV_History_SchemaV2/`. |
| 7 | Load the saved DQN checkpoint and watch it play. |
| 8 | Load the saved NEAT genome and watch it play on the training maze. |
| 8r | Same as 8 but tests on random unseen mazes (generalization test). |
| 9 | Run both saved models on the fixed benchmark maze (seed `22459265`) for 100 episodes each. |
| 10 | Generate the 5-panel result charts from CSV data. Saved to `Code/Models/Suite/Visualiser_Output_V2/`. |
| 11 | Run statistical analysis (win rates, t-tests, p-values) across all training conditions. |

---

## Configuring the game

Open `Code/game_settings.json` to change how the game behaves:

| Setting | What it does |
|---|---|
| `maze_seed` | Specific seed for a repeatable maze. Leave `""` for random each time. |
| `maze_algorithm` | Generation style: `"recursive_backtracking"`, `"prims"`, or `"random_walk"` |
| `lives` | Number of lives before game over |
| `pacman_speed` | How fast Pac-Man moves |
| `enable_ghosts` | `true` / `false` — toggle all ghosts |
| `ghost_speed` | Base ghost movement speed |
| `blinky_active` / `pinky_active` / `inky_active` / `clyde_active` | Toggle individual ghosts |
| `always_chase` | `true` forces ghosts to always chase (no scatter phase) |
| `enable_power_pellets` | Toggle power pellets |
| `god_mode` | `true` makes Pac-Man invincible |
| `window_resolution` | Window size, e.g. `"1000x1000"` |
