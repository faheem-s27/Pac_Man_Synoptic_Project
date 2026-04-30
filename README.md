# Nature vs Nurture: Comparing DQN and NEAT for Generalisable AI in Pac-Man

**BSc Computer Science, Dissertation Project**

**Faheem Saleem**

---

## What This Project Is

Most AI agents that "beat" Pac-Man are quietly cheating. They play the same fixed maze thousands of times until they have memorised every turn. Change one wall and they fall apart completely.

This project asks a different question: **can an AI agent learn to play mazes it has never seen before?** And does the method of learning actually make a difference?

Two algorithms are trained and tested under identical conditions:

- **DQN (Deep Q-Network)**: learns by trial and error, adjusting a neural network based on rewards earned over time
- **NEAT (NeuroEvolution of Augmenting Topologies)**: uses natural selection to evolve a population of networks across generations

Both are trained on procedurally generated mazes so memorisation is impossible, then tested on 100 held-out mazes neither algorithm has ever seen.

---

## Key Results

| Condition | Stage Reached | Test Win Rate |
|---|---|---|
| DQN (random training) | Stage 4 | 26% |
| NEAT (random training) | Stage 5 | **37%** |
| DQN (fixed training) | Stage 5 | 6% |
| NEAT (fixed training) | Stage 7 | 0% |

NEAT trained on random mazes won 37% of test games with roughly a third of the episode budget of the other conditions. NEAT trained on one fixed maze reached Stage 7, the full game, and won zero test games.

The biggest takeaway is not really which algorithm is better. It is that **training regime matters more than curriculum stage reached**. Memorising one layout does not transfer to new ones. Training on varied mazes does.

NEAT also collected 45% more pellets per episode than DQN on unseen mazes (p < 0.001). The win rate gap of 37% vs 26% did not reach statistical significance (p = 0.094).

---

## Project Structure

```
Code/
├── Engine/             Game loop, Pac-Man, ghosts (A* pathfinding, 4 named ghost AIs)
├── Environment/        Gymnasium wrapper (29D observation, Discrete(4) actions)
│   └── CurriculumManager.py   8-stage progressive difficulty ladder
├── Maze/               Procedural maze generation (depth-first backtracking, BFS validation)
├── Models/
│   ├── DQN/            Dueling DQN, Prioritised Experience Replay (SumTree), Welford normaliser
│   ├── NEAT/           neat-python population training, multiprocessing evaluation
│   └── Suite/          Master training pipeline, CSV logging, statistical analysis, visualiser
└── Tools/              Fixed-seed benchmark runner, maze debug utilities

game_settings.json      JSON config for all hyperparameters
```

---

## The Environment

Built from scratch in Python and Pygame with no pre-built RL environment. A few key design choices:

- **Tile-lock stepping**: the agent makes one decision per tile move rather than every frame, compressing roughly 10,000 raw frames into around 200 meaningful decisions per episode
- **Procedural maze generation**: a new maze is generated before every training episode; BFS flood-fill validation rejects any maze with unreachable areas before it starts
- **Four ghost AIs**: Blinky (direct chase), Pinky (intercept ahead), Inky (vector projection), Clyde (switches based on distance). Together they create pressure from multiple directions so the agent cannot just learn one avoidance pattern.
- **Gymnasium interface**: both DQN and NEAT use the exact same environment with no algorithm-specific modifications

### Observation Space

A 29-dimensional vector split into three blocks:

| Block | Size | Content |
|---|---|---|
| A: Raycasts | 24 values | Distance to walls, food, power pellets and ghosts in all 4 directions. Rotates with the agent so inputs always mean the same thing. |
| B: Global Paths | 3 values | Shortest walkable distance to nearest food, nearest dangerous ghost and nearest edible ghost |
| C: Power State | 2 values | Whether frightened mode is active and how much time remains |

---

## Curriculum Learning

Training on the full game from scratch produced no useful learning signal at all. The 8-stage curriculum starts with no ghosts and basic pellet collection, then introduces ghosts one by one at increasing speeds.

To advance a stage the agent must hit the win rate threshold consistently across a **150-episode rolling window**. It cannot just get lucky and skip ahead.

| Stage | Ghosts | Speed | Win Target | Notes |
|---|---|---|---|---|
| 0 | None | n/a | 20% | Basic navigation |
| 1 | None | n/a | 60% | Full maze coverage |
| 2 | Blinky | 0.9 | 75% | Low-pressure ghost awareness |
| 3 | Blinky | 1.1 | 70% | Ghost-eating bridge (+150 kill bonus) |
| 4 | Blinky | 1.4 | 70% | Single ghost, bonus removed |
| 5 | Blinky + Pinky | 1.2 | 70% | Two-ghost interaction |
| 6 | Blinky + Pinky + Inky | 1.55 | 60% | Three-ghost pressure |
| 7 | All four | 1.85 | 60% | Full game |

---

## Algorithm Details

### DQN
- Dueling architecture (separate value and advantage streams)
- Prioritised Experience Replay with SumTree (surprising transitions are replayed more often)
- Welford online reward normalisation (reward range spans -500 to +1000)
- 8 parallel environments via `SyncVectorEnv` to avoid layout memorisation
- CUDA/GPU training via PyTorch
- Epsilon-greedy exploration: 1.0 to 0.15 | LR: 3e-4 to 1e-4 | gamma = 0.997 | batch = 256

### NEAT
- Population of 150 genomes evolved using neat-python
- Evolves both weights and network topology, starting from zero hidden nodes
- Speciation keeps similar networks competing against each other first, giving new mutations time to develop before facing the whole population
- Multi-seed fitness evaluation per genome for more reliable scoring
- Runs entirely on CPU across all available cores
- Compatibility threshold = 3.0 | stagnation limit = 30 generations | elitism = 1

---

## Hardware

| Component | Spec | Role |
|---|---|---|
| CPU | Intel Core i7-8700 (6c / 12t) | 8 parallel DQN envs + NEAT population evaluation |
| GPU | NVIDIA RTX 4060 8 GB | All DQN forward passes and gradient updates via CUDA |
| RAM | 32 GB DDR4 | Holds the full 200,000-transition DQN replay buffer in memory |

---

## Requirements

```
Python 3.11
torch
neat-python
gymnasium
pygame
pandas
scipy
matplotlib
```

Install with:

```bash
pip install torch neat-python gymnasium pygame pandas scipy matplotlib
```

---

## Running the Project

**Full training pipeline (both algorithms, both seed regimes):**
```bash
python Code/Models/Suite/train_suite.py
```

**Fixed-seed benchmark tests only:**
```bash
python Code/Tools/run_fixed_seed_tests_only.py
```

**Statistical analysis and visualisation:**
```bash
python Code/Models/Suite/CSV_History_SchemaV2/statistical_analysis.py
python Code/Models/Suite/visualiser_schema_v2.py
```

Training outputs (CSV logs and checkpoints) are saved to `Code/Models/DQN/` and `Code/Models/NEAT/`. Charts go to `Code/Models/Suite/Visualiser_Output_V2/`.
