"""
neat_train.py
=============
Trains a NEAT agent to play Pac-Man using the PacManEnv Gymnasium wrapper.

Each genome is evaluated by running one full episode (headless) and its
fitness is the total accumulated reward.  The best genome of each generation
is saved to  checkpoints/best_genome.pkl  and can be replayed with
neat_replay.py.

Usage
-----
    # Train from scratch
    python -m Code.neat_train

    # Resume from a NEAT checkpoint
    python -m Code.neat_train --checkpoint checkpoints/neat-checkpoint-9
"""

import os
import sys
import pickle
import argparse
import multiprocessing

import neat

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.PacManEnv import PacManEnv
from Code.Settings  import Settings

# ── Config ───────────────────────────────────────────────────────────────────
def _load_settings() -> dict:
    return Settings(os.path.join(_HERE, "game_settings.json")).get_all()

_SETTINGS       = _load_settings()
MAZE_SEED       = _SETTINGS.get("maze_seed", None)   # read from game_settings.json
MAZE_ALGORITHM  = "recursive_backtracking"
MAX_STEPS       = 5_000
NUM_GENERATIONS = 200
CHECKPOINT_DIR  = os.path.join(_HERE, "checkpoints")
CONFIG_PATH     = os.path.join(_HERE, "neat_config.ini")
PARALLEL        = True


# ── Genome evaluation ─────────────────────────────────────────────────────────

def eval_genome(genome, config):
    """
    Run one headless episode for a single genome and return its fitness.
    fitness = total reward accumulated over the episode.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = PacManEnv(
        render_mode=None,
        obs_type="vector",
        maze_seed=MAZE_SEED,
        maze_algorithm=MAZE_ALGORITHM,
        max_episode_steps=MAX_STEPS,
        enable_sound=False,
    )

    obs, _ = env.reset()
    total_reward = 0.0

    while True:
        outputs = net.activate(obs.tolist())
        action  = outputs.index(max(outputs))   # argmax → discrete action

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()
    return total_reward


# ── Reporter that saves the best genome each generation ───────────────────────

class BestGenomeSaver(neat.reporting.BaseReporter):
    """Saves the best genome of every generation to disk."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.generation = 0
        self.best_fitness = float("-inf")

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species_set, best_genome):
        if best_genome.fitness > self.best_fitness:
            self.best_fitness = best_genome.fitness
            path = os.path.join(self.save_dir, "best_genome.pkl")
            with open(path, "wb") as f:
                pickle.dump(best_genome, f)
            print(f"  ✓ New best genome saved  fitness={self.best_fitness:.1f}  →  {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(checkpoint: str | None = None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    # Create or restore population
    if checkpoint and os.path.exists(checkpoint):
        print(f"Resuming from checkpoint: {checkpoint}")
        population = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        population = neat.Population(config)

    # Reporters
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(
        neat.Checkpointer(
            generation_interval=10,
            filename_prefix=os.path.join(CHECKPOINT_DIR, "neat-checkpoint-"),
        )
    )
    population.add_reporter(BestGenomeSaver(CHECKPOINT_DIR))

    # Evaluate genomes
    if PARALLEL:
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
        winner = population.run(pe.evaluate, NUM_GENERATIONS)
    else:
        winner = population.run(
            lambda genomes, cfg: [
                setattr(g, "fitness", eval_genome(g, cfg)) for _, g in genomes
            ],
            NUM_GENERATIONS,
        )

    # Save final winner
    winner_path = os.path.join(CHECKPOINT_DIR, "winner_genome.pkl")
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"\nWinner saved → {winner_path}")
    print(f"Winner fitness: {winner.fitness:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NEAT agent on Pac-Man")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a neat-checkpoint file to resume from",
    )
    args = parser.parse_args()
    run(checkpoint=args.checkpoint)

