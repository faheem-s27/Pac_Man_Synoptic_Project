"""
neat_train.py
=============
Trains a NEAT agent to play Pac-Man using the PacManEnv Gymnasium wrapper.

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
import numpy as np

import neat

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.PacManEnv import PacManEnv

# ── Config ───────────────────────────────────────────────────────────────────
# We define a static evaluation suite to ensure fair comparison across generations
EVALUATION_SEEDS = [42, 100, 999]
MAX_STEPS        = 10_000          # Increased to allow for multi-level survival
NUM_GENERATIONS  = 200
CHECKPOINT_DIR   = os.path.join(_HERE, "checkpoints")
CONFIG_PATH      = os.path.join(_HERE, "neat_config.cfg") # Matched your previous filename
PARALLEL         = True

# ── Genome evaluation ─────────────────────────────────────────────────────────

def eval_genome(genome, config):
    """
    Evaluates a genome across multiple procedurally generated mazes.
    Returns the mean fitness to enforce generalisation over memorisation.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Track performance across the battery of tests
    fitness_scores = []

    eval_seeds = [np.random.randint(0, 10000) for _ in range(3)]

    for seed in eval_seeds:
        env = PacManEnv(
            render_mode=None,
            obs_type="vector",
            maze_seed=seed,
            maze_algorithm="recursive_backtracking",
            max_episode_steps=MAX_STEPS
        )

        obs, _ = env.reset()
        total_reward = 0.0

        try:
            while True:
                # Pass the 40-element vector to the neural network
                outputs = net.activate(obs.tolist())
                action = int(np.argmax(outputs))

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    break
        finally:
            # Crucial: Prevent memory leaks from PyGame headless surfaces
            env.close()

        fitness_scores.append(total_reward)

    # The genome's final fitness is its average performance across all tested maps
    return float(np.mean(fitness_scores))


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

    def post_evaluate(self, config, population, species, best_genome):
        if best_genome is not None and best_genome.fitness > self.best_fitness:
            self.best_fitness = best_genome.fitness
            path = os.path.join(self.save_dir, "best_genome.pkl")
            with open(path, "wb") as f:
                pickle.dump(best_genome, f)
            print(f"  [!] New best genome saved! Fitness: {self.best_fitness:.1f} → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(checkpoint: str | None = None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"NEAT config file not found at {CONFIG_PATH}")

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
        # Use available CPU cores to evaluate the population concurrently
        cores = multiprocessing.cpu_count()
        print(f"Initiating Parallel Evaluator on {cores} cores...")
        pe = neat.ParallelEvaluator(cores, eval_genome)
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
    print(f"\nEvolution Complete. Winner saved → {winner_path}")
    print(f"Maximum Fitness Achieved: {winner.fitness:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NEAT agent on Pac-Man")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a neat-checkpoint file to resume from",
    )
    args = parser.parse_args()
    run(checkpoint=args.checkpoint)