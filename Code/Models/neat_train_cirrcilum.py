"""
neat_train_cirrcilum.py
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
import matplotlib.pyplot as plt
import graphviz

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.PacManEnv import PacManEnv
from Code.CurriculumManager import CurriculumManager

# ── Global State for Curriculum ──────────────────────────────────────────────
# This global is updated by the main loop before each generation,
# so that worker processes (if using fork) or local eval see the new difficulty.
CURRENT_SETTINGS = None
# CurriculumManager anchors game_settings.json to its own __file__ (Code/ folder)
CURRICULUM = CurriculumManager()


# ── Config ───────────────────────────────────────────────────────────────────
# We define a static evaluation suite to ensure fair comparison across generations
MAX_STEPS        = 2000          # Increased to allow for multi-level survival
NUM_GENERATIONS  = 500
CHECKPOINT_DIR   = os.path.join(_HERE, "checkpoints")
CONFIG_PATH      = os.path.join(_HERE, "neat_config.cfg") # Matched your previous filename
PARALLEL         = True

NODE_NAMES = {
    # ── Outputs (egocentric) ──
    0: "FORWARD", 1: "LEFT", 2: "RIGHT", 3: "BACKWARD"
}


def _settings_for_generation(generation: int) -> dict:
    """Map generation ranges to curriculum stages and return matching settings."""
    if generation < 50:
        CURRICULUM.current_stage = 0
    elif generation < 100:
        CURRICULUM.current_stage = 1
    elif generation < 180:
        CURRICULUM.current_stage = 2
    elif generation < 260:
        CURRICULUM.current_stage = 3
    elif generation < 360:
        CURRICULUM.current_stage = 4
    else:
        CURRICULUM.current_stage = 5
    return CURRICULUM.get_settings()

def plot_learning_curve(statistics, filename="checkpoints/learning_curve.png"):
    """
    Setup: Extracts fitness data from the NEAT reporter.
    Action: Plots the Best vs Average fitness over all generations.
    Result: Saves a PNG graph to the checkpoints folder.
    """
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = statistics.get_fitness_mean()

    plt.figure(figsize=(10, 6))
    plt.plot(generation, best_fitness, 'b-', label="Best Fitness")
    plt.plot(generation, avg_fitness, 'r-', label="Average Fitness")

    plt.title("NEAT Population Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Score - Penalties)")
    plt.grid(True)
    plt.legend(loc="best")

    plt.savefig(filename)
    plt.close()
    print(f"  [!] Learning curve saved to {filename}")


def draw_neural_net(config, genome, filename):
    """
    Setup: Takes the raw genome configuration.
    Action: Uses Graphviz to render the active topological connections using the cg.key tuple.
    Result: Saves an SVG/PNG visual of the network, skipping disconnected inputs.
    """
    dot = graphviz.Digraph(format='png', node_attr={'shape': 'circle', 'fontsize': '10', 'fontname': 'Helvetica'})
    dot.attr(rankdir='LR')  # Left to Right layout

    # Find all nodes that actually have active connections
    used_nodes = set()
    for cg in genome.connections.values():
        if cg.enabled:
            in_node, out_node = cg.key  # <-- THE FIX: Unpack the key tuple
            used_nodes.add(in_node)
            used_nodes.add(out_node)

    # Draw ONLY the used Inputs (Blue) to prevent massive visual clutter
    for i in range(config.genome_config.num_inputs):
        node_id = -(i + 1)
        if node_id in used_nodes:
            label = NODE_NAMES.get(node_id, str(node_id))
            dot.node(str(node_id), label, style='filled', fillcolor='lightblue')

    # Draw Outputs (Red)
    for i in range(config.genome_config.num_outputs):
        label = NODE_NAMES.get(i, str(i))
        dot.node(str(i), label, style='filled', fillcolor='salmon')

    # Draw Hidden Nodes (Gray)
    for n in genome.nodes:
        if n not in NODE_NAMES and n in used_nodes:
            dot.node(str(n), f"Hidden_{n}", style='filled', fillcolor='lightgray')

    # Draw Connections
    for cg in genome.connections.values():
        if cg.enabled:
            in_node, out_node = cg.key  # <-- THE FIX: Unpack the key tuple
            # Green for excitatory (positive), Red for inhibitory (negative)
            color = 'green' if cg.weight > 0 else 'red'
            # Thickness based on weight magnitude
            thickness = str(max(0.5, abs(cg.weight) * 0.8))
            dot.edge(str(in_node), str(out_node), color=color, penwidth=thickness)

    dot.render(filename, view=False, cleanup=True)
    print(f"  [!] Topology rendered to {filename}.png")

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
        # Use the CURRENT_SETTINGS which are updated per-generation by the main loop
        env = PacManEnv(
            render_mode=None,
            obs_type="vector",
            maze_seed=seed,
            maze_algorithm="recursive_backtracking",
            settings=CURRENT_SETTINGS,  # <-- New: Dynamic Curriculum Settings
            max_episode_steps=MAX_STEPS,
        )

        obs, _ = env.reset()
        total_reward = 0.0

        try:
            while True:
                outputs = net.activate(obs.tolist())
                valid_actions = env.get_valid_actions()
                action = max(valid_actions, key=lambda a: outputs[a]) if valid_actions else int(np.argmax(outputs))

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

            # Action: Draw the network of the new champion
            net_path = os.path.join(self.save_dir, f"topology_gen_{self.generation}")
            draw_neural_net(config, best_genome, net_path)


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

    probe_settings = _settings_for_generation(0)
    probe_env = PacManEnv(render_mode=None, obs_type="vector", settings=probe_settings)
    probe_obs, _ = probe_env.reset(seed=123)
    obs_dim = len(probe_obs)
    action_dim = int(getattr(probe_env.action_space, "n", 4))
    probe_env.close()
    if config.genome_config.num_inputs != obs_dim:
        raise ValueError(
            f"NEAT config num_inputs={config.genome_config.num_inputs} does not match PacManEnv obs_dim={obs_dim}."
        )
    if config.genome_config.num_outputs != action_dim:
        raise ValueError(
            f"NEAT config num_outputs={config.genome_config.num_outputs} does not match action_dim={action_dim}."
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

    # Evaluate genomes loop
    global CURRENT_SETTINGS

    winner = None

    if PARALLEL:
        # Use available CPU cores to evaluate the population concurrently
        cores = multiprocessing.cpu_count()
        print(f"Initiating Parallel Evaluator on {cores} cores...")
        pe = neat.ParallelEvaluator(cores, eval_genome)
        eval_function = pe.evaluate
    else:
        eval_function = lambda genomes, cfg: [
            setattr(g, "fitness", eval_genome(g, cfg)) for _, g in genomes
        ]

    # Manual Curriculum Loop
    # We run 1 generation at a time so we can update settings dynamically
    print(f"\nStarting Curriculum Training for {NUM_GENERATIONS} generations...")

    for _ in range(NUM_GENERATIONS):
        # 1. Update global settings for this generation
        gen_id = population.generation
        CURRENT_SETTINGS = _settings_for_generation(gen_id)

        # 2. Run ONE generation
        # 'winner' will be updated every generation, but we only care about the final one
        winner = population.run(eval_function, 1)

    # Save final winner
    winner_path = os.path.join(CHECKPOINT_DIR, "winner_genome.pkl")
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"\nEvolution Complete. Winner saved → {winner_path}")
    print(f"Maximum Fitness Achieved: {winner.fitness:.1f}")

    plot_learning_curve(population.reporters.reporters[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NEAT agent on Pac-Man")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a neat-checkpoint file to resume from",
    )
    args = parser.parse_args()
    run(checkpoint=args.checkpoint)