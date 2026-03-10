"""
neat_train_standard.py
======================
Trains a NEAT agent to play Pac-Man at FULL difficulty from generation 0.
No curriculum learning — uses game_settings.json as-is for every evaluation.

Usage
-----
    # Train from scratch
    python -m Code.Models.neat_train_standard

    # Resume from a NEAT checkpoint
    python -m Code.Models.neat_train_standard --checkpoint checkpoints/neat-checkpoint-9
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
from Code.Settings import Settings

# ── Load settings once — full difficulty, no overrides ───────────────────────
_SETTINGS_PATH = os.path.join(_ROOT, "Code", "game_settings.json")
FIXED_SETTINGS = Settings(_SETTINGS_PATH).get_all()

# ── Config ───────────────────────────────────────────────────────────────────
MAX_STEPS       = 3_500
NUM_GENERATIONS = 200
CHECKPOINT_DIR  = os.path.join(_HERE, "checkpoints_standard")
CONFIG_PATH     = os.path.join(_HERE, "neat_config.cfg")
PARALLEL        = True

NODE_NAMES = {
    # ── Pac-Man State ──
    -1: "Pac_X", -2: "Pac_Y", -3: "Pac_dX", -4: "Pac_dY",

    # ── Ghost 0 (Blinky) ──
    -5: "G0_relX", -6: "G0_relY", -7: "G0_Dist", -8: "G0_dX", -9: "G0_dY", -10: "G0_Threat",
    # ── Ghost 1 (Pinky) ──
    -11: "G1_relX", -12: "G1_relY", -13: "G1_Dist", -14: "G1_dX", -15: "G1_dY", -16: "G1_Threat",
    # ── Ghost 2 (Inky) ──
    -17: "G2_relX", -18: "G2_relY", -19: "G2_Dist", -20: "G2_dX", -21: "G2_dY", -22: "G2_Threat",
    # ── Ghost 3 (Clyde) ──
    -23: "G3_relX", -24: "G3_relY", -25: "G3_Dist", -26: "G3_dX", -27: "G3_dY", -28: "G3_Threat",

    # ── Radar & Walls ──
    -29: "Pellet_relX", -30: "Pellet_relY",
    -31: "Wall_Up", -32: "Wall_Down", -33: "Wall_Left", -34: "Wall_Right",

    # ── Global State ──
    -35: "Pellet_Ratio", -36: "Fright_Active", -37: "Fright_Timer",
    -38: "Lives", -39: "Scatter", -40: "PP_Dist",

    # ── Outputs ──
    0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"
}


# ── Visualisation helpers ─────────────────────────────────────────────────────

def plot_learning_curve(statistics, filename=None):
    if filename is None:
        filename = os.path.join(CHECKPOINT_DIR, "learning_curve.png")
    generation   = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness  = statistics.get_fitness_mean()

    plt.figure(figsize=(10, 6))
    plt.plot(generation, best_fitness, 'b-', label="Best Fitness")
    plt.plot(generation, avg_fitness,  'r-', label="Average Fitness")
    plt.title("NEAT Population Fitness over Generations (Standard)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()
    print(f"  [!] Learning curve saved to {filename}")


def draw_neural_net(config, genome, filename):
    dot = graphviz.Digraph(
        format='png',
        node_attr={'shape': 'circle', 'fontsize': '10', 'fontname': 'Helvetica'}
    )
    dot.attr(rankdir='LR')

    used_nodes = set()
    for cg in genome.connections.values():
        if cg.enabled:
            in_node, out_node = cg.key
            used_nodes.add(in_node)
            used_nodes.add(out_node)

    for i in range(config.genome_config.num_inputs):
        node_id = -(i + 1)
        if node_id in used_nodes:
            label = NODE_NAMES.get(node_id, str(node_id))
            dot.node(str(node_id), label, style='filled', fillcolor='lightblue')

    for i in range(config.genome_config.num_outputs):
        label = NODE_NAMES.get(i, str(i))
        dot.node(str(i), label, style='filled', fillcolor='salmon')

    for n in genome.nodes:
        if n not in NODE_NAMES and n in used_nodes:
            dot.node(str(n), f"Hidden_{n}", style='filled', fillcolor='lightgray')

    for cg in genome.connections.values():
        if cg.enabled:
            in_node, out_node = cg.key
            color     = 'green' if cg.weight > 0 else 'red'
            thickness = str(max(0.5, abs(cg.weight) * 0.8))
            dot.edge(str(in_node), str(out_node), color=color, penwidth=thickness)

    dot.render(filename, view=False, cleanup=True)
    print(f"  [!] Topology rendered to {filename}.png")


# ── Genome evaluation ─────────────────────────────────────────────────────────

def eval_genome(genome, config):
    """
    Evaluates a genome across 3 random mazes at full difficulty.
    Uses FIXED_SETTINGS — no curriculum adjustments.
    """
    net = neat.nn.RecurrentNetwork.create(genome, config)
    fitness_scores = []
    eval_seeds = [np.random.randint(0, 10000) for _ in range(3)]

    for seed in eval_seeds:
        env = PacManEnv(
            render_mode=None,
            obs_type="vector",
            maze_seed=seed,
            maze_algorithm="recursive_backtracking",
            max_episode_steps=MAX_STEPS,
            settings=FIXED_SETTINGS  # Full difficulty, always
        )

        obs, _ = env.reset()
        total_reward = 0.0

        try:
            while True:
                outputs = net.activate(obs.tolist())
                action  = int(np.argmax(outputs))
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
        finally:
            env.close()

        fitness_scores.append(total_reward)

    return float(np.mean(fitness_scores))


# ── Reporter that saves the best genome each generation ───────────────────────

class BestGenomeSaver(neat.reporting.BaseReporter):
    """Saves the best genome of every generation to disk."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.generation  = 0
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

            net_path = os.path.join(self.save_dir, f"topology_gen_{self.generation}")
            draw_neural_net(config, best_genome, net_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def run(checkpoint: str | None = None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"NEAT config not found at {CONFIG_PATH}")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    if checkpoint and os.path.exists(checkpoint):
        print(f"Resuming from checkpoint: {checkpoint}")
        population = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(
        neat.Checkpointer(
            generation_interval=10,
            filename_prefix=os.path.join(CHECKPOINT_DIR, "neat-checkpoint-"),
        )
    )
    population.add_reporter(BestGenomeSaver(CHECKPOINT_DIR))

    print(f"\n[Standard Training] Full difficulty from generation 0. "
          f"Running for {NUM_GENERATIONS} generations...\n")

    if PARALLEL:
        cores = multiprocessing.cpu_count()
        print(f"Initiating Parallel Evaluator on {cores} cores...")
        pe     = neat.ParallelEvaluator(cores, eval_genome)
        winner = population.run(pe.evaluate, NUM_GENERATIONS)
    else:
        winner = population.run(
            lambda genomes, cfg: [
                setattr(g, "fitness", eval_genome(g, cfg)) for _, g in genomes
            ],
            NUM_GENERATIONS,
        )

    winner_path = os.path.join(CHECKPOINT_DIR, "winner_genome.pkl")
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"\nEvolution Complete. Winner saved → {winner_path}")
    print(f"Maximum Fitness Achieved: {winner.fitness:.1f}")

    plot_learning_curve(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NEAT agent on Pac-Man (no curriculum)")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a neat-checkpoint file to resume from",
    )
    args = parser.parse_args()
    run(checkpoint=args.checkpoint)

