"""
neat_replay.py
==============
Load a saved NEAT genome and watch it play Pac-Man in a visible window.

Usage
-----
    # Watch the best genome on the default settings map
    python -m Code.neat_replay --genome checkpoints/best_genome.pkl

    # Prove Generalisation: Watch the genome on a completely random, unseen map
    python -m Code.neat_replay --genome checkpoints/best_genome.pkl --random
"""

import os
import sys
import pickle
import argparse
import time
import numpy as np
import random

import neat

_HERE      = os.path.dirname(os.path.abspath(__file__))
_NEAT_ROOT = os.path.dirname(_HERE)                          # Code/Models/NEAT/
_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))  # project root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.Environment.PacManEnv import PacManEnv
from Code.Settings  import Settings
from Code.Environment.CurriculumManager import CurriculumManager

# ── Config ───────────────────────────────────────────────────────────────────
_SETTINGS      = Settings(os.path.join(_ROOT, "Code", "game_settings.json")).get_all()
CONFIG_PATH    = os.path.join(_NEAT_ROOT, "neat_config.cfg")
ACTION_NAMES   = {0: "FORWARD", 1: "LEFT", 2: "RIGHT", 3: "BACKWARD"}
NEAT_MAX_EPISODE_STEPS = None


def _settings_for_generation(cm: CurriculumManager, generation: int) -> dict:
    stage_count = max(1, len(cm.stage_profiles))
    # Replay accepts an absolute generation number, so map by fixed generation span per stage.
    stage_span = 50
    stage_idx = min(stage_count - 1, max(0, int(generation // stage_span)))
    cm.current_stage = stage_idx
    settings = cm.get_settings()
    settings["max_episode_steps"] = NEAT_MAX_EPISODE_STEPS
    return settings


def replay(genome_path: str, test_generalisation: bool, generation: int):
    if not os.path.exists(genome_path):
        print(f"[ERROR] Genome file not found: {genome_path}")
        sys.exit(1)

    # Load genome
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    print(f"Loaded genome from: {genome_path}")
    print(f"  Genome fitness : {getattr(genome, 'fitness', 'unknown')}")
    print(f"  Nodes          : {len(genome.nodes)}")
    print(f"  Connections    : {len(genome.connections)}")

    # Build network
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    probe_env = PacManEnv(render_mode=None, obs_type="vector", settings=_settings_for_generation(CurriculumManager(), generation))
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

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Determine seed for generalisation testing
    eval_seed = random.randint(0, 999999) if test_generalisation else _SETTINGS.get("maze_seed", None)
    if test_generalisation:
        print(f"  [!] Zero-Shot Generalisation Test Activated. Procedural Seed: {eval_seed}")

    # Load Curriculum Settings
    cm = CurriculumManager()
    print(f"  [!] Applying Curriculum Settings for Generation {generation}")
    curriculum_settings = _settings_for_generation(cm, generation)

    # Run with rendering
    env = PacManEnv(
        render_mode="human",
        obs_type="vector",
        maze_seed=eval_seed,
        settings=curriculum_settings,
    )

    obs, _ = env.reset()
    total_reward = 0.0
    step = 0

    print("\nWatching NEAT genome play — close the window to stop.\n")

    try:
        while True:
            # Replaced native Python indexing with NumPy argmax for mathematical consistency
            outputs = net.activate(obs.tolist())
            valid_actions = env.get_valid_actions()
            action = max(valid_actions, key=lambda a: outputs[a]) if valid_actions else int(np.argmax(outputs))

            obs, reward, terminated, truncated, info = env.step(action)
            reward = float(reward)
            total_reward += reward
            step += 1

            if reward > 0.1 or reward < -0.5:
                print(f"  [step {step:>5}] {ACTION_NAMES[action]:<5}  "
                      f"reward={reward:+.1f}  cumulative={total_reward:+.1f}  "
                      f"score={info.get('score', 0)}  lives={getattr(env.engine, 'lives', 0)}")

            if terminated or truncated:
                status = "GAME OVER" if info.get("game_over") else "TRUNCATED"
                print(f"\nEpisode ended ({status}) after {step} steps")
                print(f"  Final score    : {info.get('score', 0)}")
                print(f"  Levels cleared : {getattr(env.engine, 'level', 1)}")
                print(f"  Total reward   : {total_reward:+.1f}")
                break

            time.sleep(0.016)   # ~60 fps

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a saved NEAT genome")
    parser.add_argument(
        "--genome",
        type=str,
        default=os.path.join(_NEAT_ROOT, "Checkpoints", "best_genome.pkl"),
        help="Path to the pickled genome file",
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=1,
        help="Simulate the curriculum difficulty of this generation (default: 100 - full difficulty)",
    )
    parser.add_argument(
        "--random",
        action="store_false",
        help="Test zero-shot generalisation by generating a completely random, unseen map.",
    )
    args = parser.parse_args()
    replay(args.genome, args.random, args.generation)
