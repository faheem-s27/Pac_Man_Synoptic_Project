"""
train_suite.py
==============
Master orchestration script for dissertation data-collection runs.

Pipeline order (strictly sequential):
1) DQN training + zero-shot test
2) NEAT training + zero-shot test

Both pipelines write to one shared CSV with identical schema and an Is_Test flag.
"""

import os
import sys
import gc
import csv
import copy
import random
import pickle
import argparse
from datetime import datetime
from collections import deque

import numpy as np
import torch
import neat

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.Environment.PacManEnv import PacManEnv
from Code.Environment.CurriculumManager import CurriculumManager
from Code.Models.DQN.dqn_agent import DQNAgent
from Code.Models.DQN.action_masking_wrapper import DQNActionMaskingWrapper
from Code.Models.DQN.checkpoint_utils import save_checkpoint, load_checkpoint


# -----------------------------------------------------------------------------
# Seed bank (generalization framework)
# -----------------------------------------------------------------------------
TRAIN_SEEDS = list(range(0, 10000))
TEST_SEEDS = list(range(10000, 10100))


# -----------------------------------------------------------------------------
# Suite constants
# -----------------------------------------------------------------------------
MAX_EPISODES = 50_000
EARLY_STOP_STAGE = 5
EARLY_STOP_WINDOW = 20
EARLY_STOP_WIN_RATE = 0.85

DQN_REWARD_SCALE = 100.0
DQN_EPSILON_DECAY_SUITE = 6_000_000
DQN_EPSILON_DECAY_MIN_WIN_RATE = 0.20
DQN_TRAIN_RENDER_MODE = "human"

DQN_CHECKPOINT_DIR = os.path.join(_HERE, "Models", "DQN", "Checkpoints")
NEAT_CHECKPOINT_DIR = os.path.join(_HERE, "Models", "NEAT", "Checkpoints")
NEAT_CONFIG_PATH = os.path.join(_HERE, "Models", "NEAT", "neat_config.cfg")

DQN_CHAMPION_PATH = os.path.join(DQN_CHECKPOINT_DIR, "dqn_champion.pth")
NEAT_CHAMPION_PATH = os.path.join(NEAT_CHECKPOINT_DIR, "neat_champion.pkl")
DQN_SUITE_CHECKPOINT_PATH = os.path.join(DQN_CHECKPOINT_DIR, "dqn_suite_checkpoint.pt")
NEAT_SUITE_CHECKPOINT_PREFIX = os.path.join(NEAT_CHECKPOINT_DIR, "neat-suite-checkpoint-")
DQN_SAVE_EVERY_EPISODES = 200

SUITE_LOG_DIR = os.path.join(_HERE, "Models", "Suite", "CSV_History")
RUN_TIMESTAMP = datetime.now().strftime("%d-%m_%H-%M-%S")
SUITE_LOG_PATH = os.path.join(SUITE_LOG_DIR, f"train_suite_{RUN_TIMESTAMP}.csv")

CSV_HEADER = [
    "Algorithm", "Episode", "Stage", "Maze_Seed", "Reward", "Macro_Steps", "Micro_Ticks",
    "Outcome", "Win", "Epsilon", "Pellets", "Power_Pellets",
    "Ghosts", "Explore_Rate", "Avg_Loss",
    "Generation", "Best_Fitness", "Avg_Fitness", "Species_Count", "Eval_Seeds_Per_Genome", "Max_Episode_Steps",
    "Is_Test",
]


def _init_dirs() -> None:
    os.makedirs(SUITE_LOG_DIR, exist_ok=True)
    os.makedirs(DQN_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(NEAT_CHECKPOINT_DIR, exist_ok=True)


def _init_csv(path: str) -> None:
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)


def _append_csv(path: str, row: list) -> None:
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _print_success_banner(algorithm: str, stage: int, win_rate: float, episodes: int) -> None:
    print("\n" + "=" * 90)
    print(f"*** {algorithm} EARLY-STOP SUCCESS ***")
    print(f"Stage: {stage} | Rolling win-rate (last {EARLY_STOP_WINDOW}): {win_rate:.2%} | Episodes: {episodes}")
    print("Champion criteria reached: stopping training early and saving final model.")
    print("=" * 90 + "\n")


def _latest_checkpoint(prefix: str) -> str | None:
    """Return the newest checkpoint matching a NEAT filename prefix."""
    folder = os.path.dirname(prefix)
    stem = os.path.basename(prefix)
    if not os.path.isdir(folder):
        return None
    candidates = []
    for name in os.listdir(folder):
        if name.startswith(stem):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _validate_neat_schema(config: neat.Config, settings: dict) -> None:
    probe_settings = dict(settings)
    probe_settings["enable_sound"] = False
    probe_settings["max_episode_steps"] = None
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
            f"NEAT config num_outputs={config.genome_config.num_outputs} does not match PacManEnv action_dim={action_dim}."
        )


def _stage_settings(curriculum: CurriculumManager, stage: int) -> dict:
    stage_idx = max(0, min(stage, len(curriculum.stage_profiles) - 1))
    curriculum.current_stage = stage_idx
    settings = curriculum.get_settings()
    settings["max_episode_steps"] = None
    return settings


def _dqn_episode(
    agent: DQNAgent,
    settings: dict,
    maze_seed: int,
    is_test: bool,
    render_mode: str | None = None,
):
    run_settings = dict(settings)
    run_settings["enable_sound"] = False
    run_settings["max_episode_steps"] = None
    env = DQNActionMaskingWrapper(PacManEnv(render_mode=render_mode, obs_type="vector", settings=run_settings))
    state, _ = env.reset(seed=maze_seed)

    total_reward = 0.0
    macro_steps = 0
    micro_ticks = 0
    loss_values = []
    last_info = {}

    done = False
    while not done:
        valid_actions = env.get_valid_actions()

        if is_test:
            action = agent.select_action(state, valid_actions=valid_actions, return_exploration=False)
        else:
            policy_action, exploring = agent.select_action(
                state,
                valid_actions=valid_actions,
                return_exploration=True,
            )
            action = env.pick_action(policy_action, exploring=exploring)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        macro_steps += 1
        step_ticks = int(info.get("internal_ticks", 0)) if isinstance(info, dict) else 0
        micro_ticks += max(0, step_ticks)
        discount_pow = float(agent.gamma ** max(1, step_ticks))

        total_reward += float(reward)
        last_info = info if isinstance(info, dict) else {}

        if not is_test:
            next_valid_actions = env.get_valid_actions()
            next_valid_mask = np.zeros(agent.action_dim, dtype=np.float32)
            for a in next_valid_actions:
                if 0 <= int(a) < agent.action_dim:
                    next_valid_mask[int(a)] = 1.0

            scaled_reward = float(reward) / DQN_REWARD_SCALE
            agent.memory.push(
                state,
                int(action),
                scaled_reward,
                next_state,
                done,
                next_valid_mask=next_valid_mask,
                discount_pow=discount_pow,
            )

            loss = agent.optimize_model()
            if loss is not None:
                loss_values.append(float(loss))
                agent.update_target_network()

        state = next_state

    won = bool(env.unwrapped.engine.won)
    outcome = "WIN" if won else str(last_info.get("death_cause", "NONE"))
    avg_loss = float(np.mean(loss_values)) if loss_values else 0.0

    row_metrics = {
        "reward": float(total_reward),
        "macro_steps": int(macro_steps),
        "micro_ticks": int(micro_ticks),
        "outcome": outcome,
        "win": int(won),
        "pellets": int(last_info.get("pellets", 0)),
        "power_pellets": int(last_info.get("power_pellets", 0)),
        "ghosts": int(last_info.get("ghosts", 0)),
        "explore_rate": float(last_info.get("explore_rate", 0.0)),
        "avg_loss": avg_loss,
    }

    env.close()
    return row_metrics


def _neat_episode(net, settings: dict, maze_seed: int):
    run_settings = dict(settings)
    run_settings["enable_sound"] = False
    run_settings["max_episode_steps"] = None
    env = PacManEnv(render_mode=None, obs_type="vector", settings=run_settings)
    obs, _ = env.reset(seed=maze_seed)

    total_reward = 0.0
    macro_steps = 0
    micro_ticks = 0
    last_info = {}

    while True:
        outputs = net.activate(obs.tolist())
        valid_actions = env.get_valid_actions()

        # Continuous output -> discrete action via argmax over valid actions.
        if valid_actions:
            action = max(valid_actions, key=lambda a: outputs[a])
        else:
            action = int(np.argmax(outputs))

        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        macro_steps += 1
        if isinstance(info, dict):
            last_info = info
            micro_ticks += max(0, int(info.get("internal_ticks", 0)))

        if terminated or truncated:
            break

    won = bool(env.engine.won)
    outcome = "WIN" if won else str(last_info.get("death_cause", "NONE"))

    row_metrics = {
        "reward": float(total_reward),
        "macro_steps": int(macro_steps),
        "micro_ticks": int(micro_ticks),
        "outcome": outcome,
        "win": int(won),
        "pellets": int(last_info.get("pellets", 0)),
        "power_pellets": int(last_info.get("power_pellets", 0)),
        "ghosts": int(last_info.get("ghosts", 0)),
        "explore_rate": float(last_info.get("explore_rate", 0.0)),
    }

    env.close()
    return row_metrics


def run_dqn_pipeline(log_path: str, resume: bool = False, resume_path: str | None = None) -> str:
    print("\n=== DQN PIPELINE START ===")

    curriculum = CurriculumManager()
    rng = random.Random(12345)
    recent_wins = deque(maxlen=EARLY_STOP_WINDOW)

    agent = DQNAgent(input_dim=29, output_dim=4, epsilon_decay=DQN_EPSILON_DECAY_SUITE)

    start_episode = 0
    if resume:
        ckpt_path = resume_path or DQN_SUITE_CHECKPOINT_PATH
        if os.path.exists(ckpt_path):
            meta = load_checkpoint(ckpt_path, agent, curriculum=curriculum, map_location=agent.device)
            if meta.get("loaded"):
                start_episode = int(meta.get("episode", 0))
                print(f"[DQN] Resumed from {ckpt_path} at episode {start_episode}")
            else:
                print(f"[DQN] Resume requested but checkpoint incompatible: {meta.get('reason', 'unknown')}")
        else:
            print(f"[DQN] Resume requested but checkpoint not found: {ckpt_path}")

    final_episode = 0
    for episode in range(start_episode + 1, MAX_EPISODES + 1):
        final_episode = episode

        pre_window_win_rate = (sum(recent_wins) / len(recent_wins)) if recent_wins else 0.0
        freeze_decay_this_episode = pre_window_win_rate < DQN_EPSILON_DECAY_MIN_WIN_RATE
        step_count_before_episode = int(agent.step_count)
        epsilon_before_episode = float(agent.epsilon)

        settings = curriculum.get_settings()
        seed = int(rng.choice(TRAIN_SEEDS))
        settings["maze_seed"] = seed
        settings["max_episode_steps"] = None

        metrics = _dqn_episode(
            agent,
            settings=settings,
            maze_seed=seed,
            is_test=False,
            render_mode=DQN_TRAIN_RENDER_MODE,
        )

        if freeze_decay_this_episode:
            # Performance-gated decay: no epsilon decay while policy is still weak.
            agent.step_count = step_count_before_episode
            agent.epsilon = epsilon_before_episode

        won = bool(metrics["win"])
        curriculum.update_performance(won)
        promoted = curriculum.check_promotion()
        if promoted:
            agent.apply_exploration_jolt(min_epsilon=0.2, duration_steps=50_000)

        recent_wins.append(int(won))
        win_rate = (sum(recent_wins) / len(recent_wins)) if recent_wins else 0.0

        _append_csv(log_path, [
            "DQN", episode, curriculum.current_stage, seed, metrics["reward"], metrics["macro_steps"], metrics["micro_ticks"],
            metrics["outcome"], metrics["win"], float(agent.epsilon), metrics["pellets"], metrics["power_pellets"],
            metrics["ghosts"], metrics["explore_rate"], metrics["avg_loss"],
            -1, 0.0, 0.0, 0, 1, settings.get("max_episode_steps", None),
            False,
        ])

        print(f"[DQN] Stage={curriculum.current_stage} | Outcome={metrics['outcome']} | Episode={episode}")

        if (
            curriculum.current_stage >= EARLY_STOP_STAGE
            and len(recent_wins) == EARLY_STOP_WINDOW
            and win_rate >= EARLY_STOP_WIN_RATE
        ):
            _print_success_banner("DQN", curriculum.current_stage, win_rate, episode)
            break

        if episode % DQN_SAVE_EVERY_EPISODES == 0:
            save_checkpoint(
                DQN_SUITE_CHECKPOINT_PATH,
                agent,
                episode,
                curriculum=curriculum,
                include_curriculum=True,
            )

    torch.save(agent.policy_net.state_dict(), DQN_CHAMPION_PATH)
    print(f"[DQN] Champion saved -> {DQN_CHAMPION_PATH}")

    # Zero-shot test (stage 5 fixed, 100 held-out seeds).
    test_curriculum = CurriculumManager()
    test_settings = _stage_settings(test_curriculum, EARLY_STOP_STAGE)
    agent.epsilon = 0.0

    for i, seed in enumerate(TEST_SEEDS, start=1):
        test_settings_ep = dict(test_settings)
        test_settings_ep["maze_seed"] = int(seed)
        test_settings_ep["max_episode_steps"] = None

        metrics = _dqn_episode(
            agent,
            settings=test_settings_ep,
            maze_seed=int(seed),
            is_test=True,
            render_mode=None,
        )

        _append_csv(log_path, [
            "DQN", i, test_curriculum.current_stage, int(seed), metrics["reward"], metrics["macro_steps"], metrics["micro_ticks"],
            metrics["outcome"], metrics["win"], 0.0, metrics["pellets"], metrics["power_pellets"],
            metrics["ghosts"], metrics["explore_rate"], 0.0,
            -1, 0.0, 0.0, 0, 1, test_settings_ep.get("max_episode_steps", None),
            True,
        ])

        print(f"[DQN][TEST] Stage={test_curriculum.current_stage} | Outcome={metrics['outcome']} | Episode={i}")

    print(f"[DQN] Training episodes: {final_episode}, test episodes: {len(TEST_SEEDS)}")
    print("=== DQN PIPELINE END ===\n")
    return DQN_CHAMPION_PATH


def run_neat_pipeline(log_path: str, resume: bool = False, resume_path: str | None = None) -> str:
    print("\n=== NEAT PIPELINE START ===")

    if not os.path.exists(NEAT_CONFIG_PATH):
        raise FileNotFoundError(f"NEAT config not found at {NEAT_CONFIG_PATH}")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        NEAT_CONFIG_PATH,
    )

    curriculum = CurriculumManager()
    _validate_neat_schema(config, curriculum.get_settings())

    if resume:
        chosen = resume_path or _latest_checkpoint(NEAT_SUITE_CHECKPOINT_PREFIX)
        if chosen and os.path.exists(chosen):
            population = neat.Checkpointer.restore_checkpoint(chosen)
            print(f"[NEAT] Resumed from {chosen}")
        else:
            print("[NEAT] Resume requested but no suite checkpoint found; starting fresh.")
            population = neat.Population(config)
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats_reporter = neat.StatisticsReporter()
    population.add_reporter(stats_reporter)
    population.add_reporter(
        neat.Checkpointer(generation_interval=5, filename_prefix=NEAT_SUITE_CHECKPOINT_PREFIX)
    )

    rng = random.Random(67890)
    recent_wins = deque(maxlen=EARLY_STOP_WINDOW)

    episode_counter = 0
    early_stop = False
    champion_genome = None
    champion_fitness = float("-inf")
    current_generation = 0

    def eval_genomes(genomes, cfg):
        nonlocal episode_counter
        nonlocal early_stop
        nonlocal champion_genome
        nonlocal champion_fitness
        nonlocal current_generation

        gen_fitness = []
        species_count = len(getattr(population.species, "species", {}))

        for _, genome in genomes:
            if early_stop or episode_counter >= MAX_EPISODES:
                genome.fitness = -1e9
                continue

            settings = curriculum.get_settings()
            seed = int(rng.choice(TRAIN_SEEDS))
            settings["maze_seed"] = seed
            settings["max_episode_steps"] = None

            net = neat.nn.FeedForwardNetwork.create(genome, cfg)
            metrics = _neat_episode(net, settings=settings, maze_seed=seed)

            genome.fitness = float(metrics["reward"])
            gen_fitness.append(float(genome.fitness))
            episode_counter += 1

            won = bool(metrics["win"])
            curriculum.update_performance(won)
            curriculum.check_promotion()
            recent_wins.append(int(won))

            if genome.fitness > champion_fitness:
                champion_fitness = float(genome.fitness)
                champion_genome = copy.deepcopy(genome)

            best_fit = max(gen_fitness) if gen_fitness else float(genome.fitness)
            avg_fit = float(np.mean(gen_fitness)) if gen_fitness else float(genome.fitness)

            _append_csv(log_path, [
                "NEAT", episode_counter, curriculum.current_stage, seed, metrics["reward"], metrics["macro_steps"], metrics["micro_ticks"],
                metrics["outcome"], metrics["win"], 0.0, metrics["pellets"], metrics["power_pellets"],
                metrics["ghosts"], metrics["explore_rate"], 0.0,
                current_generation, best_fit, avg_fit, species_count, 1, settings.get("max_episode_steps", None),
                False,
            ])

            print(f"[NEAT] Stage={curriculum.current_stage} | Outcome={metrics['outcome']} | Episode={episode_counter}")

            win_rate = (sum(recent_wins) / len(recent_wins)) if recent_wins else 0.0
            if (
                curriculum.current_stage >= EARLY_STOP_STAGE
                and len(recent_wins) == EARLY_STOP_WINDOW
                and win_rate >= EARLY_STOP_WIN_RATE
            ):
                _print_success_banner("NEAT", curriculum.current_stage, win_rate, episode_counter)
                early_stop = True
                break

    while (not early_stop) and (episode_counter < MAX_EPISODES):
        current_generation = int(population.generation)
        population.run(eval_genomes, 1)

    if champion_genome is None:
        # Fallback: best known genome from statistics reporter.
        if stats_reporter.best_genome() is not None:
            champion_genome = copy.deepcopy(stats_reporter.best_genome())
        else:
            raise RuntimeError("NEAT did not produce a valid champion genome.")

    with open(NEAT_CHAMPION_PATH, "wb") as f:
        pickle.dump(champion_genome, f)
    print(f"[NEAT] Champion saved -> {NEAT_CHAMPION_PATH}")

    # Zero-shot test (stage 5 fixed, 100 held-out seeds).
    test_curriculum = CurriculumManager()
    test_settings = _stage_settings(test_curriculum, EARLY_STOP_STAGE)
    champion_net = neat.nn.FeedForwardNetwork.create(champion_genome, config)

    for i, seed in enumerate(TEST_SEEDS, start=1):
        test_settings_ep = dict(test_settings)
        test_settings_ep["maze_seed"] = int(seed)
        test_settings_ep["max_episode_steps"] = None

        metrics = _neat_episode(champion_net, settings=test_settings_ep, maze_seed=int(seed))

        _append_csv(log_path, [
            "NEAT", i, test_curriculum.current_stage, int(seed), metrics["reward"], metrics["macro_steps"], metrics["micro_ticks"],
            metrics["outcome"], metrics["win"], 0.0, metrics["pellets"], metrics["power_pellets"],
            metrics["ghosts"], metrics["explore_rate"], 0.0,
            current_generation, float(champion_fitness), float(champion_fitness),
            len(getattr(population.species, "species", {})), 1, test_settings_ep.get("max_episode_steps", None),
            True,
        ])

        print(f"[NEAT][TEST] Stage={test_curriculum.current_stage} | Outcome={metrics['outcome']} | Episode={i}")

    print(f"[NEAT] Training episodes: {episode_counter}, test episodes: {len(TEST_SEEDS)}")
    print("=== NEAT PIPELINE END ===\n")
    return NEAT_CHAMPION_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential DQN+NEAT dissertation training suite")
    parser.add_argument("--resume-dqn", action="store_true", help="Resume DQN from suite checkpoint")
    parser.add_argument("--resume-neat", action="store_true", help="Resume NEAT from latest suite checkpoint")
    parser.add_argument("--dqn-checkpoint", type=str, default=None, help="Optional explicit DQN checkpoint path")
    parser.add_argument("--neat-checkpoint", type=str, default=None, help="Optional explicit NEAT checkpoint path")
    args = parser.parse_args()

    _init_dirs()
    _init_csv(SUITE_LOG_PATH)

    print(f"Suite log: {SUITE_LOG_PATH}")

    # 1) DQN pipeline (must fully complete first)
    run_dqn_pipeline(
        SUITE_LOG_PATH,
        resume=bool(args.resume_dqn),
        resume_path=args.dqn_checkpoint,
    )

    # Clear memory before NEAT pipeline.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2) NEAT pipeline
    run_neat_pipeline(
        SUITE_LOG_PATH,
        resume=bool(args.resume_neat),
        resume_path=args.neat_checkpoint,
    )

    print("All pipelines complete.")


if __name__ == "__main__":
    main()

