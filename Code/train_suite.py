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
import time
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
DEFAULT_FIXED_BENCHMARK_SEED = 22459265


# -----------------------------------------------------------------------------
# Suite constants
# -----------------------------------------------------------------------------
MAX_EPISODES = 100_000
EARLY_STOP_STAGE = 7
EARLY_STOP_WINDOW = 20
EARLY_STOP_WIN_RATE = 0.85

DQN_REWARD_SCALE = 100.0
DQN_EPSILON_START_SUITE = 1.0
DQN_EPSILON_END_SUITE = 0.15
DQN_EPSILON_DECAY_SUITE = 1_000_000
DQN_TRAIN_RENDER_MODE = None

NEAT_GHOST_FITNESS_WEIGHT = 150.0
NEAT_EVAL_SEEDS_EARLY = 3
NEAT_EVAL_SEEDS_DEFAULT = 5
NEAT_MULTI_SEED_FROM_STAGE = 3

DQN_CHECKPOINT_DIR = os.path.join(_HERE, "Models", "DQN", "Checkpoints")
NEAT_CHECKPOINT_DIR = os.path.join(_HERE, "Models", "NEAT", "Checkpoints")
NEAT_CONFIG_PATH = os.path.join(_HERE, "Models", "NEAT", "neat_config.cfg")

DQN_CHAMPION_PATH = os.path.join(DQN_CHECKPOINT_DIR, "dqn_champion.pth")
NEAT_CHAMPION_PATH = os.path.join(NEAT_CHECKPOINT_DIR, "neat_champion.pkl")
DQN_SUITE_CHECKPOINT_PATH = os.path.join(DQN_CHECKPOINT_DIR, "dqn_suite_checkpoint.pt")
NEAT_SUITE_CHECKPOINT_PREFIX = os.path.join(NEAT_CHECKPOINT_DIR, "neat-suite-checkpoint-")
DQN_SAVE_EVERY_EPISODES = 200

SUITE_LOG_DIR = os.path.join(_HERE, "Models", "Suite", "CSV_History_SchemaV2")
RUN_TIMESTAMP = datetime.now().strftime("%d-%m_%H-%M-%S")
SUITE_LOG_PATH = os.path.join(SUITE_LOG_DIR, f"train_suite_{RUN_TIMESTAMP}.csv")

CSV_HEADER = [
    "Algorithm", "Episode", "Stage", "Maze_Seed", "Reward", "Macro_Steps", "Micro_Ticks",
    "Outcome", "Win", "Epsilon", "Pellets", "Power_Pellets",
    "Ghosts", "Explore_Rate", "Avg_Loss",
    "Generation", "Best_Fitness", "Avg_Fitness", "Species_Count", "Eval_Seeds_Per_Genome", "Max_Episode_Steps",
    "Episode_Duration_Sec", "Pipeline_Elapsed_Sec", "Test_Run_Elapsed_Sec",
    "Test_Mode", "Seed_Regime",
    "Is_Test",
]


def _build_seed_regimes(seed_regime_arg: str, fixed_seed: int) -> list[dict]:
    random_regime = {
        "name": "random",
        "train_seeds": list(TRAIN_SEEDS),
        "test_seeds": list(TEST_SEEDS),
        "is_fixed": False,
    }
    fixed_regime = {
        "name": f"fixed_{int(fixed_seed)}",
        "train_seeds": [int(fixed_seed)],
        "test_seeds": [int(fixed_seed)],
        "is_fixed": True,
    }

    if seed_regime_arg == "random":
        return [random_regime]
    if seed_regime_arg == "fixed":
        return [fixed_regime]
    return [random_regime, fixed_regime]


def _artifact_paths_for_regime(regime_name: str) -> dict:
    safe_regime = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(regime_name).lower())
    return {
        "dqn_champion_path": os.path.join(DQN_CHECKPOINT_DIR, f"dqn_champion_{safe_regime}.pth"),
        "neat_champion_path": os.path.join(NEAT_CHECKPOINT_DIR, f"neat_champion_{safe_regime}.pkl"),
        "dqn_suite_checkpoint_path": os.path.join(DQN_CHECKPOINT_DIR, f"dqn_suite_checkpoint_{safe_regime}.pt"),
        "neat_suite_checkpoint_prefix": os.path.join(NEAT_CHECKPOINT_DIR, f"neat-suite-checkpoint-{safe_regime}-"),
    }


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
    settings["curriculum_stage"] = int(stage_idx)
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


def run_dqn_pipeline(
    log_path: str,
    train_seeds: list[int],
    test_seeds: list[int],
    seed_regime_name: str,
    dqn_champion_path: str,
    dqn_suite_checkpoint_path: str,
    resume: bool = False,
    resume_path: str | None = None,
) -> str:
    print(f"\n=== DQN PIPELINE START ({seed_regime_name}) ===")

    curriculum = CurriculumManager(
        recent_window=150,
        promotion_threshold_stages_0_2=0.65,
        promotion_threshold_stages_3_5=0.70,
        promotion_threshold_stages_6_plus=0.65,
        tail_check_enabled=True,
        tail_check_size=5,
        tail_threshold_margin=0.05,
    )
    rng = random.Random(12345)
    recent_wins = deque(maxlen=EARLY_STOP_WINDOW)

    agent = DQNAgent(
        input_dim=29,
        output_dim=4,
        epsilon_start=DQN_EPSILON_START_SUITE,
        epsilon_end=DQN_EPSILON_END_SUITE,
        epsilon_decay=DQN_EPSILON_DECAY_SUITE,
    )

    start_episode = 0
    if resume:
        ckpt_path = resume_path or dqn_suite_checkpoint_path
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
    pipeline_start = time.perf_counter()
    for episode in range(start_episode + 1, MAX_EPISODES + 1):
        final_episode = episode

        settings = curriculum.get_settings()
        settings["curriculum_stage"] = int(curriculum.current_stage)
        seed = int(rng.choice(train_seeds))
        settings["maze_seed"] = seed
        settings["max_episode_steps"] = None

        episode_start = time.perf_counter()
        metrics = _dqn_episode(
            agent,
            settings=settings,
            maze_seed=seed,
            is_test=False,
            render_mode=DQN_TRAIN_RENDER_MODE,
        )
        episode_duration = time.perf_counter() - episode_start
        pipeline_elapsed = time.perf_counter() - pipeline_start


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
            episode_duration, pipeline_elapsed, 0.0,
            "train",
            seed_regime_name,
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
                dqn_suite_checkpoint_path,
                agent,
                episode,
                curriculum=curriculum,
                include_curriculum=True,
            )

    torch.save(agent.policy_net.state_dict(), dqn_champion_path)
    print(f"[DQN] Champion saved -> {dqn_champion_path}")

    # Zero-shot tests:
    #  1) reached_stage: stage the agent actually reached in training
    #  2) fixed_stage7: fixed hardest-stage challenge
    agent.epsilon = 0.0
    dqn_final_stage = int(curriculum.current_stage)
    dqn_test_modes = [
        ("reached_stage", dqn_final_stage),
        ("fixed_stage7", EARLY_STOP_STAGE),
    ]
    total_test_episodes = len(test_seeds) * len(dqn_test_modes)
    total_test_start = time.perf_counter()
    test_elapsed = 0.0

    for test_mode, test_stage in dqn_test_modes:
        test_curriculum = CurriculumManager(
            recent_window=150,
            promotion_threshold_stages_0_2=0.65,
            promotion_threshold_stages_3_5=0.70,
            promotion_threshold_stages_6_plus=0.65,
            tail_check_enabled=True,
            tail_check_size=5,
            tail_threshold_margin=0.05,
        )
        test_settings = _stage_settings(test_curriculum, test_stage)
        test_start = time.perf_counter()

        for i, seed in enumerate(test_seeds, start=1):
            test_settings_ep = dict(test_settings)
            test_settings_ep["curriculum_stage"] = int(test_curriculum.current_stage)
            test_settings_ep["maze_seed"] = int(seed)
            test_settings_ep["max_episode_steps"] = None

            episode_start = time.perf_counter()
            metrics = _dqn_episode(
                agent,
                settings=test_settings_ep,
                maze_seed=int(seed),
                is_test=True,
                render_mode=None,
            )
            episode_duration = time.perf_counter() - episode_start
            pipeline_elapsed = time.perf_counter() - pipeline_start
            test_elapsed = time.perf_counter() - test_start

            _append_csv(log_path, [
                "DQN", i, test_curriculum.current_stage, int(seed), metrics["reward"], metrics["macro_steps"], metrics["micro_ticks"],
                metrics["outcome"], metrics["win"], 0.0, metrics["pellets"], metrics["power_pellets"],
                metrics["ghosts"], metrics["explore_rate"], 0.0,
                -1, 0.0, 0.0, 0, 1, test_settings_ep.get("max_episode_steps", None),
                episode_duration, pipeline_elapsed, test_elapsed,
                test_mode,
                seed_regime_name,
                True,
            ])

            print(f"[DQN][TEST:{test_mode}] Stage={test_curriculum.current_stage} | Outcome={metrics['outcome']} | Episode={i}")

    total_test_elapsed = time.perf_counter() - total_test_start
    print(f"[DQN] Training episodes: {final_episode}, test episodes: {total_test_episodes}")
    print(f"[DQN] Test run elapsed: {total_test_elapsed:.2f}s")
    print(f"=== DQN PIPELINE END ({seed_regime_name}) ===\n")
    return dqn_champion_path


def run_neat_pipeline(
    log_path: str,
    train_seeds: list[int],
    test_seeds: list[int],
    seed_regime_name: str,
    neat_champion_path: str,
    neat_suite_checkpoint_prefix: str,
    resume: bool = False,
    resume_path: str | None = None,
) -> str:
    print(f"\n=== NEAT PIPELINE START ({seed_regime_name}) ===")

    if not os.path.exists(NEAT_CONFIG_PATH):
        raise FileNotFoundError(f"NEAT config not found at {NEAT_CONFIG_PATH}")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        NEAT_CONFIG_PATH,
    )

    curriculum = CurriculumManager(
        recent_window=50,
        promotion_threshold_all_stages=0.55,
        tail_check_enabled=True,
        tail_check_size=10,
        tail_threshold_margin=0.05,
    )
    _validate_neat_schema(config, curriculum.get_settings())

    if resume:
        chosen = resume_path or _latest_checkpoint(neat_suite_checkpoint_prefix)
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
        neat.Checkpointer(generation_interval=5, filename_prefix=neat_suite_checkpoint_prefix)
    )

    rng = random.Random(67890)
    recent_wins = deque(maxlen=EARLY_STOP_WINDOW)

    episode_counter = 0
    early_stop = False
    champion_genome = None
    champion_fitness = float("-inf")
    current_generation = 0
    pipeline_start = time.perf_counter()

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

            stage_before = int(curriculum.current_stage)
            eval_seeds = (
                NEAT_EVAL_SEEDS_EARLY
                if stage_before < NEAT_MULTI_SEED_FROM_STAGE
                else NEAT_EVAL_SEEDS_DEFAULT
            )
            if len(train_seeds) == 1:
                # In fixed-seed benchmarking, repeated seed evaluations add little signal.
                eval_seeds = 1
            eval_records = []
            net = neat.nn.FeedForwardNetwork.create(genome, cfg)

            for _ in range(eval_seeds):
                if early_stop or episode_counter >= MAX_EPISODES:
                    break

                settings = curriculum.get_settings()
                settings["curriculum_stage"] = int(stage_before)
                seed = int(rng.choice(train_seeds))
                settings["maze_seed"] = seed
                settings["max_episode_steps"] = None

                episode_start = time.perf_counter()
                metrics = _neat_episode(net, settings=settings, maze_seed=seed)
                episode_duration = time.perf_counter() - episode_start
                pipeline_elapsed = time.perf_counter() - pipeline_start

                eval_records.append((metrics, seed, stage_before, episode_duration, pipeline_elapsed, settings))
                episode_counter += 1

                won = bool(metrics["win"])
                curriculum.update_performance(won)
                curriculum.check_promotion()
                recent_wins.append(int(won))

                win_rate = (sum(recent_wins) / len(recent_wins)) if recent_wins else 0.0
                if (
                    curriculum.current_stage >= EARLY_STOP_STAGE
                    and len(recent_wins) == EARLY_STOP_WINDOW
                    and win_rate >= EARLY_STOP_WIN_RATE
                ):
                    _print_success_banner("NEAT", curriculum.current_stage, win_rate, episode_counter)
                    early_stop = True
                    break

            if not eval_records:
                genome.fitness = -1e9
                continue

            mean_reward = float(np.mean([rec[0]["reward"] for rec in eval_records]))
            mean_ghosts = float(np.mean([rec[0]["ghosts"] for rec in eval_records]))
            genome.fitness = mean_reward + (mean_ghosts * NEAT_GHOST_FITNESS_WEIGHT)
            gen_fitness.append(float(genome.fitness))

            if genome.fitness > champion_fitness:
                champion_fitness = float(genome.fitness)
                champion_genome = copy.deepcopy(genome)

            best_fit = max(gen_fitness) if gen_fitness else float(genome.fitness)
            avg_fit = float(np.mean(gen_fitness)) if gen_fitness else float(genome.fitness)

            rep_metrics, rep_seed, rep_stage, rep_duration, rep_elapsed, rep_settings = eval_records[-1]
            _append_csv(log_path, [
                "NEAT", episode_counter, rep_stage, rep_seed, rep_metrics["reward"], rep_metrics["macro_steps"], rep_metrics["micro_ticks"],
                rep_metrics["outcome"], rep_metrics["win"], 0.0, rep_metrics["pellets"], rep_metrics["power_pellets"],
                rep_metrics["ghosts"], rep_metrics["explore_rate"], 0.0,
                current_generation, best_fit, avg_fit, species_count, len(eval_records), rep_settings.get("max_episode_steps", None),
                rep_duration, rep_elapsed, 0.0,
                "train",
                seed_regime_name,
                False,
            ])

            print(
                f"[NEAT] Stage={rep_stage} | Outcome={rep_metrics['outcome']} "
                f"| Episode={episode_counter} | EvalSeeds={len(eval_records)}"
            )

            if early_stop:
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

    with open(neat_champion_path, "wb") as f:
        pickle.dump(champion_genome, f)
    print(f"[NEAT] Champion saved -> {neat_champion_path}")

    # Zero-shot tests:
    #  1) reached_stage: stage the agent actually reached in training
    #  2) fixed_stage7: fixed hardest-stage challenge
    champion_net = neat.nn.FeedForwardNetwork.create(champion_genome, config)
    neat_final_stage = int(curriculum.current_stage)
    neat_test_modes = [
        ("reached_stage", neat_final_stage),
        ("fixed_stage7", EARLY_STOP_STAGE),
    ]
    total_test_episodes = len(test_seeds) * len(neat_test_modes)
    total_test_start = time.perf_counter()
    test_elapsed = 0.0

    for test_mode, test_stage in neat_test_modes:
        test_curriculum = CurriculumManager(
            recent_window=50,
            promotion_threshold_all_stages=0.55,
            tail_check_enabled=True,
            tail_check_size=10,
            tail_threshold_margin=0.05,
        )
        test_settings = _stage_settings(test_curriculum, test_stage)
        test_start = time.perf_counter()

        for i, seed in enumerate(test_seeds, start=1):
            test_settings_ep = dict(test_settings)
            test_settings_ep["curriculum_stage"] = int(test_curriculum.current_stage)
            test_settings_ep["maze_seed"] = int(seed)
            test_settings_ep["max_episode_steps"] = None

            episode_start = time.perf_counter()
            metrics = _neat_episode(champion_net, settings=test_settings_ep, maze_seed=int(seed))
            episode_duration = time.perf_counter() - episode_start
            pipeline_elapsed = time.perf_counter() - pipeline_start
            test_elapsed = time.perf_counter() - test_start

            _append_csv(log_path, [
                "NEAT", i, test_curriculum.current_stage, int(seed), metrics["reward"], metrics["macro_steps"], metrics["micro_ticks"],
                metrics["outcome"], metrics["win"], 0.0, metrics["pellets"], metrics["power_pellets"],
                metrics["ghosts"], metrics["explore_rate"], 0.0,
                current_generation, float(champion_fitness), float(champion_fitness),
                len(getattr(population.species, "species", {})), 1, test_settings_ep.get("max_episode_steps", None),
                episode_duration, pipeline_elapsed, test_elapsed,
                test_mode,
                seed_regime_name,
                True,
            ])

            print(f"[NEAT][TEST:{test_mode}] Stage={test_curriculum.current_stage} | Outcome={metrics['outcome']} | Episode={i}")

    total_test_elapsed = time.perf_counter() - total_test_start
    print(f"[NEAT] Training episodes: {episode_counter}, test episodes: {total_test_episodes}")
    print(f"[NEAT] Test run elapsed: {total_test_elapsed:.2f}s")
    print(f"=== NEAT PIPELINE END ({seed_regime_name}) ===\n")
    return neat_champion_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential DQN+NEAT dissertation training suite")
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["dqn", "neat", "both"],
        default="both",
        help="Which pipeline(s) to run: dqn, neat, or both (default).",
    )
    parser.add_argument("--resume-dqn", action="store_true", help="Resume DQN from suite checkpoint")
    parser.add_argument("--resume-neat", action="store_true", help="Resume NEAT from latest suite checkpoint")
    parser.add_argument("--dqn-checkpoint", type=str, default=None, help="Optional explicit DQN checkpoint path")
    parser.add_argument("--neat-checkpoint", type=str, default=None, help="Optional explicit NEAT checkpoint path")
    parser.add_argument(
        "--seed-regime",
        type=str,
        choices=["random", "fixed", "both"],
        default="both",
        help="Seed benchmark mode: random train/test splits, fixed single seed, or both sequentially.",
    )
    parser.add_argument(
        "--fixed-seed",
        type=int,
        default=DEFAULT_FIXED_BENCHMARK_SEED,
        help="Single seed used for fixed-seed benchmarking (train and test).",
    )
    args = parser.parse_args()

    _init_dirs()
    _init_csv(SUITE_LOG_PATH)

    print(f"Suite log: {SUITE_LOG_PATH}")

    run_dqn = args.pipeline in ("dqn", "both")
    run_neat = args.pipeline in ("neat", "both")

    seed_regimes = _build_seed_regimes(args.seed_regime, args.fixed_seed)
    for regime in seed_regimes:
        regime_name = str(regime["name"])
        train_seeds = [int(s) for s in regime["train_seeds"]]
        test_seeds = [int(s) for s in regime["test_seeds"]]
        artifact_paths = _artifact_paths_for_regime(regime_name)

        print("\n" + "-" * 90)
        print(f"Seed regime: {regime_name}")
        print(f"Train seeds: {train_seeds if len(train_seeds) <= 5 else f'{len(train_seeds)} seeds'}")
        print(f"Test seeds:  {test_seeds if len(test_seeds) <= 5 else f'{len(test_seeds)} seeds'}")
        print("-" * 90)

        if run_dqn:
            run_dqn_pipeline(
                SUITE_LOG_PATH,
                train_seeds=train_seeds,
                test_seeds=test_seeds,
                seed_regime_name=regime_name,
                dqn_champion_path=artifact_paths["dqn_champion_path"],
                dqn_suite_checkpoint_path=artifact_paths["dqn_suite_checkpoint_path"],
                resume=bool(args.resume_dqn),
                resume_path=args.dqn_checkpoint,
            )

        # Clear memory only when transitioning DQN -> NEAT in combined runs.
        if run_dqn and run_neat:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if run_neat:
            run_neat_pipeline(
                SUITE_LOG_PATH,
                train_seeds=train_seeds,
                test_seeds=test_seeds,
                seed_regime_name=regime_name,
                neat_champion_path=artifact_paths["neat_champion_path"],
                neat_suite_checkpoint_prefix=artifact_paths["neat_suite_checkpoint_prefix"],
                resume=bool(args.resume_neat),
                resume_path=args.neat_checkpoint,
            )

    print("All pipelines complete.")


if __name__ == "__main__":
    main()

