import os
import csv
import glob
import time
import pickle
import sys
from datetime import datetime

import pandas as pd
import torch
import neat

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.Environment.CurriculumManager import CurriculumManager
from Code.Models.DQN.dqn_agent import DQNAgent
from Code.Models.DQN.checkpoint_utils import load_checkpoint
from Code import train_suite as ts


FIXED_TEST_EPISODES = 100


def _to_bool(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["1", "true", "t", "yes", "y"])


def _find_latest_suite_csv() -> str:
    files = glob.glob(os.path.join(ts.SUITE_LOG_DIR, "train_suite_*.csv"))
    # Retest CSVs contain only test rows, so they cannot provide reached training stages.
    files = [f for f in files if not os.path.basename(f).startswith("train_suite_fixed_retest_")]
    if not files:
        raise FileNotFoundError(
            f"No training suite CSV found in {ts.SUITE_LOG_DIR} (excluding fixed retest files)"
        )
    return max(files, key=os.path.getmtime)


def _latest_stage_for_algo_fixed(csv_path: str, algo: str) -> int:
    df = pd.read_csv(csv_path)
    if "Is_Test" in df.columns:
        df["Is_Test"] = _to_bool(df["Is_Test"])
    else:
        df["Is_Test"] = False

    if "Seed_Regime" not in df.columns:
        return 0

    d = df[
        (df["Algorithm"].astype(str).str.upper() == algo.upper())
        & (df["Is_Test"] == False)
        & (df["Seed_Regime"].astype(str).str.lower().str.startswith("fixed"))
    ].copy()
    if d.empty:
        return 0

    d["Episode"] = pd.to_numeric(d["Episode"], errors="coerce")
    d["Stage"] = pd.to_numeric(d["Stage"], errors="coerce")
    d = d.sort_values(["Episode", "Stage"], ascending=[True, True])
    return int(d["Stage"].iloc[-1])


def _init_csv(path: str) -> None:
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ts.CSV_HEADER)


def _append_csv(path: str, row: list) -> None:
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def run_dqn_tests_only(log_path: str, fixed_seed: int, n_episodes: int, reached_stage: int) -> None:
    print(f"[DQN][TEST-ONLY] Loading fixed champion/checkpoint (seed={fixed_seed})")

    curriculum = CurriculumManager(
        recent_window=150,
        promotion_threshold_stages_0_2=0.65,
        promotion_threshold_stages_3_5=0.70,
        promotion_threshold_stages_6_plus=0.65,
        tail_check_enabled=True,
        tail_check_size=5,
        tail_threshold_margin=0.05,
    )

    agent = DQNAgent(
        input_dim=29,
        output_dim=4,
        epsilon_start=ts.DQN_EPSILON_START_SUITE,
        epsilon_end=ts.DQN_EPSILON_END_SUITE,
        epsilon_decay=ts.DQN_EPSILON_DECAY_SUITE,
        use_amp=ts.DQN_USE_AMP_SUITE,
    )

    fixed_paths = ts._artifact_paths_for_regime(f"fixed_{int(fixed_seed)}")
    ckpt_path = fixed_paths["dqn_suite_checkpoint_path"]
    champion_path = fixed_paths["dqn_champion_path"]

    if os.path.exists(ckpt_path):
        meta = load_checkpoint(ckpt_path, agent, curriculum=curriculum, map_location=agent.device)
        print(f"[DQN][TEST-ONLY] Checkpoint load: {meta}")
    elif os.path.exists(champion_path):
        state = torch.load(champion_path, map_location=agent.device, weights_only=False)
        agent.policy_net.load_state_dict(state)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("[DQN][TEST-ONLY] Loaded champion weights only.")
    else:
        raise FileNotFoundError(f"No DQN fixed model found at {ckpt_path} or {champion_path}")

    agent.epsilon = 0.0
    test_modes = [
        ("reached_stage", int(reached_stage)),
        ("fixed_stage7", int(ts.EARLY_STOP_STAGE)),
    ]

    pipeline_start = time.perf_counter()
    for test_mode, stage in test_modes:
        test_curriculum = CurriculumManager(
            recent_window=150,
            promotion_threshold_stages_0_2=0.65,
            promotion_threshold_stages_3_5=0.70,
            promotion_threshold_stages_6_plus=0.65,
            tail_check_enabled=True,
            tail_check_size=5,
            tail_threshold_margin=0.05,
        )
        test_settings = ts._stage_settings(test_curriculum, stage)
        test_start = time.perf_counter()

        for i in range(1, n_episodes + 1):
            settings = dict(test_settings)
            settings["curriculum_stage"] = int(test_curriculum.current_stage)
            settings["maze_seed"] = int(fixed_seed)
            settings["max_episode_steps"] = None

            ep_start = time.perf_counter()
            metrics = ts._dqn_episode(
                agent,
                settings=settings,
                maze_seed=int(fixed_seed),
                is_test=True,
                render_mode=None,
            )
            episode_duration = time.perf_counter() - ep_start
            pipeline_elapsed = time.perf_counter() - pipeline_start
            test_elapsed = time.perf_counter() - test_start

            _append_csv(log_path, [
                "DQN", i, int(test_curriculum.current_stage), int(fixed_seed),
                metrics["reward"], metrics["macro_steps"], metrics["micro_ticks"],
                metrics["outcome"], metrics["win"], 0.0, metrics["pellets"], metrics["power_pellets"],
                metrics["ghosts"], metrics["explore_rate"], 0.0,
                -1, 0.0, 0.0, 0, 1, settings.get("max_episode_steps", None),
                episode_duration, pipeline_elapsed, test_elapsed,
                test_mode, f"fixed_{int(fixed_seed)}", True,
            ])

            print(f"[DQN][TEST-ONLY:{test_mode}] Episode={i} Outcome={metrics['outcome']}")


def run_neat_tests_only(log_path: str, fixed_seed: int, n_episodes: int, reached_stage: int) -> None:
    print(f"[NEAT][TEST-ONLY] Loading fixed champion (seed={fixed_seed})")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        ts.NEAT_CONFIG_PATH,
    )

    fixed_paths = ts._artifact_paths_for_regime(f"fixed_{int(fixed_seed)}")
    champion_path = fixed_paths["neat_champion_path"]
    if not os.path.exists(champion_path):
        raise FileNotFoundError(f"No NEAT fixed champion found at {champion_path}")

    with open(champion_path, "rb") as f:
        champion_genome = pickle.load(f)
    champion_net = neat.nn.FeedForwardNetwork.create(champion_genome, config)

    test_modes = [
        ("reached_stage", int(reached_stage)),
        ("fixed_stage7", int(ts.EARLY_STOP_STAGE)),
    ]

    pipeline_start = time.perf_counter()
    for test_mode, stage in test_modes:
        test_curriculum = CurriculumManager(
            recent_window=50,
            promotion_threshold_all_stages=0.55,
            tail_check_enabled=True,
            tail_check_size=10,
            tail_threshold_margin=0.05,
        )
        test_settings = ts._stage_settings(test_curriculum, stage)
        test_start = time.perf_counter()

        for i in range(1, n_episodes + 1):
            settings = dict(test_settings)
            settings["curriculum_stage"] = int(test_curriculum.current_stage)
            settings["maze_seed"] = int(fixed_seed)
            settings["max_episode_steps"] = None

            ep_start = time.perf_counter()
            metrics = ts._neat_episode(champion_net, settings=settings, maze_seed=int(fixed_seed))
            episode_duration = time.perf_counter() - ep_start
            pipeline_elapsed = time.perf_counter() - pipeline_start
            test_elapsed = time.perf_counter() - test_start

            _append_csv(log_path, [
                "NEAT", i, int(test_curriculum.current_stage), int(fixed_seed),
                metrics["reward"], metrics["macro_steps"], metrics["micro_ticks"],
                metrics["outcome"], metrics["win"], 0.0, metrics["pellets"], metrics["power_pellets"],
                metrics["ghosts"], metrics["explore_rate"], 0.0,
                -1, 0.0, 0.0, 0, 1, settings.get("max_episode_steps", None),
                episode_duration, pipeline_elapsed, test_elapsed,
                test_mode, f"fixed_{int(fixed_seed)}", True,
            ])

            print(f"[NEAT][TEST-ONLY:{test_mode}] Episode={i} Outcome={metrics['outcome']}")


def main() -> None:
    fixed_seed = int(ts.DEFAULT_FIXED_BENCHMARK_SEED)
    n = int(FIXED_TEST_EPISODES)

    os.makedirs(ts.SUITE_LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%d-%m_%H-%M-%S")
    out_csv = os.path.join(ts.SUITE_LOG_DIR, f"train_suite_fixed_retest_{stamp}.csv")
    _init_csv(out_csv)

    src_csv = _find_latest_suite_csv()
    dqn_stage = _latest_stage_for_algo_fixed(src_csv, "DQN")
    neat_stage = _latest_stage_for_algo_fixed(src_csv, "NEAT")

    print(f"[TEST-ONLY] Source CSV: {src_csv}")
    print(f"[TEST-ONLY] Reached stages -> DQN: {dqn_stage}, NEAT: {neat_stage}")
    print(f"[TEST-ONLY] Output CSV: {out_csv}")

    run_dqn_tests_only(out_csv, fixed_seed=fixed_seed, n_episodes=n, reached_stage=dqn_stage)
    run_neat_tests_only(out_csv, fixed_seed=fixed_seed, n_episodes=n, reached_stage=neat_stage)

    print("[TEST-ONLY] Done.")


if __name__ == "__main__":
    main()

