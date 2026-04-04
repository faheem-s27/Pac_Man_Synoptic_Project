import time
import os
import numpy as np
from Code.PacManEnv import PacManEnv
from Code.Settings  import Settings

_HERE     = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_HERE, "..")   # Models/ is inside Code/
_settings = Settings(os.path.join(_CODE_DIR, "game_settings.json")).get_all()
MAZE_SEED = _settings.get("maze_seed", None)

ACTION_NAMES = {
    PacManEnv.FORWARD: "FORWARD",
    PacManEnv.LEFT: "LEFT",
    PacManEnv.RIGHT: "RIGHT",
    PacManEnv.BACKWARD: "BACKWARD",
}
RAY_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "UP-LEFT", "UP-RIGHT", "DOWN-LEFT", "DOWN-RIGHT"]

env = PacManEnv(render_mode="human", maze_algorithm="recursive_backtracking", maze_seed=MAZE_SEED)
obs, info = env.reset()

print("=" * 60)
print("RANDOM AGENT — observation debug run")
print(f"Observation space shape : {env.observation_space.shape}")
print(f"Action space            : {env.action_space.n} discrete actions")
print("=" * 60)

def print_obs(obs):
    """Pretty-print the observation vector.

    Layout (36 values, all in [-1, 1]):
      [0..31] 8 egocentric rays x 4 channels: [wall, food, power, ghost_signal]
      [32]    nearest food BFS (normalised)
      [33]    nearest dangerous ghost BFS (normalised)
      [34]    is_powered
      [35]    power_time_remaining
    """
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    print("\n── Observation ──────────────────────────────────────────")
    if obs.shape[0] < 36:
        print(f"  Unexpected observation size: {obs.shape[0]} (expected 36)")
        print("─────────────────────────────────────────────────────────")
        return

    for i, ray_name in enumerate(RAY_NAMES):
        b = i * 4
        w_d, f_d, p_d, g_d = obs[b:b + 4]
        print(
            f"  Ray {i} ({ray_name:<9})  "
            f"W={w_d:+.3f}  F={f_d:+.3f}  P={p_d:+.3f}  G={g_d:+.3f}"
        )

    print(f"  Near food BFS        : {obs[32]:+.3f}")
    print(f"  Near danger BFS      : {obs[33]:+.3f}")
    print(f"  Powered              : {obs[34]:+.3f}")
    print(f"  Power time remaining : {obs[35]:+.3f}")
    print("─────────────────────────────────────────────────────────")

step = 0
total_reward = 0.0
PRINT_EVERY = 60   # print obs + cumulative reward every N steps

try:
    while True:
        valid_actions = env.get_valid_actions()
        action = int(np.random.choice(valid_actions)) if valid_actions else env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Every PRINT_EVERY steps — show full obs + running totals
        if step % PRINT_EVERY == 0:
            print_obs(obs)
            print(f"  → Step {step}  action={ACTION_NAMES.get(action, str(action)):<8}  "
                  f"reward={reward:+.2f}  cumulative={total_reward:+.1f}  "
                  f"score={info.get('score', 0)}  pellets={info.get('pellets', 0)}  "
                  f"power={info.get('power_pellets', 0)}  ghosts={info.get('ghosts', 0)}  "
                  f"steps={info.get('steps', step)}")

        # Any step where something interesting happened (non-trivial reward)
        elif abs(reward) > 0.1:
            print(f"  [step {step:>5}] action={ACTION_NAMES.get(action, str(action)):<8}  "
                  f"reward={reward:+.1f}  cumulative={total_reward:+.1f}  "
                  f"score={info.get('score', 0)}")

        if terminated or truncated:
            status = info.get("death_cause", "TRUNCATED" if truncated else "TERMINATED")
            print(f"\nEpisode ended ({status}) after {step} steps | "
                  f"Final score: {info.get('score', 0)} | "
                  f"Pellets: {info.get('pellets', 0)} | "
                  f"Power: {info.get('power_pellets', 0)} | "
                  f"Ghosts: {info.get('ghosts', 0)} | "
                  f"Total reward: {total_reward:+.1f}")
            break

        time.sleep(0.01)

finally:
    env.close()