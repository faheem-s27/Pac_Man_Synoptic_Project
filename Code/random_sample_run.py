import time
import os
from Code.PacManEnv import PacManEnv
from Code.Settings  import Settings

_HERE     = os.path.dirname(os.path.abspath(__file__))
_settings = Settings(os.path.join(_HERE, "game_settings.json")).get_all()
MAZE_SEED = _settings.get("maze_seed", None)

ACTION_NAMES = {0: "NOOP", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}


env = PacManEnv(render_mode="human", maze_algorithm="recursive_backtracking", maze_seed=MAZE_SEED)
obs, info = env.reset()

print("=" * 60)
print("RANDOM AGENT — observation debug run")
print(f"Observation space shape : {env.observation_space.shape}")
print(f"Action space            : {env.action_space.n} discrete actions")
print("=" * 60)

def print_obs(obs):
    """Pretty-print the observation vector.

    Layout (34 values):
      [0-1]   PacMan pos (norm)
      [2-3]   PacMan direction
      [4-19]  Ghosts × 4  (rel_x, rel_y, dist, threat)
      [20-21] Nearest pellet rel_x, rel_y
      [22-25] Wall sensors  (up, down, left, right)
      [26]    Pellet ratio eaten
      [27]    Frightened active
      [28]    Frightened timer ratio
      [29]    Lives ratio
      [30]    Scatter mode
      [31-32] Nearest power pellet rel_x, rel_y
      [33]    Power pellets remaining (normalised)
    """
    print("\n── Observation ──────────────────────────────────────────")
    print(f"  PacMan pos   : ({obs[0]:.3f}, {obs[1]:.3f})  dir=({obs[2]:.0f}, {obs[3]:.0f})")
    ghost_names = ["Blinky", "Pinky", "Inky", "Clyde"]
    for i, name in enumerate(ghost_names):
        base = 4 + i * 4
        print(f"  {name:<7} rel=({obs[base]:.3f}, {obs[base+1]:.3f})  "
              f"dist={obs[base+2]:.3f}  threat={obs[base+3]:.0f}")
    print(f"  Nearest pellet       rel=({obs[20]:.3f}, {obs[21]:.3f})")
    print(f"  Nearest power pellet rel=({obs[31]:.3f}, {obs[32]:.3f})  "
          f"remaining={obs[33]:.3f}")
    print(f"  Walls        : up={obs[22]:.0f}  down={obs[23]:.0f}  left={obs[24]:.0f}  right={obs[25]:.0f}")
    print(f"  Pellet ratio : {obs[26]:.3f}  Frightened={obs[27]:.0f}  frit_timer={obs[28]:.3f}")
    print(f"  Lives ratio  : {obs[29]:.3f}  Scatter={obs[30]:.0f}")
    print("─────────────────────────────────────────────────────────")

step = 0
total_reward = 0.0
PRINT_EVERY = 60   # print obs + cumulative reward every N steps

try:
    while True:
        action = env.action_space.sample()   # purely random

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Every PRINT_EVERY steps — show full obs + running totals
        if step % PRINT_EVERY == 0:
            print_obs(obs)
            print(f"  → Step {step}  action={ACTION_NAMES[action]:<5}  "
                  f"reward={reward:+.2f}  cumulative={total_reward:+.1f}  "
                  f"score={info['score']}  lives={info['lives']}  level={info['level']}")

        # Any step where something interesting happened (non-trivial reward)
        elif abs(reward) > 0.1:
            print(f"  [step {step:>5}] action={ACTION_NAMES[action]:<5}  "
                  f"reward={reward:+.1f}  cumulative={total_reward:+.1f}  "
                  f"score={info['score']}  lives={info['lives']}")

        if terminated or truncated:
            status = "GAME OVER" if info.get("game_over") else "TRUNCATED"
            print(f"\nEpisode ended ({status}) after {step} steps | "
                  f"Mazes cleared: {info['levels_completed']} | "
                  f"Final score: {info['score']} | "
                  f"Total reward: {total_reward:+.1f}")
            break

        time.sleep(0.01)

finally:
    env.close()