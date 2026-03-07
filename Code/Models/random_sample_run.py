import time
import os
from Code.PacManEnv import PacManEnv
from Code.Settings  import Settings

_HERE     = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_HERE, "..")   # Models/ is inside Code/
_settings = Settings(os.path.join(_CODE_DIR, "game_settings.json")).get_all()
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

    Layout (40 values):
      [0-3]   PacMan pos (norm) + direction
      [4-27]  Ghosts × 4  (rel_x, rel_y, dist, dir_x, dir_y, threat)
      [28-29] Nearest pellet rel_x, rel_y
      [30-33] Wall sensors  (up, down, left, right)
      [34]    Pellet ratio eaten
      [35]    Frightened active
      [36]    Frightened timer ratio
      [37]    Lives ratio
      [38]    Scatter mode
      [39]    Nearest power pellet distance (normalised)
    """
    print("\n── Observation ──────────────────────────────────────────")
    print(f"  PacMan pos   : ({obs[0]:.3f}, {obs[1]:.3f})  dir=({obs[2]:.0f}, {obs[3]:.0f})")
    ghost_names = ["Blinky", "Pinky", "Inky", "Clyde"]
    for i, name in enumerate(ghost_names):
        base = 4 + i * 6
        print(f"  {name:<7} rel=({obs[base]:.3f}, {obs[base+1]:.3f})  "
              f"dist={obs[base+2]:.3f}  "
              f"dir=({obs[base+3]:.0f}, {obs[base+4]:.0f})  "
              f"threat={obs[base+5]:.0f}")
    print(f"  Nearest pellet       rel=({obs[28]:.3f}, {obs[29]:.3f})")
    print(f"  Walls        : up={obs[30]:.0f}  down={obs[31]:.0f}  left={obs[32]:.0f}  right={obs[33]:.0f}")
    print(f"  Pellet ratio : {obs[34]:.3f}  Frightened={obs[35]:.0f}  frit_timer={obs[36]:.3f}")
    print(f"  Lives ratio  : {obs[37]:.3f}  Scatter={obs[38]:.0f}  pp_dist={obs[39]:.3f}")
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