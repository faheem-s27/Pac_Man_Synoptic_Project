import time
from Code.PacManEnv import PacManEnv


def get_strategic_action(obs):
    """
    Decodes the observation vector.
    Prioritizes survival, then power-ups, then pellets.
    """
    # 1. Immediate Threat Detection (Ghost proximity)
    ghost_indices = [4, 8, 12, 16]
    for idx in ghost_indices:
        rel_x, rel_y, dist, threat = obs[idx:idx + 4]
        if dist < 0.15 and threat > 0:
            if abs(rel_x) > abs(rel_y):
                return 4 if rel_x < 0 else 3
            else:
                return 2 if rel_y < 0 else 1

    # 2. Aggression Mode (frightened ghosts)
    if obs[28] > 0 and obs[29] > 0.2:
        for idx in ghost_indices:
            rel_x, rel_y, dist, threat = obs[idx:idx + 4]
            if threat < 0:
                if abs(rel_x) > abs(rel_y):
                    return 3 if rel_x < 0 else 4
                else:
                    return 1 if rel_y < 0 else 2

    # 3. Objective Navigation (Pellet Radar)
    p_rel_x, p_rel_y = obs[21], obs[22]
    if abs(p_rel_x) > abs(p_rel_y):
        return 4 if p_rel_x > 0 else 3
    else:
        return 2 if p_rel_y > 0 else 1


# --- Execution ---
TARGET_MAZES = 5  # How many mazes to clear before stopping

env = PacManEnv(render_mode="human", maze_algorithm="recursive_backtracking")
obs, info = env.reset()

print(f"STRESS TEST: Attempting to clear {TARGET_MAZES} mazes in a single episode.")
print("Winning a level auto-advances to a fresh maze — no reset needed.\n")

prev_levels = 0

try:
    while True:
        action = get_strategic_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # Announce each new maze clear
        if info["levels_completed"] > prev_levels:
            prev_levels = info["levels_completed"]
            print(f"  ✓ Maze {prev_levels} cleared! Score so far: {info['score']} | "
                  f"Lives: {info['lives']} | Engine level: {info['level']}")
            if prev_levels >= TARGET_MAZES:
                print(f"\nTarget of {TARGET_MAZES} mazes reached — done!")
                break

        if terminated or truncated:
            status = "GAME OVER" if info["game_over"] else "TRUNCATED"
            print(f"\nEpisode ended ({status}) | "
                  f"Mazes cleared: {info['levels_completed']} | "
                  f"Final score: {info['score']}")
            break

        time.sleep(0.01)

finally:
    env.close()