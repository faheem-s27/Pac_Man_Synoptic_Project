"""
dqn_train_visual.py
===================
Egocentric 35-input DQN visual trainer.
"""

import sys
import os
import pygame
import torch
import random
import csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.PacManEnv import PacManEnv
from Code.CurriculumManager import CurriculumManager
from dqn_agent import DQNAgent
from checkpoint_utils import save_checkpoint, load_checkpoint

MAX_WINDOW_W = 800
MAX_WINDOW_H = 800
DASHBOARD_W  = 400
INFO_BAR_H   = 60
TARGET_FPS   = 120
SAVE_PATH    = os.path.join(_HERE, "dqn_pacman.pth")
CHECKPOINT_PATH = os.path.join(_HERE, "dqn_checkpoint.pt")
LOG_PATH     = os.path.join(_HERE, "training_log.csv")
SAVE_EVERY_EPISODES = 50
INCLUDE_CURRICULUM_STATE = True

ACTION_NAMES = ['FORWARD', 'LEFT', 'RIGHT', 'BACKWARD']
CARDINAL_TO_VEC = {
    PacManEnv.UP: (0, -1),
    PacManEnv.DOWN: (0, 1),
    PacManEnv.LEFT_C: (-1, 0),
    PacManEnv.RIGHT_C: (1, 0),
}
RAY_DIRS = [
    (0, -1), (0, 1), (-1, 0), (1, 0),
    (-1, -1), (1, -1), (-1, 1), (1, 1),
]


def _draw_action_arrow(surface, env: PacManEnv, target_dir: int | None, blocked: bool):
    if target_dir not in CARDINAL_TO_VEC:
        return
    eng = env.engine
    ts = eng.tile_size
    px = int(eng.pacman.x + eng.pacman.size // 2)
    py = int(eng.pacman.y + eng.pacman.size // 2)
    dx, dy = CARDINAL_TO_VEC[target_dir]
    length = max(12, ts // 2)
    end_x = px + dx * length
    end_y = py + dy * length
    color = (255, 120, 0) if blocked else (0, 255, 120)
    pygame.draw.line(surface, color, (px, py), (end_x, end_y), 3)
    pygame.draw.circle(surface, color, (end_x, end_y), 4)


def _draw_visited_heatmap(surface, env: PacManEnv):
    if not env._visit_counts:
        return
    eng = env.engine
    ts = eng.tile_size
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    max_visits = max(env._visit_counts.values())
    for (tx, ty), count in env._visit_counts.items():
        if count <= 0:
            continue
        if count == 1:
            color = (80, 200, 80, 45)
        else:
            alpha = min(180, 45 + int(135 * (count / max(2, max_visits))))
            color = (255, 160, 60, alpha)
        pygame.draw.rect(overlay, color, pygame.Rect(tx * ts, ty * ts, ts, ts))
    surface.blit(overlay, (0, 0))


def _draw_raycast_overlay(surface, env: PacManEnv):
    eng = env.engine
    ts = eng.tile_size
    px = eng.pacman.x + eng.pacman.size / 2.0
    py = eng.pacman.y + eng.pacman.size / 2.0
    tx = int(px // ts)
    ty = int(py // ts)

    food_tiles = {(int(x // ts), int(y // ts)) for x, y in eng.pellets}
    power_tiles = {(int(x // ts), int(y // ts)) for x, y in eng.power_pellets}

    lethal_ghost_tiles = set()
    edible_ghost_tiles = set()
    for g in eng.ghosts:
        gx = int((g.x + ts / 2.0) // ts)
        gy = int((g.y + ts / 2.0) // ts)
        if g.state.name == "FRIGHTENED":
            edible_ghost_tiles.add((gx, gy))
        elif g.state.name != "EATEN":
            lethal_ghost_tiles.add((gx, gy))

    for dx, dy in RAY_DIRS:
        cx, cy = tx, ty
        wall_point = None
        food_point = None
        power_point = None
        ghost_point = None
        while True:
            cx += dx
            cy += dy
            if not (0 <= cx < eng.maze.width and 0 <= cy < eng.maze.height):
                break
            if eng.maze.maze[cy][cx] == 1:
                wall_point = (int((cx + 0.5) * ts), int((cy + 0.5) * ts))
                break
            center = (int((cx + 0.5) * ts), int((cy + 0.5) * ts))
            if food_point is None and (cx, cy) in food_tiles:
                food_point = center
            if power_point is None and (cx, cy) in power_tiles:
                power_point = center
            if ghost_point is None:
                if (cx, cy) in lethal_ghost_tiles:
                    ghost_point = (center, (255, 80, 80))
                elif (cx, cy) in edible_ghost_tiles:
                    ghost_point = (center, (80, 255, 255))

        if wall_point is not None:
            pygame.draw.line(surface, (230, 230, 230), (int(px), int(py)), wall_point, 1)
            pygame.draw.circle(surface, (230, 230, 230), wall_point, 2)
        if food_point is not None:
            pygame.draw.circle(surface, (80, 255, 80), food_point, 3)
        if power_point is not None:
            pygame.draw.circle(surface, (80, 130, 255), power_point, 3)
        if ghost_point is not None:
            point, color = ghost_point
            pygame.draw.circle(surface, color, point, 4)


def run_visual_dqn():
    pygame.init()

    win_w = MAX_WINDOW_W + DASHBOARD_W
    win_h = MAX_WINDOW_H + INFO_BAR_H
    window = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("DQN Pac-Man — Egocentric 35D")

    info_font = pygame.font.Font(None, 28)
    dash_font = pygame.font.Font(None, 22)
    header_font = pygame.font.Font(None, 26)
    fps_clock = pygame.time.Clock()

    curriculum = CurriculumManager()
    base_settings = curriculum.get_settings()

    env = PacManEnv(render_mode="rgb_array", **base_settings)
    agent = DQNAgent(input_dim=35, output_dim=4)
    start_episode = 0
    if os.path.exists(CHECKPOINT_PATH):
        try:
            load_meta = load_checkpoint(CHECKPOINT_PATH, agent, curriculum=curriculum, map_location=agent.device)
            if load_meta.get("loaded"):
                start_episode = int(load_meta.get("episode", 0))
                print(
                    f"Resumed checkpoint {CHECKPOINT_PATH} | "
                    f"episode={start_episode} epsilon={agent.epsilon:.4f} step_count={agent.step_count}"
                )
        except Exception as e:
            print(f"Failed to load checkpoint. Starting fresh. Error: {e}")
    elif os.path.exists(SAVE_PATH):
        try:
            load_meta = load_checkpoint(SAVE_PATH, agent, curriculum=None, map_location=agent.device)
            if load_meta.get("loaded"):
                print(f"Loaded legacy weights from {SAVE_PATH}")
        except Exception as e:
            print(f"Failed to load weights. Starting fresh. Error: {e}")

    batch_size = 128
    episode = start_episode

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Episode", "Stage", "Maze_Seed", "Reward", "Steps",
                "Outcome", "Win", "Epsilon", "Pellets", "Power_Pellets",
                "Ghosts", "Explore_Rate", "Avg_Loss"
            ])

    while True:
        episode += 1

        current_settings = curriculum.get_settings()
        dynamic_seed = random.randint(0, 9999999)
        current_settings['maze_seed'] = dynamic_seed

        env._base_cfg.update(current_settings)
        env.max_episode_steps = current_settings.get('max_episode_steps', 2000)
        env.current_stage = curriculum.current_stage

        state, _ = env.reset(seed=dynamic_seed)

        total_reward = 0.0
        loss_history = []
        done = False
        episode_steps = 0
        last_death_cause = "NONE"
        last_info = {}

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    print(f"\nSaving checkpoint to {CHECKPOINT_PATH}...")
                    save_checkpoint(
                        CHECKPOINT_PATH,
                        agent,
                        episode,
                        curriculum=curriculum,
                        include_curriculum=INCLUDE_CURRICULUM_STATE,
                    )
                    torch.save(agent.policy_net.state_dict(), SAVE_PATH)
                    env.close()
                    pygame.quit()
                    sys.exit()

            action = agent.select_action(state, valid_actions=[0, 1, 2, 3])
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_steps += 1

            if isinstance(info, dict):
                last_info = info
                if "death_cause" in info:
                    last_death_cause = info["death_cause"]

            agent.memory.push(state, action, reward, next_state, done)
            total_reward += reward

            loss = agent.optimize_model(batch_size)
            if loss is not None:
                loss_history.append(loss)

            # High-Performance Rendering + debug overlays
            rgb_array = env.render()
            if rgb_array is not None:
                # Draw overlays in native game coordinates first, then scale once.
                native_surf = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))
                _draw_visited_heatmap(native_surf, env)
                _draw_raycast_overlay(native_surf, env)
                target_dir = info.get("target_dir") if isinstance(info, dict) else None
                blocked = bool(info.get("blocked_action", False)) if isinstance(info, dict) else False
                _draw_action_arrow(native_surf, env, target_dir, blocked)

                surf = pygame.transform.scale(native_surf, (MAX_WINDOW_W, MAX_WINDOW_H))
                window.blit(surf, (0, 0))

            # Dashboard
            window.fill((15, 15, 20), (MAX_WINDOW_W, 0, DASHBOARD_W, win_h))

            with torch.no_grad():
                st_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                q_vals = agent.policy_net(st_tensor).squeeze().cpu().numpy()
                target_q_vals = agent.target_net(st_tensor).squeeze().cpu().numpy()

            action_names = ACTION_NAMES

            # Decode 35D state: 8 rays*4 + frightened + norm_x + norm_y.
            rays = np.array(state, dtype=float).reshape(-1)
            ray_features = []
            norm_x = 0.0
            norm_y = 0.0
            if rays.shape[0] >= 35:
                ray_features = [
                    tuple(rays[i*4:(i+1)*4])
                    for i in range(8)
                ]
                frightened_val = float(rays[32])
                norm_x = float(rays[33])
                norm_y = float(rays[34])
            elif rays.shape[0] == 33:
                # Backward compatibility with old models/checkpoints.
                ray_features = [
                    tuple(rays[i*4:(i+1)*4])
                    for i in range(8)
                ]
                frightened_val = float(rays[-1])
            else:
                frightened_val = 0.0

            y_off = 20
            window.blit(header_font.render("=== DQN LOGIC (EGOCENTRIC) ===", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 24
            window.blit(header_font.render(f"Decision: {action_names[action]}", True, (0, 255, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 22

            prev_action = info.get("prev_action") if isinstance(info, dict) else None
            prev_label = action_names[prev_action] if isinstance(prev_action, int) and 0 <= prev_action < len(action_names) else "NONE"
            reversal = bool(info.get("reversal", False)) if isinstance(info, dict) else False
            blocked = bool(info.get("blocked_action", False)) if isinstance(info, dict) else False
            flags_text = f"Prev: {prev_label} | Reversal: {reversal} | Blocked: {blocked}"
            window.blit(dash_font.render(flags_text, True, (255, 180, 120) if reversal or blocked else (200, 200, 200)), (MAX_WINDOW_W + 15, y_off))
            y_off += 20

            window.blit(header_font.render("--- Q-Values ---", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 20
            for i, name in enumerate(action_names):
                q_delta = q_vals[i] - target_q_vals[i]
                q_label = f"{name:<7}: P={q_vals[i]:+07.3f} T={target_q_vals[i]:+07.3f} d={q_delta:+06.3f}"
                window.blit(dash_font.render(q_label, True, (255, 255, 255)), (MAX_WINDOW_W + 15, y_off))
                y_off += 18

            # Show raycast values beneath Q-values
            if ray_features:
                window.blit(header_font.render("--- Egocentric Rays (w,f,p,g) ---", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
                y_off += 20
                for idx, (w_d, f_d, p_d, g_d) in enumerate(ray_features):
                    label = f"Ray {idx}: W={w_d:+.3f} F={f_d:+.3f} P={p_d:+.3f} G={g_d:+.3f}"
                    window.blit(dash_font.render(label, True, (200, 200, 200)), (MAX_WINDOW_W + 15, y_off))
                    y_off += 16
                # Show frightened scalar at the end
                window.blit(dash_font.render(f"Frightened: {frightened_val:+.3f}", True, (135, 206, 250)), (MAX_WINDOW_W + 15, y_off))
                y_off += 18
                window.blit(dash_font.render(f"Tile Norm X: {norm_x:+.3f}", True, (173, 216, 230)), (MAX_WINDOW_W + 15, y_off))
                y_off += 16
                window.blit(dash_font.render(f"Tile Norm Y: {norm_y:+.3f}", True, (173, 216, 230)), (MAX_WINDOW_W + 15, y_off))
                y_off += 18

            reward_breakdown = info.get("reward_breakdown", {}) if isinstance(info, dict) else {}
            if reward_breakdown:
                window.blit(header_font.render("--- Reward Breakdown ---", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
                y_off += 20
                reward_lines = [
                    ("Time", reward_breakdown.get("time_penalty", 0.0)),
                    ("Pellet", reward_breakdown.get("pellet_reward", 0.0)),
                    ("Power", reward_breakdown.get("power_reward", 0.0)),
                    ("Explore", reward_breakdown.get("exploration_reward", 0.0)),
                    ("GhostP", reward_breakdown.get("ghost_pressure", 0.0)),
                    ("Total", reward_breakdown.get("total", 0.0)),
                ]
                for label, value in reward_lines:
                    window.blit(dash_font.render(f"{label:<7}: {value:+07.3f}", True, (190, 220, 255)), (MAX_WINDOW_W + 15, y_off))
                    y_off += 16

            eng = env.engine
            window.blit(header_font.render("--- Ghost States ---", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 20
            for g in eng.ghosts:
                window.blit(dash_font.render(f"{g.name:<6}: {g.state.name}", True, (255, 200, 200)), (MAX_WINDOW_W + 15, y_off))
                y_off += 16

            if eng.frightened_mode:
                remaining_s = max(0.0, (eng.frightened_duration - eng.frightened_timer) / 60.0)
                mode_label = f"Mode: FRIGHTENED ({remaining_s:0.1f}s)"
            elif eng.always_chase:
                mode_label = "Mode: CHASE (always)"
            else:
                if eng.global_scatter_mode:
                    rem_frames = max(0, eng.scatter_duration - eng.scatter_chase_timer)
                    mode_label = f"Mode: SCATTER ({rem_frames/60.0:0.1f}s)"
                else:
                    rem_frames = max(0, eng.chase_duration - eng.scatter_chase_timer)
                    mode_label = f"Mode: CHASE ({rem_frames/60.0:0.1f}s)"
            window.blit(dash_font.render(mode_label, True, (255, 230, 160)), (MAX_WINDOW_W + 15, y_off))
            y_off += 18

            maze_seed = info.get("maze_seed", getattr(eng, "maze_seed", None)) if isinstance(info, dict) else getattr(eng, "maze_seed", None)
            params = (
                f"Seed: {maze_seed} | GSpd:{eng.ghost_speed:.2f} | "
                f"Scat/Chase:{eng.scatter_duration//60}/{eng.chase_duration//60}s"
            )
            window.blit(dash_font.render(params, True, (180, 220, 180)), (MAX_WINDOW_W + 15, y_off))
            y_off += 18

            internal_ticks = int(info.get("internal_ticks", 0)) if isinstance(info, dict) else 0
            window.blit(dash_font.render(f"Internal ticks/step: {internal_ticks}", True, (220, 220, 220)), (MAX_WINDOW_W + 15, y_off))
            y_off += 16

            pellets_remaining = int(info.get("pellets_remaining", len(eng.pellets) + len(eng.power_pellets))) if isinstance(info, dict) else (len(eng.pellets) + len(eng.power_pellets))
            map_clear_pct = float(info.get("map_clear_pct", 0.0)) if isinstance(info, dict) else 0.0
            window.blit(dash_font.render(f"Pellets left: {pellets_remaining} | Map clear: {map_clear_pct:0.1f}%", True, (180, 230, 255)), (MAX_WINDOW_W + 15, y_off))
            y_off += 16

            window.fill((20, 20, 20), (0, MAX_WINDOW_H, MAX_WINDOW_W, INFO_BAR_H))
            avg_loss_disp = sum(loss_history)/len(loss_history) if loss_history else 0.0

            # Pull current step count from env.info if available, otherwise fall back to local episode_steps
            steps_disp = episode_steps
            if isinstance(info, dict):
                steps_disp = int(info.get("steps", steps_disp))

            info_bar = (
                f"Ep: {episode} | Stg: {curriculum.current_stage} | "
                f"Steps: {steps_disp} | Rwd: {total_reward:+.1f} | "
                f"Eps: {agent.epsilon:.3f} | Loss: {avg_loss_disp:.2f} | "
                f"Blk:{'Y' if blocked else 'N'} Rev:{'Y' if reversal else 'N'}"
            )
            window.blit(info_font.render(info_bar, True, (255, 215, 0)), (15, MAX_WINDOW_H + 20))

            pygame.display.flip()
            fps_clock.tick(TARGET_FPS)

            # Sync state for next iteration
            state = next_state

        # -------- POST EPISODE --------
        won = env.engine.won
        curriculum.update_performance(won)

        # Check Promotion and Demotion
        promoted = curriculum.check_promotion()

        # -- DISABLE DEMOTION
        #demoted = curriculum.check_demotion()
        demoted = False

        # Exploration Jolt on Curriculum Change
        if promoted or demoted:
            agent.epsilon = max(agent.epsilon, 0.2)

        avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0.0

        stage = curriculum.current_stage
        maze_seed = dynamic_seed

        pellets = int(last_info.get("pellets", 0)) if isinstance(last_info, dict) else 0
        power_pellets = int(last_info.get("power_pellets", 0)) if isinstance(last_info, dict) else 0
        ghosts = int(last_info.get("ghosts", 0)) if isinstance(last_info, dict) else 0
        explore_rate = float(last_info.get("explore_rate", 0.0)) if isinstance(last_info, dict) else 0.0

        outcome = "WIN" if won else last_death_cause

        with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, stage, maze_seed, float(total_reward), int(env._step_count),
                outcome, int(won), float(agent.epsilon), pellets, power_pellets,
                ghosts, explore_rate, float(avg_loss)
            ])

        agent.update_target_network()

        if episode % SAVE_EVERY_EPISODES == 0:
            save_checkpoint(
                CHECKPOINT_PATH,
                agent,
                episode,
                curriculum=curriculum,
                include_curriculum=INCLUDE_CURRICULUM_STATE,
            )
            torch.save(agent.policy_net.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    run_visual_dqn()