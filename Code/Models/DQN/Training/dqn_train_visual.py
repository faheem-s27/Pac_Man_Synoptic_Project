"""
dqn_train_visual.py
===================
Egocentric 27-input DQN visual trainer.
"""

import sys
import os
import pygame
import torch
import random
import csv
import numpy as np
from datetime import datetime

_HERE      = os.path.dirname(os.path.abspath(__file__))
_DQN_ROOT  = os.path.dirname(_HERE)                          # Code/Models/DQN/
_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))  # project root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _DQN_ROOT not in sys.path:
    sys.path.insert(0, _DQN_ROOT)   # dqn_agent, checkpoint_utils, action_masking_wrapper

from Code.Environment.PacManEnv import PacManEnv
from Code.Environment.CurriculumManager import CurriculumManager
from Code.Models.DQN.dqn_agent import DQNAgent
from Code.Models.DQN.checkpoint_utils import save_checkpoint, load_checkpoint
from Code.Models.DQN.action_masking_wrapper import DQNActionMaskingWrapper

MAX_WINDOW_W = 800
MAX_WINDOW_H = 800
DASHBOARD_W  = 400
INFO_BAR_H   = 60
TARGET_FPS   = 120
SAVE_PATH    = os.path.join(_DQN_ROOT, "Checkpoints", "dqn_pacman.pth")
CHECKPOINT_PATH = os.path.join(_DQN_ROOT, "Checkpoints", "dqn_checkpoint.pt")
LOG_DIR      = os.path.join(_DQN_ROOT, "CSV_History")
RUN_TIMESTAMP = datetime.now().strftime("%d-%m_%H-%M-%S")
LOG_PATH     = os.path.join(LOG_DIR, f"training_log_{RUN_TIMESTAMP}.csv")
SAVE_EVERY_EPISODES = 50
INCLUDE_CURRICULUM_STATE = True
# For DQN runs we prefer starvation/game-over as terminal causes over max-step truncation.
DQN_MAX_EPISODE_STEPS = None
# Scale raw env rewards before replay insertion for DQN target stability.
DQN_REWARD_SCALE = 100.0

ACTION_NAMES = ['FORWARD', 'LEFT', 'RIGHT', 'BACKWARD']
CARDINAL_TO_VEC = {
    PacManEnv.UP: (0, -1),
    PacManEnv.DOWN: (0, 1),
    PacManEnv.LEFT_C: (-1, 0),
    PacManEnv.RIGHT_C: (1, 0),
}
RAY_DIRS = [
    (0, -1), (0, 1), (-1, 0), (1, 0),
]

# Runtime speed presets for visual training loop (decision steps per second).
SPEED_MODES = [
    ("1x Slow", 1),
    ("5x Slow", 5),
    ("Normal", 30),
    ("Fast", 60),
    ("Max", TARGET_FPS),
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

    if DQN_MAX_EPISODE_STEPS is None:
        print("[DQN][Visual] max_episode_steps=None -> no max-step truncation; episodes end via win/death/starvation.")
    else:
        print(f"[DQN][Visual] max_episode_steps={DQN_MAX_EPISODE_STEPS} -> max-step truncation enabled.")

    win_w = MAX_WINDOW_W + DASHBOARD_W
    win_h = MAX_WINDOW_H + INFO_BAR_H
    window = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("DQN Pac-Man — Egocentric 27D")

    info_font = pygame.font.Font(None, 28)
    dash_font = pygame.font.Font(None, 22)
    header_font = pygame.font.Font(None, 26)
    fps_clock = pygame.time.Clock()
    speed_mode_index = 4  # Default to Max (existing behavior).

    curriculum = CurriculumManager()
    base_settings = curriculum.get_settings()

    env = DQNActionMaskingWrapper(PacManEnv(render_mode="rgb_array", **base_settings))
    base_env = env.unwrapped
    # 29-dim obs: 4 dirs × 6 ray channels + 3 BFS + 2 power
    agent = DQNAgent(input_dim=29, output_dim=4)
    start_episode = 0
    if os.path.exists(CHECKPOINT_PATH):
        try:
            load_meta = load_checkpoint(CHECKPOINT_PATH, agent, curriculum=curriculum, map_location=agent.device)
            if load_meta.get("loaded"):
                start_episode = int(load_meta.get("episode", 0))
                loaded_keys = int(load_meta.get("loaded_keys", 0))
                total_keys = int(load_meta.get("total_keys", 0))
                load_mode = str(load_meta.get("load_mode", "full"))
                print(
                    f"Resumed checkpoint {CHECKPOINT_PATH} | "
                    f"episode={start_episode} epsilon={agent.epsilon:.4f} step_count={agent.step_count} "
                    f"load={load_mode} ({loaded_keys}/{total_keys} tensors)"
                )
            else:
                reason = load_meta.get("reason", "unknown")
                print(f"Checkpoint skipped ({reason}); starting fresh training state.")
        except Exception as e:
            print(f"Failed to load checkpoint. Starting fresh. Error: {e}")
    elif os.path.exists(SAVE_PATH):
        try:
            load_meta = load_checkpoint(SAVE_PATH, agent, curriculum=None, map_location=agent.device)
            if load_meta.get("loaded"):
                loaded_keys = int(load_meta.get("loaded_keys", 0))
                total_keys = int(load_meta.get("total_keys", 0))
                load_mode = str(load_meta.get("load_mode", "full"))
                print(f"Loaded legacy weights from {SAVE_PATH} ({load_mode}, {loaded_keys}/{total_keys} tensors).")
            else:
                reason = load_meta.get("reason", "unknown")
                print(f"Legacy weights skipped ({reason}); starting with random initialization.")
        except Exception as e:
            print(f"Failed to load weights. Starting fresh. Error: {e}")

    episode = start_episode

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Algorithm", "Episode", "Stage", "Maze_Seed", "Reward", "Macro_Steps", "Micro_Ticks",
                "Outcome", "Win", "Epsilon", "Pellets", "Power_Pellets",
                "Ghosts", "Explore_Rate", "Avg_Loss"
            ])

    while True:
        episode += 1

        current_settings = curriculum.get_settings()

        dynamic_seed = random.randint(0, 9999999)
        #dynamic_seed = 22459265
        current_settings['maze_seed'] = dynamic_seed
        current_settings['max_episode_steps'] = DQN_MAX_EPISODE_STEPS

        base_env._base_cfg.update(current_settings)
        base_env.max_episode_steps = DQN_MAX_EPISODE_STEPS
        base_env.current_stage = curriculum.current_stage

        state, _ = env.reset(seed=dynamic_seed)

        total_reward = 0.0
        loss_history = []
        done = False
        episode_steps = 0
        episode_micro_ticks = 0
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
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_1, pygame.K_KP1):
                        speed_mode_index = 0
                    elif event.key in (pygame.K_2, pygame.K_KP2):
                        speed_mode_index = 1
                    elif event.key in (pygame.K_3, pygame.K_KP3):
                        speed_mode_index = 2
                    elif event.key in (pygame.K_4, pygame.K_KP4):
                        speed_mode_index = 3
                    elif event.key in (pygame.K_5, pygame.K_KP5):
                        speed_mode_index = 4

            valid_actions = env.get_valid_actions()
            policy_action, exploring = agent.select_action(
                state,
                valid_actions=valid_actions,
                return_exploration=True,
            )
            action = env.pick_action(policy_action, exploring=exploring)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_steps += 1
            internal_ticks_step = int(info.get("internal_ticks", 0)) if isinstance(info, dict) else 0
            episode_micro_ticks += max(0, internal_ticks_step)
            step_ticks = max(1, internal_ticks_step)

            next_valid_actions = env.get_valid_actions()
            next_valid_mask = np.zeros(agent.action_dim, dtype=np.float32)
            for a in next_valid_actions:
                if 0 <= int(a) < agent.action_dim:
                    next_valid_mask[int(a)] = 1.0
            discount_pow = float(agent.gamma ** step_ticks)

            if isinstance(info, dict):
                last_info = info
                if "death_cause" in info:
                    last_death_cause = info["death_cause"]

            scaled_reward = float(reward) / DQN_REWARD_SCALE

            agent.memory.push(
                state,
                action,
                scaled_reward,
                next_state,
                done,
                next_valid_mask=next_valid_mask,
                discount_pow=discount_pow,
            )
            total_reward += reward

            loss = agent.optimize_model()
            if loss is not None:
                loss_history.append(loss)
                # Keep target network close to policy network during long episodes.
                agent.update_target_network()

            # High-Performance Rendering + debug overlays
            rgb_array = env.render()
            if rgb_array is not None:
                # Draw overlays in native game coordinates first, then scale once.
                native_surf = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))
                _draw_visited_heatmap(native_surf, base_env)
                _draw_raycast_overlay(native_surf, base_env)
                target_dir = info.get("target_dir") if isinstance(info, dict) else None
                blocked = bool(info.get("blocked_action", False)) if isinstance(info, dict) else False
                _draw_action_arrow(native_surf, base_env, target_dir, blocked)

                surf = pygame.transform.scale(native_surf, (MAX_WINDOW_W, MAX_WINDOW_H))
                window.blit(surf, (0, 0))

            # Dashboard
            window.fill((15, 15, 20), (MAX_WINDOW_W, 0, DASHBOARD_W, win_h))

            with torch.no_grad():
                st_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                q_vals = agent.policy_net(st_tensor).squeeze().cpu().numpy()
                target_q_vals = agent.target_net(st_tensor).squeeze().cpu().numpy()

            action_names = ACTION_NAMES

            # Decode 29D state:
            #   4 rays × 6 channels (wall,food,power,lethal_ghost,edible_ghost,visit_sat) = 24
            #   + 3 BFS (near_food=24, near_danger=25, near_edible=26)
            #   + 2 power (is_powered=27, power_remaining=28)
            # All values in [-1, 1] via 2x-1 normalisation.
            rays = np.array(state, dtype=float).reshape(-1)
            ray_features = []
            nearest_food = 0.0
            nearest_danger = 0.0
            nearest_edible = 0.0
            is_powered = 0.0
            power_remaining = 0.0
            if rays.shape[0] >= 29:
                # Current 29D layout: 6 channels per ray direction
                ray_features = [
                    tuple(rays[i*6:(i+1)*6])
                    for i in range(4)
                ]
                nearest_food            = float(rays[24])
                nearest_danger          = float(rays[25])
                nearest_edible          = float(rays[26])
                is_powered              = float(rays[27])
                power_remaining         = float(rays[28])
            elif rays.shape[0] >= 21:
                # Legacy obs fallback
                ray_features = [
                    tuple(rays[i*4:(i+1)*4])
                    for i in range(4)
                ]
                nearest_food    = float(rays[16])
                nearest_danger  = float(rays[17])
                nearest_edible  = float(rays[18])
                is_powered      = float(rays[19])
                power_remaining = float(rays[20])

            y_off = 20
            window.blit(header_font.render("=== DQN LOGIC (EGOCENTRIC) ===", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 24
            window.blit(header_font.render(f"Decision: {action_names[action]}", True, (0, 255, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 22

            target_dir = info.get("target_dir") if isinstance(info, dict) else None
            dir_name = "NONE"
            if isinstance(target_dir, int) and target_dir in (PacManEnv.UP, PacManEnv.DOWN, PacManEnv.LEFT_C, PacManEnv.RIGHT_C):
                dir_name = {
                    PacManEnv.UP: "UP",
                    PacManEnv.DOWN: "DOWN",
                    PacManEnv.LEFT_C: "LEFT",
                    PacManEnv.RIGHT_C: "RIGHT",
                }[target_dir]
            flags_text = f"TargetDir: {dir_name}"
            window.blit(dash_font.render(flags_text, True, (200, 200, 200)), (MAX_WINDOW_W + 15, y_off))
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
                n_ch = len(ray_features[0])
                ray_header = "--- Egocentric Rays (w,f,p,lg,eg,vs) ---" if n_ch == 6 else "--- Egocentric Rays (w,f,p,g) ---"
                window.blit(header_font.render(ray_header, True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
                y_off += 20
                dir_names = ["FWD", "LFT", "RGT", "BCK"]
                for idx, channels in enumerate(ray_features):
                    if n_ch == 6:
                        w_d, f_d, p_d, lg_d, eg_d, vs = channels
                        label = f"{dir_names[idx]}: W={w_d:.2f} F={f_d:.2f} P={p_d:.2f} LG={lg_d:.2f} EG={eg_d:.2f} VS={vs:.2f}"
                    else:
                        w_d, f_d, p_d, g_d = channels
                        label = f"Ray {idx}: W={w_d:.2f} F={f_d:.2f} P={p_d:.2f} G={g_d:+.2f}"
                    window.blit(dash_font.render(label, True, (200, 200, 200)), (MAX_WINDOW_W + 15, y_off))
                    y_off += 16
                window.blit(dash_font.render(f"Near Food(BFS):   {nearest_food:+.3f}", True, (120, 240, 120)), (MAX_WINDOW_W + 15, y_off))
                y_off += 16
                window.blit(dash_font.render(f"Near Danger(BFS): {nearest_danger:+.3f}", True, (255, 120, 120)), (MAX_WINDOW_W + 15, y_off))
                y_off += 16
                window.blit(dash_font.render(f"Near Edible(BFS): {nearest_edible:+.3f}", True, (120, 220, 255)), (MAX_WINDOW_W + 15, y_off))
                y_off += 16
                window.blit(dash_font.render(f"Powered: {is_powered:+.1f}  Remain: {power_remaining:+.3f}", True, (255, 220, 140)), (MAX_WINDOW_W + 15, y_off))
                y_off += 16
                y_off += 2

            reward_breakdown = info.get("reward_breakdown", {}) if isinstance(info, dict) else {}
            if reward_breakdown:
                window.blit(header_font.render("--- Reward Breakdown ---", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
                y_off += 20
                reward_lines = [
                    ("Pellet", reward_breakdown.get("pellet_reward", 0.0)),
                    ("Power", reward_breakdown.get("power_reward", 0.0)),
                    ("Ghost", reward_breakdown.get("ghost_reward", 0.0)),
                    ("Win", reward_breakdown.get("win_reward", 0.0)),
                    ("Death", reward_breakdown.get("death_penalty", 0.0)),
                    ("Starve", reward_breakdown.get("starvation_penalty", 0.0)),
                    ("Shape", reward_breakdown.get("food_shaping_reward", 0.0)),
                    ("Invalid", reward_breakdown.get("invalid_action_penalty", 0.0)),
                    ("Total", reward_breakdown.get("total", 0.0)),
                ]
                for label, value in reward_lines:
                    window.blit(dash_font.render(f"{label:<7}: {value:+07.3f}", True, (190, 220, 255)), (MAX_WINDOW_W + 15, y_off))
                    y_off += 16

            eng = base_env.engine
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

            window.blit(dash_font.render(f"Internal ticks/step: {internal_ticks_step}", True, (220, 220, 220)), (MAX_WINDOW_W + 15, y_off))
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
                f"Macro: {steps_disp} | Micro: {episode_micro_ticks} | Rwd: {total_reward:+.1f} | "
                f"Eps: {agent.epsilon:.3f} | Loss: {avg_loss_disp:.2f}"
            )
            window.blit(info_font.render(info_bar, True, (255, 215, 0)), (15, MAX_WINDOW_H + 20))

            speed_name, speed_fps = SPEED_MODES[speed_mode_index]
            speed_hint = f"Speed: {speed_name} ({speed_fps} step/s)  [1..5]"
            window.blit(dash_font.render(speed_hint, True, (180, 220, 255)), (15, MAX_WINDOW_H + 42))

            pygame.display.flip()
            fps_clock.tick(speed_fps)

            # Sync state for next iteration
            state = next_state

        # -------- POST EPISODE --------
        won = base_env.engine.won
        curriculum.update_performance(won)

        # Check Promotion and Demotion
        promoted = curriculum.check_promotion()

        # -- DISABLE DEMOTION
        #demoted = curriculum.check_demotion()
        demoted = False

        # Exploration Jolt on Curriculum Change
        if promoted or demoted:
            agent.apply_exploration_jolt(min_epsilon=0.2, duration_steps=50_000)

        avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0.0

        stage = curriculum.current_stage
        maze_seed = dynamic_seed

        pellets = int(last_info.get("pellets", 0)) if isinstance(last_info, dict) else 0
        pellets_remaining = int(last_info.get("pellets_remaining", 0)) if isinstance(last_info, dict) else 0
        power_pellets = int(last_info.get("power_pellets", 0)) if isinstance(last_info, dict) else 0
        ghosts = int(last_info.get("ghosts", 0)) if isinstance(last_info, dict) else 0
        explore_rate = float(last_info.get("explore_rate", 0.0)) if isinstance(last_info, dict) else 0.0

        outcome = "WIN" if won else last_death_cause

        with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "DQN", episode, stage, maze_seed, float(total_reward), int(episode_steps), int(episode_micro_ticks),
                outcome, int(won), float(agent.epsilon), pellets, power_pellets,
                ghosts, explore_rate, float(avg_loss)
            ])


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