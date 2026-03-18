"""
dqn_train_visual.py
===================
Egocentric 33-input DQN visual trainer.
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

MAX_WINDOW_W = 800
MAX_WINDOW_H = 800
DASHBOARD_W  = 400
INFO_BAR_H   = 60
TARGET_FPS   = 120
SAVE_PATH    = os.path.join(_HERE, "dqn_pacman.pth")
LOG_PATH     = os.path.join(_HERE, "training_log.csv")


def run_visual_dqn():
    pygame.init()

    win_w = MAX_WINDOW_W + DASHBOARD_W
    win_h = MAX_WINDOW_H + INFO_BAR_H
    window = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("DQN Pac-Man — Egocentric 33D")

    info_font = pygame.font.Font(None, 28)
    dash_font = pygame.font.Font(None, 22)
    header_font = pygame.font.Font(None, 26)
    fps_clock = pygame.time.Clock()

    curriculum = CurriculumManager()
    base_settings = curriculum.get_settings()

    env = PacManEnv(render_mode="rgb_array", **base_settings)
    agent = DQNAgent(input_dim=33, output_dim=4)

    if os.path.exists(SAVE_PATH):
        try:
            agent.policy_net.load_state_dict(torch.load(SAVE_PATH, map_location=agent.device, weights_only=True))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Loaded existing weights from {SAVE_PATH}")
        except Exception as e:
            print(f"Failed to load weights. Starting fresh. Error: {e}")

    batch_size = 64
    episode = 0

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
                    print(f"\nSaving weights to {SAVE_PATH}...")
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

            # High-Performance Rendering
            rgb_array = env.render()
            if rgb_array is not None:
                surf = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))
                surf = pygame.transform.scale(surf, (MAX_WINDOW_W, MAX_WINDOW_H))
                window.blit(surf, (0, 0))

            # Dashboard
            window.fill((15, 15, 20), (MAX_WINDOW_W, 0, DASHBOARD_W, win_h))

            with torch.no_grad():
                st_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                q_vals = agent.policy_net(st_tensor).squeeze().cpu().numpy()

            action_names = ['FORWARD', 'LEFT', 'RIGHT', 'BACKWARD']

            # Decode the 8 egocentric rays from the 33D state vector:
            # 8 rays * 4 features (wall, food, power, ghost) + 1 frightened scalar.
            rays = np.array(state, dtype=float).reshape(-1)
            ray_features = []
            if rays.shape[0] == 33:
                ray_features = [
                    tuple(rays[i*4:(i+1)*4])
                    for i in range(8)
                ]
                frightened_val = float(rays[-1])
            else:
                frightened_val = 0.0

            y_off = 20
            window.blit(header_font.render("=== DQN LOGIC (EGOCENTRIC) ===", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 30
            window.blit(header_font.render(f"Decision: {action_names[action]}", True, (0, 255, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 30

            window.blit(header_font.render("--- Q-Values ---", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 25
            for i, name in enumerate(action_names):
                window.blit(dash_font.render(f"{name:<7}: {q_vals[i]:+08.3f}", True, (255, 255, 255)), (MAX_WINDOW_W + 15, y_off))
                y_off += 24

            # Show raycast values beneath Q-values
            if ray_features:
                window.blit(header_font.render("--- Egocentric Rays (w,f,p,g) ---", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
                y_off += 25
                for idx, (w_d, f_d, p_d, g_d) in enumerate(ray_features):
                    label = f"Ray {idx}: W={w_d:+.3f} F={f_d:+.3f} P={p_d:+.3f} G={g_d:+.3f}"
                    window.blit(dash_font.render(label, True, (200, 200, 200)), (MAX_WINDOW_W + 15, y_off))
                    y_off += 20
                # Show frightened scalar at the end
                window.blit(dash_font.render(f"Frightened: {frightened_val:+.3f}", True, (135, 206, 250)), (MAX_WINDOW_W + 15, y_off))
                y_off += 24

            window.fill((20, 20, 20), (0, MAX_WINDOW_H, MAX_WINDOW_W, INFO_BAR_H))
            avg_loss_disp = sum(loss_history)/len(loss_history) if loss_history else 0.0

            # Pull current step count from env.info if available, otherwise fall back to local episode_steps
            steps_disp = episode_steps
            if isinstance(info, dict):
                steps_disp = int(info.get("steps", steps_disp))

            info_bar = (
                f"Ep: {episode} | Stg: {curriculum.current_stage} | "
                f"Steps: {steps_disp} | Rwd: {total_reward:+.1f} | "
                f"Eps: {agent.epsilon:.3f} | Loss: {avg_loss_disp:.2f}"
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
        demoted = curriculum.check_demotion()

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

        if episode % 50 == 0:
            torch.save(agent.policy_net.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    run_visual_dqn()