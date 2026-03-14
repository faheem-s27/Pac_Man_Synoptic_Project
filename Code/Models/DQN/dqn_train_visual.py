"""
dqn_train_visual.py
===================
A real-time monitor tailored for the 27-input Raycast + Compass + Fright-Timer.
Incorporates Egocentric Action Masking. Optimized for high-speed intersection jumping.
"""

import sys
import os
import pygame
import torch
import random
import numpy as np

# Path resolution for imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.PacManEnv import PacManEnv
from Code.CurriculumManager import CurriculumManager
from dqn_agent import DQNAgent

# --- Tunables ---
MAX_WINDOW_W = 800
MAX_WINDOW_H = 800
DASHBOARD_W  = 400
INFO_BAR_H   = 60
TARGET_FPS   = 120 # Dashboard update speed
SAVE_PATH    = os.path.join(_HERE, "dqn_pacman.pth")

def run_visual_dqn():
    pygame.init()

    win_w = MAX_WINDOW_W + DASHBOARD_W
    win_h = MAX_WINDOW_H + INFO_BAR_H
    window = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("DQN Pac-Man — EGOCENTRIC VISUAL TRAINER")

    info_font = pygame.font.Font(None, 28)
    dash_font = pygame.font.Font(None, 24)
    fps_clock = pygame.time.Clock()

    curriculum = CurriculumManager(stage_duration=500)
    base_settings = curriculum.get_settings_for_generation(0)

    env = PacManEnv(render_mode=None, **base_settings)
    agent = DQNAgent(input_dim=27, output_dim=3)

    if os.path.exists(SAVE_PATH):
        try:
            agent.policy_net.load_state_dict(torch.load(SAVE_PATH, map_location=agent.device, weights_only=True))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Loaded existing weights from {SAVE_PATH}")
        except Exception as e:
            print(f"Failed to load weights. Starting fresh. Error: {e}")

    batch_size = 64
    episode = 0

    while True:
        episode += 1
        current_settings = curriculum.get_settings_for_generation(episode)
        dynamic_seed = random.randint(0, 9999999)
        current_settings['maze_seed'] = dynamic_seed

        env._base_cfg.update(current_settings)
        env.max_episode_steps = current_settings.get('max_episode_steps', 2000)

        state, _ = env.reset(seed=dynamic_seed)
        total_reward = 0
        loss_history = []
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    print(f"\nSaving weights to {SAVE_PATH}...")
                    torch.save(agent.policy_net.state_dict(), SAVE_PATH)
                    pygame.quit()
                    sys.exit()

            valid_rel_actions = env.get_valid_relative_actions()
            action = agent.select_action(state, valid_actions=valid_rel_actions)

            # High-speed physics execution without rendering callback
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            total_reward += reward

            loss = agent.optimize_model(batch_size)
            if loss is not None: loss_history.append(loss)

            # RENDERING
            game_surf = pygame.Surface((env.engine.screen_width, env.engine.screen_height))
            game_surf.fill((0, 0, 0))
            env.engine.draw(game_surf)
            env._screen = game_surf
            env._draw_debug_sensors()
            window.blit(pygame.transform.scale(game_surf, (MAX_WINDOW_W, MAX_WINDOW_H)), (0, 0))

            # DASHBOARD
            window.fill((15, 15, 20), (MAX_WINDOW_W, 0, DASHBOARD_W, win_h))
            with torch.no_grad():
                st_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_vals = agent.policy_net(st_tensor).squeeze().cpu().numpy()

            action_names = ['FORWARD', 'LEFT', 'RIGHT']

            lines = [
                "=== DQN DEBUG HUD (27-D Ego-Centric) ===",
                f"Action: {action_names[action]}",
                f"Valid:  {[action_names[a] for a in valid_rel_actions]}",
                "",
                "--- Raw Q-Values ---",
                f"FWD: {q_vals[0]:+08.2f}",
                f"LFT: {q_vals[1]:+08.2f}",
                f"RGT: {q_vals[2]:+08.2f}",
                "",
                "--- 8 Ego-Raycasts (Wall|Food|Ghost) ---",
                "Dir | Wall | Food | Ghost",
            ]

            dir_labels = ["F", "B", "L", "R", "F-L", "F-R", "B-L", "B-R"]
            for i, d in enumerate(dir_labels):
                row = f"{d:<3} | {state[i*3]:4.2f} | {state[i*3+1]:4.2f} | {state[i*3+2]:+4.2f}"
                lines.append(row)

            compass_x, compass_y = state[24], state[25]
            fright_flag = state[26]

            lines.extend([
                "",
                "--- Ego Compass (X=FWD, Y=LFT) ---",
                f"Vec X: {compass_x:+4.2f}",
                f"Vec Y: {compass_y:+4.2f}",
                "",
                "--- Global Status ---",
                f"Fright Mode: {'ACTIVE' if fright_flag > 0.5 else 'INACTIVE'}",
            ])

            y_off = 20
            for line in lines:
                color = (255, 215, 0) if "---" in line or "===" in line else (255, 255, 255)
                if "Valid" in line: color = (0, 255, 255)
                elif "Action:" in line: color = (0, 255, 0)
                elif "ACTIVE" in line: color = (0, 255, 0)
                window.blit(dash_font.render(line, True, color), (MAX_WINDOW_W + 15, y_off))
                y_off += 25

            # INFO BAR
            window.fill((20, 20, 20), (0, MAX_WINDOW_H, MAX_WINDOW_W, INFO_BAR_H))
            avg_loss = sum(loss_history)/len(loss_history) if loss_history else 0.0
            info = f"Ep: {episode} | Step: {env._step_count} | Rwd: {total_reward:+.1f} | Eps: {agent.epsilon:.3f} | Loss: {avg_loss:.2f}"
            window.blit(info_font.render(info, True, (255, 215, 0)), (15, MAX_WINDOW_H + 20))

            pygame.display.flip()
            fps_clock.tick(TARGET_FPS)
            state = next_state

        agent.update_target_network()
        if episode % 50 == 0: torch.save(agent.policy_net.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    run_visual_dqn()