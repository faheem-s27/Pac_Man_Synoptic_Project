"""
dqn_train_visual.py
===================
A real-time monitor tailored for the Dense 11x11 Grid (122-input) Architecture.
Features a fully colored, monospaced diagnostic matrix.
"""

import sys
import os
import pygame
import torch
import random
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

def map_value_to_char(val):
    """Converts the raw float to an ASCII character for the diagnostic mini-map."""
    if val == -1.0: return '#'  # Wall
    if val == -0.8: return 'G'  # Lethal Ghost
    if val == 0.0:  return '.'  # Empty floor
    if val == 0.5:  return 'o'  # Pellet
    if val == 0.8:  return 'F'  # Frightened Ghost
    if val == 1.0:  return 'O'  # Power Pellet
    return '?'

def get_color_for_char(char):
    """Maps the ASCII character to a high-contrast RGB tuple."""
    if char == '#': return (50, 100, 255)   # Deep Blue (Walls)
    if char == '.': return (70, 70, 70)     # Dark Gray (Floor)
    if char == 'o': return (255, 255, 0)    # Yellow (Pellets)
    if char == 'O': return (0, 255, 255)    # Cyan (Power Pellets)
    if char == 'G': return (255, 0, 0)      # Red (Lethal Ghosts)
    if char == 'F': return (255, 105, 180)  # Pink (Frightened)
    if char == 'P': return (0, 255, 0)      # Neon Green (Pac-Man)
    return (255, 255, 255)

def run_visual_dqn():
    pygame.init()

    win_w = MAX_WINDOW_W + DASHBOARD_W
    win_h = MAX_WINDOW_H + INFO_BAR_H
    window = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("DQN Pac-Man — DENSE 11x11 COLOR MATRIX")

    info_font = pygame.font.Font(None, 28)
    # CRITICAL: Must use SysFont('monospace') to ensure the matrix aligns perfectly
    dash_font = pygame.font.SysFont('monospace', 22, bold=True)
    header_font = pygame.font.Font(None, 26)
    fps_clock = pygame.time.Clock()

    curriculum = CurriculumManager(stage_duration=1000)
    base_settings = curriculum.get_settings_for_generation(0)

    # ACTION: 122 inputs, 4 absolute outputs
    env = PacManEnv(render_mode=None, **base_settings)
    agent = DQNAgent(input_dim=122, output_dim=4)

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

            # The environment handles collision now; pass all 4 absolute actions
            action = agent.select_action(state, valid_actions=[0, 1, 2, 3])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            total_reward += reward

            loss = agent.optimize_model(batch_size)
            if loss is not None: loss_history.append(loss)

            game_surf = pygame.Surface((env.engine.screen_width, env.engine.screen_height))
            game_surf.fill((0, 0, 0))
            env.engine.draw(game_surf)
            window.blit(pygame.transform.scale(game_surf, (MAX_WINDOW_W, MAX_WINDOW_H)), (0, 0))

            # --- DIAGNOSTIC DASHBOARD ---
            window.fill((15, 15, 20), (MAX_WINDOW_W, 0, DASHBOARD_W, win_h))
            with torch.no_grad():
                st_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_vals = agent.policy_net(st_tensor).squeeze().cpu().numpy()

            action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

            y_off = 20
            window.blit(header_font.render("=== DQN LOGIC (ABSOLUTE) ===", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 30
            window.blit(header_font.render(f"Decision: {action_names[action]}", True, (0, 255, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 30

            window.blit(header_font.render("--- Q-Values ---", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 25
            for i, name in enumerate(action_names):
                window.blit(dash_font.render(f"{name:<5}: {q_vals[i]:+08.2f}", True, (255, 255, 255)), (MAX_WINDOW_W + 15, y_off))
                y_off += 25

            y_off += 10
            window.blit(header_font.render("--- 11x11 Local Grid ---", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 25

            # --- COLOR MATRIX RENDERER ---
            grid_slice = state[:121].reshape((11, 11))
            char_width = dash_font.size("#")[0] + 4 # Pixel width per character + spacing padding

            for r in range(11):
                curr_x = MAX_WINDOW_W + 15
                for c in range(11):
                    if r == 5 and c == 5:
                        char = 'P' # Pac-Man is always center
                    else:
                        char = map_value_to_char(grid_slice[r][c])

                    color = get_color_for_char(char)
                    char_surf = dash_font.render(char, True, color)
                    window.blit(char_surf, (curr_x, y_off))

                    curr_x += char_width # Step to the right for the next column
                y_off += 20 # Step down for the next row

            y_off += 15
            fright_flag = state[121]
            window.blit(header_font.render(f"Fright: {fright_flag:.2f}", True, (255, 105, 180)), (MAX_WINDOW_W + 15, y_off))

            # --- INFO BAR ---
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