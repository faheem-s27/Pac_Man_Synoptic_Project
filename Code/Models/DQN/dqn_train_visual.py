"""
dqn_train_visual.py
===================
A real-time, single-agent visual training monitor tailored specifically
for the PyTorch DQNAgent class.

Controls
--------
  ESC / Close Window → Stop training safely and save the dqn_pacman.pth weights.
"""

import sys
import os
import pygame
import torch
import random

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
DASHBOARD_W  = 400  # ACTION: Width of the new neural network HUD
INFO_BAR_H   = 60
TARGET_FPS   = 120
SAVE_PATH    = os.path.join(_HERE, "dqn_pacman.pth")

def run_visual_dqn():
    pygame.init()

    # Expand window width to accommodate the HUD
    win_w = MAX_WINDOW_W + DASHBOARD_W
    win_h = MAX_WINDOW_H + INFO_BAR_H
    window = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("DQN Pac-Man — Real-Time Visual Training & Telemetry")

    info_font = pygame.font.Font(None, 28)
    dash_font = pygame.font.Font(None, 24)
    fps_clock = pygame.time.Clock()

    curriculum = CurriculumManager()
    base_settings = curriculum.get_settings_for_generation(0)

    # ACTION: Ensure input_dim is set to 24 for the expanded vector field
    env = PacManEnv(render_mode=None, **base_settings)
    agent = DQNAgent(input_dim=24, output_dim=4)

    if os.path.exists(SAVE_PATH):
        try:
            agent.policy_net.load_state_dict(torch.load(SAVE_PATH, map_location=agent.device))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Loaded existing weights from {SAVE_PATH}")
        except Exception as e:
            print(f"Failed to load weights. (Did you change input_dim?). Starting fresh. Error: {e}")

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
            # --- EVENT PUMP & SAFE EXIT ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    print(f"\nTraining visually interrupted. Saving PyTorch weights to {SAVE_PATH}...")
                    torch.save(agent.policy_net.state_dict(), SAVE_PATH)
                    pygame.quit()
                    sys.exit()

            # --- DQN LOGIC WITH ACTION COMMITMENT ---
            pacman = env._engine.pacman
            ts = env._engine.tile_size

            is_aligned = (pacman.x % ts == 0) and (pacman.y % ts == 0)
            is_stopped = (pacman.direction == (0, 0))

            if is_aligned or is_stopped or env._last_action is None:
                action = agent.select_action(state)
            else:
                action = env._last_action

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            total_reward += reward

            loss = agent.optimize_model(batch_size)
            if loss is not None:
                loss_history.append(loss)

            # --- RENDERING (GAME) ---
            game_surf = pygame.Surface((env._engine.screen_width, env._engine.screen_height))
            game_surf.fill((0, 0, 0))
            env._engine.draw(game_surf)
            env._screen = game_surf
            env._draw_debug_sensors()

            scaled_game = pygame.transform.scale(game_surf, (MAX_WINDOW_W, MAX_WINDOW_H))
            window.blit(scaled_game, (0, 0))

            # --- RENDERING (TELEMETRY DASHBOARD) ---
            # Fill the dashboard background
            dash_rect = (MAX_WINDOW_W, 0, DASHBOARD_W, win_h)
            window.fill((15, 15, 20), dash_rect)
            pygame.draw.line(window, (100, 100, 100), (MAX_WINDOW_W, 0), (MAX_WINDOW_W, win_h), 2)

            # ACTION: Intercept the Q-Values for visual rendering
            with torch.no_grad():
                st_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_vals = agent.policy_net(st_tensor).squeeze().cpu().numpy()

            # Formatting the text layout
            action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            lines = [
                "=== NEURAL NETWORK STATE ===",
                f"Action Taken: {action_names[action]}",
                "",
                "--- Q-Values (Expectation) ---",
                f"UP:    {q_vals[0]:+08.2f}",
                f"DOWN:  {q_vals[1]:+08.2f}",
                f"LEFT:  {q_vals[2]:+08.2f}",
                f"RIGHT: {q_vals[3]:+08.2f}",
                "",
                "--- Sensor Array (24 Elements) ---",
                f"Pac Dir:   [{state[0]:+0.2f}, {state[1]:+0.2f}]",
                f"Ghost 1:   [{state[2]:+0.2f}, {state[3]:+0.2f}]  T:{state[4]:+0.1f}",
                f"Ghost 2:   [{state[5]:+0.2f}, {state[6]:+0.2f}]  T:{state[7]:+0.1f}",
                f"Pellet 1:  [{state[8]:+0.2f}, {state[9]:+0.2f}]",
                f"Pellet 2:  [{state[10]:+0.2f}, {state[11]:+0.2f}]",
                f"Pellet 3:  [{state[12]:+0.2f}, {state[13]:+0.2f}]",
                f"Pellet 4:  [{state[14]:+0.2f}, {state[15]:+0.2f}]",
                f"Pellet 5:  [{state[16]:+0.2f}, {state[17]:+0.2f}]",
                f"Wall Ray:  U:{state[18]:.2f} D:{state[19]:.2f} L:{state[20]:.2f} R:{state[21]:.2f}",
                f"Power Pel: [{state[22]:+0.2f}, {state[23]:+0.2f}]"
            ]

            # Blit the text to the side panel
            y_offset = 20
            for line in lines:
                color = (255, 255, 255)
                if "NEURAL" in line or "Q-Values" in line or "Sensor" in line:
                    color = (255, 215, 0) # Gold headers
                elif "Action Taken" in line:
                    color = (0, 255, 0)   # Green action

                text_surf = dash_font.render(line, True, color)
                window.blit(text_surf, (MAX_WINDOW_W + 15, y_offset))
                y_offset += 25

            # --- RENDERING (BOTTOM INFO BAR) ---
            window.fill((20, 20, 20), (0, MAX_WINDOW_H, MAX_WINDOW_W, INFO_BAR_H))
            avg_loss = sum(loss_history)/len(loss_history) if loss_history else 0.0

            info_text = (f"Ep: {episode} | Step: {env._step_count}/{env.max_episode_steps} | "
                         f"Rwd: {total_reward:+.1f} | Eps: {agent.epsilon:.3f} | Loss: {avg_loss:.2f}")

            text_surf = info_font.render(info_text, True, (255, 215, 0))
            window.blit(text_surf, (15, MAX_WINDOW_H + 20))

            pygame.display.flip()
            fps_clock.tick(TARGET_FPS)

            state = next_state

        agent.update_target_network()

        if episode % 50 == 0:
            torch.save(agent.policy_net.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    run_visual_dqn()