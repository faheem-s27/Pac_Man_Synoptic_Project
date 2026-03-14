import os
import sys
import torch
import pygame
import numpy as np

# Path resolution for imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.PacManEnv import PacManEnv
from dqn_agent import QNetwork


def evaluate_model(model_path, episodes=5):
    # CRITICAL: Initialize Pygame and the Clock
    pygame.init()
    fps_clock = pygame.time.Clock()
    TARGET_FPS = 1  # Slowed down for high-level observation

    # 1. Initialize Env (Render mode must be None so WE can control the drawing)
    env = PacManEnv(render_mode=None, obs_type="vector")

    DASHBOARD_W = 400
    win_w = 800 + DASHBOARD_W
    win_h = 860  # 800 for game + 60 for info bar

    # Master Display
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("DQN Egocentric Evaluation Dashboard")

    dash_font = pygame.font.Font(None, 24)
    info_font = pygame.font.Font(None, 28)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Rebuild architecture (27 In, 3 Out)
    policy_net = QNetwork(input_dim=27, output_dim=3).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    policy_net.eval()

    print(f"Zero-Shot Evaluation Started. Target FPS: {TARGET_FPS}")

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # --- 1. HANDLE EVENTS (Safety Exit & Screenshots) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit();
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    pygame.image.save(screen, f"eval_ep{episode}_step{env._step_count}.png")

            # --- 2. AI DECISION ---
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor).squeeze(0).cpu().numpy()
                action = np.argmax(q_values)

            # --- 3. PHYSICS UPDATE ---
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # --- 4. RENDER GAME TO LEFT SIDE ---
            game_surf = pygame.Surface((env.engine.screen_width, env.engine.screen_height))
            game_surf.fill((0, 0, 0))
            env.engine.draw(game_surf)
            # Draw the sensor lines on the game_surf
            env._screen = game_surf
            env._draw_debug_sensors()

            # Scaled game display (800x800)
            screen.blit(pygame.transform.scale(game_surf, (800, 800)), (0, 0))

            # --- 5. RENDER DASHBOARD TO RIGHT SIDE ---
            screen.fill((15, 15, 20), (800, 0, DASHBOARD_W, win_h))
            action_names = ['FORWARD', 'TURN LEFT', 'TURN RIGHT']

            lines = [
                "=== NEURAL LOGIC (EGOCENTRIC) ===",
                f"Chosen Action: {action_names[action]}",
                "",
                "--- Relative Q-Values ---",
                f"FORW: {q_values[0]:+08.2f}",
                f"LEFT: {q_values[1]:+08.2f}",
                f"RGHT: {q_values[2]:+08.2f}",
                "",
                "--- Ego-Sensors (Wall|Food|Ghost) ---",
                "Dir | W-Dist | F-Dist | G-Dist"
            ]

            labels = ["F", "B", "L", "R", "FL", "FR", "BL", "BR"]
            for i, lab in enumerate(labels):
                w, f, g = state[i * 3], state[i * 3 + 1], state[i * 3 + 2]
                lines.append(f"{lab:<3} | {w:6.2f} | {f:6.2f} | {g:+6.2f}")

            lines.extend([
                "",
                "--- Nav Target ---",
                f"Relative X: {state[24]:+4.2f}",
                f"Relative Y: {state[25]:+4.2f}",
                "",
                f"Fright Mode: {'ACTIVE' if state[26] > 0.5 else 'OFF'}"
            ])

            y_pos = 20
            for line in lines:
                color = (255, 215, 0) if "---" in line or "===" in line else (255, 255, 255)
                if "Chosen Action" in line: color = (0, 255, 0)
                screen.blit(dash_font.render(line, True, color), (815, y_pos))
                y_pos += 24

            # --- 6. BOTTOM INFO BAR ---
            screen.fill((25, 25, 30), (0, 800, win_w, 60))
            info_str = f"EPISODE: {episode + 1} | TOTAL REWARD: {total_reward:.1f} | STEP: {env._step_count}"
            screen.blit(info_font.render(info_str, True, (255, 215, 0)), (20, 818))

            # --- 7. FINAL DISPLAY FLIP & CLOCK TICK ---
            pygame.display.flip()
            fps_clock.tick(TARGET_FPS)  # This is the ONLY thing that controls speed

            state = next_state
            done = terminated or truncated

    env.close()


if __name__ == "__main__":
    model_file = os.path.join(_HERE, "dqn_pacman.pth")
    if os.path.exists(model_file):
        evaluate_model(model_file)
    else:
        print("ERROR: dqn_pacman.pth missing.")