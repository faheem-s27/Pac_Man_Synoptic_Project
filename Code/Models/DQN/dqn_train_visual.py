"""
dqn_train_visual.py
===================
Egocentric 25-input DQN visual trainer.
"""

import sys
import os
import pygame
import torch
import random
import csv

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
    pygame.display.set_caption("DQN Pac-Man — Egocentric 25D")

    info_font = pygame.font.Font(None, 28)
    dash_font = pygame.font.Font(None, 22)
    header_font = pygame.font.Font(None, 26)
    fps_clock = pygame.time.Clock()

    curriculum = CurriculumManager()
    base_settings = curriculum.get_settings()

    env = PacManEnv(render_mode=None, **base_settings)
    agent = DQNAgent(input_dim=25, output_dim=4)

    if os.path.exists(SAVE_PATH):
        try:
            agent.policy_net.load_state_dict(torch.load(SAVE_PATH, map_location=agent.device, weights_only=True))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Loaded existing weights from {SAVE_PATH}")
        except Exception as e:
            print(f"Failed to load weights. Starting fresh. Error: {e}")

    batch_size = 64
    episode = 0

    # Ensure CSV log file exists with header
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Episode",
                "Stage",
                "Maze_Seed",
                "Reward",
                "Steps",
                "Outcome",
                "Epsilon",
                "Pellets",
                "Power_Pellets",
                "Ghosts",
                "Explore_Rate",
                "Avg_Loss",
            ])

    while True:
        episode += 1

        current_settings = curriculum.get_settings()
        dynamic_seed = random.randint(0, 9999999)
        current_settings['maze_seed'] = dynamic_seed

        # Sync environment telemetry with curriculum stage for logging
        env.current_stage = getattr(curriculum, "current_stage", None)

        env._base_cfg.update(current_settings)
        env.max_episode_steps = current_settings.get('max_episode_steps', 2000)

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
                    pygame.quit()
                    sys.exit()

            # DQN step
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

            # Draw every step for smooth visuals
            game_surf = pygame.Surface((env.engine.screen_width, env.engine.screen_height))
            game_surf.fill((0, 0, 0))
            env.engine.draw(game_surf)
            window.blit(pygame.transform.scale(game_surf, (MAX_WINDOW_W, MAX_WINDOW_H)), (0, 0))

            window.fill((15, 15, 20), (MAX_WINDOW_W, 0, DASHBOARD_W, win_h))

            with torch.no_grad():
                st_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_vals = agent.policy_net(st_tensor).squeeze().cpu().numpy()

            action_names = ['FORWARD', 'LEFT', 'RIGHT', 'BACKWARD']

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

            y_off += 10
            window.blit(header_font.render("--- Egocentric Rays (W|F|G) ---", True, (255, 215, 0)), (MAX_WINDOW_W + 15, y_off))
            y_off += 25

            labels = ['F', 'B', 'L', 'R', 'FL', 'FR', 'BL', 'BR']
            for i, lab in enumerate(labels):
                w, f, g = state[i*3], state[i*3+1], state[i*3+2]
                line = f"{lab:<3} | W:{w:6.2f} F:{f:6.2f} G:{g:6.2f}"
                window.blit(dash_font.render(line, True, (255, 255, 255)), (MAX_WINDOW_W + 15, y_off))
                y_off += 22

            y_off += 10
            # Frightened timer observation: 1.0 at start of frightened mode, down to 0.0 as it expires
            fright_flag = state[24]
            window.blit(header_font.render(f"Fright timer: {fright_flag:.2f}", True, (255, 105, 180)), (MAX_WINDOW_W + 15, y_off))

            window.fill((20, 20, 20), (0, MAX_WINDOW_H, MAX_WINDOW_W, INFO_BAR_H))
            avg_loss = sum(loss_history)/len(loss_history) if loss_history else 0.0
            info = f"Ep: {episode} | Step: {env._step_count} | Rwd: {total_reward:+.1f} | Eps: {agent.epsilon:.3f} | Loss: {avg_loss:.2f}"
            window.blit(info_font.render(info, True, (255, 215, 0)), (15, MAX_WINDOW_H + 20))

            pygame.display.flip()

            fps_clock.tick(TARGET_FPS)
            state = next_state

        won = env.engine.won
        curriculum.update_performance(won)
        curriculum.check_promotion()

        # Per-episode logging to CSV
        avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0.0

        # Extract telemetry fields from final info dict (fallbacks if missing)
        stage = last_info.get("stage") if isinstance(last_info, dict) else None
        maze_seed = last_info.get("maze_seed") if isinstance(last_info, dict) else None
        pellets = int(last_info.get("pellets", 0)) if isinstance(last_info, dict) else 0
        power_pellets = int(last_info.get("power_pellets", 0)) if isinstance(last_info, dict) else 0
        ghosts = int(last_info.get("ghosts", 0)) if isinstance(last_info, dict) else 0
        explore_rate = float(last_info.get("explore_rate", 0.0)) if isinstance(last_info, dict) else 0.0

        with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                stage,
                maze_seed,
                float(total_reward),
                int(episode_steps),
                last_death_cause,
                float(agent.epsilon),
                pellets,
                power_pellets,
                ghosts,
                explore_rate,
                float(avg_loss),
            ])

        agent.update_target_network()
        if episode % 50 == 0:
            torch.save(agent.policy_net.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    run_visual_dqn()