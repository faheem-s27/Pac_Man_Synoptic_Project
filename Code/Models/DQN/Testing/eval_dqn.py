import os
import sys
import torch
import pygame
import numpy as np

# Path resolution for imports
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DQN_ROOT = os.path.dirname(_HERE)                          # Code/Models/DQN/
_ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))  # project root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _DQN_ROOT not in sys.path:
    sys.path.insert(0, _DQN_ROOT)   # dqn_agent

from Code.Environment.PacManEnv import PacManEnv
from dqn_agent import DuelingQNetwork


def evaluate_model(model_path, episodes=3):
    """Visual evaluation for the 27-dim egocentric DQN (4 actions)."""
    pygame.init()
    fps_clock = pygame.time.Clock()
    TARGET_FPS = 10

    env = PacManEnv(render_mode=None, max_episode_steps=None)

    win_w = env.engine.screen_width
    win_h = env.engine.screen_height + 60
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("DQN Evaluation — Egocentric 27D")

    info_font = pygame.font.Font(None, 28)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 29-dim in (4 dirs × 6 ray channels + 3 BFS + 2 power), 4 egocentric actions out
    policy_net = DuelingQNetwork(input_dim=29, output_dim=4).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    policy_net.eval()

    print(f"Evaluation started using {device}.")

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor).squeeze(0).cpu().numpy()
                valid_actions = env.get_valid_actions()
                action = max(valid_actions, key=lambda a: q_values[a]) if valid_actions else int(np.argmax(q_values))

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            game_surf = pygame.Surface((env.engine.screen_width, env.engine.screen_height))
            game_surf.fill((0, 0, 0))
            env.engine.draw(game_surf)
            screen.blit(game_surf, (0, 0))

            screen.fill((25, 25, 30), (0, env.engine.screen_height, win_w, 60))
            action_names = ['FORWARD', 'LEFT', 'RIGHT', 'BACKWARD']
            info_str = (
                f"EP: {episode + 1} | Step: {env._step_count} | R: {total_reward:+.2f} | "
                f"Act: {action_names[action]} | Q: " + ", ".join(f"{v:+.2f}" for v in q_values)
            )
            screen.blit(info_font.render(info_str, True, (255, 215, 0)), (10, env.engine.screen_height + 20))

            pygame.display.flip()
            fps_clock.tick(TARGET_FPS)

            state = next_state
            done = terminated or truncated

    env.close()


if __name__ == "__main__":
    model_file = os.path.join(_DQN_ROOT, "Checkpoints", "dqn_pacman.pth")
    if os.path.exists(model_file):
        evaluate_model(model_file)
    else:
        print("ERROR: dqn_pacman.pth missing.")
