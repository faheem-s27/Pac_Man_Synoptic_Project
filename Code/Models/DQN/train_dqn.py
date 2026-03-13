import os
import sys
import torch
import time
import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.PacManEnv import PacManEnv
from Code.CurriculumManager import CurriculumManager
from dqn_agent import DQNAgent

# ── Configuration ──
# DQN requires massively extended episodic cycles compared to NEAT generations
EPISODES = 2000
TARGET_UPDATE_FREQUENCY = 1000
BATCH_SIZE = 64
EPISODE_TO_GEN_SCALE = 10  # 10 Episodes = 1 Curriculum Generation


def train():
    curriculum = CurriculumManager()

    # Initialize the 16-input PyTorch Agent
    agent = DQNAgent(input_dim=16, output_dim=4)
    print(f"Agent initialized on: {agent.device}")

    total_steps = 0

    for episode in range(EPISODES):
        # ACTION: Dilate the timeline and inject dynamic parameters
        virtual_generation = episode // EPISODE_TO_GEN_SCALE
        current_settings = curriculum.get_settings_for_generation(virtual_generation)

        # Instantiate environment with the specific curriculum stage settings
        env = PacManEnv(render_mode=None, obs_type="vector", settings=current_settings)
        state, _ = env.reset()

        total_reward = 0
        loss_tracker = []

        while True:
            # 1. Agent selects an action (Epsilon-Greedy)
            action = agent.select_action(state)

            # 2. Environment executes the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # 3. Store the transition in the Replay Buffer
            agent.memory.push(state, action, reward, next_state, done)

            # 4. Perform Gradient Descent
            loss = agent.optimize_model(batch_size=BATCH_SIZE)
            if loss is not None:
                loss_tracker.append(loss)

            # 5. Advance the state
            state = next_state
            total_steps += 1

            # 6. Synchronize the Target Network
            if total_steps % TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()

            if done:
                avg_loss = sum(loss_tracker) / len(loss_tracker) if loss_tracker else 0.0
                print(
                    f"Episode {episode + 1}/{EPISODES} | Virtual Gen: {virtual_generation} | Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f} | Steps: {total_steps} | Avg Loss: {avg_loss:.4f}")
                break

        # Crucial memory leak prevention between episodes
        env.close()

        # ── INTERMITTENT VISUAL EVALUATION ──
        # Every 50 episodes, pause high-speed training to render a single pure-exploitation test
        if (episode + 1) % 50 == 0:
            print(f"\n--- VISUAL EVALUATION: EPISODE {episode + 1} ---")

            # 1. Instantiate a temporary human-rendered environment
            eval_env = PacManEnv(render_mode="human", obs_type="vector", settings=current_settings)
            eval_state, _ = eval_env.reset()
            eval_done = False
            eval_reward = 0

            while not eval_done:
                # ACTION: Pump the OS event queue to prevent the window from freezing
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        eval_done = True

                eval_env.render()

                # 2. Pure Exploitation (No Epsilon Randomness)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(eval_state).unsqueeze(0).to(agent.device)
                    q_values = agent.policy_net(state_tensor)
                    eval_action = q_values.argmax(dim=1).item()

                eval_next_state, reward, eval_terminated, eval_truncated, _ = eval_env.step(eval_action)
                eval_done = eval_terminated or eval_truncated
                eval_state = eval_next_state
                eval_reward += reward

                # Throttle the loop slightly so human eyes can track the movement
                time.sleep(0.03)

            eval_env.close()
            print(f"--- EVALUATION COMPLETE | SCORE: {eval_reward:.1f} | RESUMING HIGH-SPEED TRAINING ---\n")

    # Save the final optimized weights
    torch.save(agent.policy_net.state_dict(), os.path.join(_HERE, "dqn_pacman.pth"))
    print("Training Complete. Model saved to dqn_pacman.pth")


if __name__ == "__main__":
    train()