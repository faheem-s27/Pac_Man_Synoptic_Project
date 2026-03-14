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
TARGET_UPDATE_FREQUENCY = 1000
BATCH_SIZE = 64
STEPS_PER_CURRICULUM_STAGE = 200


def train():
    curriculum = CurriculumManager(stage_duration=STEPS_PER_CURRICULUM_STAGE)

    agent = DQNAgent(input_dim=27, output_dim=3)
    print(f"Agent initialized on: {agent.device}")

    total_steps = 0
    episode = 0

    while True:
        episode += 1

        virtual_generation = (episode - 1) // STEPS_PER_CURRICULUM_STAGE
        current_settings = curriculum.get_settings_for_generation(virtual_generation)

        env = PacManEnv(render_mode=None, obs_type="vector", settings=current_settings)
        state, _ = env.reset()

        total_reward = 0
        loss_tracker = []

        while True:
            # Relative 3-action selection (no explicit masking; env handles walls)
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            agent.memory.push(state, action, reward, next_state, done)

            loss = agent.optimize_model(batch_size=BATCH_SIZE)
            if loss is not None:
                loss_tracker.append(loss)

            state = next_state
            total_steps += 1

            if total_steps % TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()

            if done:
                avg_loss = sum(loss_tracker) / len(loss_tracker) if loss_tracker else 0.0
                print(
                    f"Episode {episode} | Virtual Gen: {virtual_generation} | Reward: {total_reward:.1f} | "
                    f"Epsilon: {agent.epsilon:.3f} | Steps: {total_steps} | Avg Loss: {avg_loss:.4f}")
                break

        env.close()

        # Periodically save
        if episode % 50 == 0:
            torch.save(agent.policy_net.state_dict(), os.path.join(_HERE, "dqn_pacman.pth"))


if __name__ == "__main__":
    train()