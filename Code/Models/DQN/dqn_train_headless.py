import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.PacManEnv import PacManEnv
from Code.CurriculumManager import CurriculumManager
from dqn_agent import DQNAgent
from eval_dqn import evaluate_model

# ── Configuration ──
TARGET_UPDATE_FREQUENCY = 1000
BATCH_SIZE = 64


def train():
    curriculum = CurriculumManager()

    # Egocentric 25-dim observation, 4 relative actions (F,L,R,B)
    agent = DQNAgent(input_dim=25, output_dim=4)
    print(f"Agent initialized on: {agent.device}")

    total_steps = 0
    episode = 0

    while True:
        episode += 1

        # Get current curriculum stage settings and randomize maze seed
        current_settings = curriculum.get_settings()
        dynamic_seed = int(torch.randint(0, 10_000_000, (1,)).item())
        current_settings['maze_seed'] = dynamic_seed

        env = PacManEnv(render_mode=None, **current_settings)
        state, _ = env.reset(seed=dynamic_seed)

        total_reward = 0.0
        loss_tracker = []
        done = False

        while not done:
            # Egocentric 4-action selection (0:F,1:L,2:R,3:B)
            action = agent.select_action(state, valid_actions=[0, 1, 2, 3])

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

        # Episode finished
        avg_loss = sum(loss_tracker) / len(loss_tracker) if loss_tracker else 0.0
        won = env.engine.won
        max_steps = current_settings.get('max_episode_steps')
        print(
            f"Episode {episode} | Stage: {curriculum.current_stage} | MaxSteps: {max_steps} | "
            f"Reward: {total_reward:.1f} | Won: {won} | Epsilon: {agent.epsilon:.3f} | "
            f"Steps: {total_steps} | Avg Loss: {avg_loss:.4f}"
        )

        # Update curriculum performance and possibly promote stage
        curriculum.update_performance(won)
        curriculum.check_promotion()

        env.close()

        # Periodically save
        if episode % 50 == 0:
            save_path = os.path.join(_HERE, "dqn_pacman.pth")
            torch.save(agent.policy_net.state_dict(), save_path)
            print(f"Saved weights to {save_path}")

            # Launch a short visual evaluation run using the freshly saved model
            # This will open a pygame window and play a few episodes.
            try:
                evaluate_model(save_path, episodes=2)
            except SystemExit:
                # Allow user to close the eval window without killing training loop
                pass
            except Exception as e:
                print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    train()