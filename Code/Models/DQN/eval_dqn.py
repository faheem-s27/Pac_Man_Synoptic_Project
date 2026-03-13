import os
import sys
import torch
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.PacManEnv import PacManEnv
from dqn_agent import QNetwork


def evaluate_model(model_path, episodes=5):
    # 1. Initialize the hardest environment (No Curriculum, Render = Human)
    # This forces the agent to prove it can generalize to the full game
    env = PacManEnv(render_mode="human", obs_type="vector")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Rebuild the skeletal architecture
    policy_net = QNetwork(input_dim=24, output_dim=4).to(device)

    # 3. Inject the optimized weights from the .pth file
    policy_net.load_state_dict(torch.load(model_path, map_location=device))

    # 4. Lock the network into deterministic evaluation mode (disables dropouts/gradients)
    policy_net.eval()

    print(f"Model loaded successfully from {model_path}. Commencing Zero-Shot Evaluation.")

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Render the frame so you can physically watch the AI's logic
            env.render()

            # 5. Pure Exploitation (No Epsilon, No Randomness)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                # Select the action with the highest Q-value
                action = q_values.argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            done = terminated or truncated

            # Slow down the render slightly so it is human-observable
            #time.sleep(0.01)

        print(f"Evaluation Episode {episode + 1} | Final Score: {total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    model_file = os.path.join(_HERE, "dqn_pacman.pth")
    if os.path.exists(model_file):
        evaluate_model(model_file)
    else:
        print("ERROR: dqn_pacman.pth not found. You must finish training first.")