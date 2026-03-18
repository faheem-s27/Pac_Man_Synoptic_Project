import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    """Upgraded 33->128->128->4 MLP for egocentric raycasts."""

    def __init__(self, input_dim: int = 33, output_dim: int = 4):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, input_dim: int = 33, output_dim: int = 4, lr: float = 1e-4, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: int = 1_000_000):
        self.action_dim = output_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(input_dim, output_dim).to(self.device)
        self.target_net = QNetwork(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=200_000)

    def select_action(self, state, valid_actions=None) -> int:
        self.step_count += 1
        # Linear decay mapping steps directly to epsilon drop
        self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 / self.epsilon_decay))
        candidate_actions = valid_actions if valid_actions else list(range(self.action_dim))

        if random.random() < self.epsilon:
            return random.choice(candidate_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze(0)
            masked_q = torch.full_like(q_values, float('-inf'))
            for a in candidate_actions:
                masked_q[a] = q_values[a]
            return masked_q.argmax().item()

    def optimize_model(self, batch_size: int = 64):
        if len(self.memory) < batch_size:
            return None

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size)
        state_tensor = torch.FloatTensor(state_batch).to(self.device)
        action_tensor = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_tensor = torch.FloatTensor(reward_batch).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state_batch).to(self.device)
        done_tensor = torch.FloatTensor(done_batch).to(self.device)

        # Current Q
        current_q_values = self.policy_net(state_tensor).gather(1, action_tensor).squeeze(1)

        # -------- DOUBLE DQN --------
        with torch.no_grad():
            # Use policy network to select the best next action
            next_actions = self.policy_net(next_state_tensor).argmax(1, keepdim=True)
            # Use target network to evaluate that action
            next_q_values = self.target_net(next_state_tensor).gather(1, next_actions).squeeze(1)
            expected_q_values = reward_tensor + (self.gamma * next_q_values * (1 - done_tensor))

        # -------- HUBER LOSS --------
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        # -------- GRADIENT CLIPPING --------
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())