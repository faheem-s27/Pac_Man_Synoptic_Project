import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque


# =========================
# Q-Network
# =========================
class QNetwork(nn.Module):
    """
    Improved network:
    36 -> 256 -> 256 -> 128 -> 4
    """

    def __init__(self, input_dim=36, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# =========================
# Replay Buffer
# =========================
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
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =========================
# DQN Agent
# =========================
class DQNAgent:
    def __init__(
        self,
        input_dim=36,
        output_dim=4,
        lr=3e-4,
        gamma=0.995,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=2_000_000,
    ):
        self.action_dim = output_dim
        self.gamma = gamma

        # Epsilon
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.step_count = 0

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = QNetwork(input_dim, output_dim).to(self.device)
        self.target_net = QNetwork(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer(capacity=200_000)

        # Training config
        self.batch_size = 256
        self.train_freq = 4
        self.warmup_steps = 10_000
        self.tau = 0.005  # soft update

    # =========================
    # Action Selection
    # =========================
    def select_action(self, state, valid_actions=None):
        self.step_count += 1

        # Stable epsilon decay
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_end
            + (self.epsilon_start - self.epsilon_end)
            * np.exp(-self.step_count / self.epsilon_decay),
        )

        candidate_actions = valid_actions if valid_actions else list(range(self.action_dim))

        if random.random() < self.epsilon:
            return random.choice(candidate_actions)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze(0)

            # Mask invalid actions
            masked_q = torch.full_like(q_values, float("-inf"))
            for a in candidate_actions:
                masked_q[a] = q_values[a]

            return masked_q.argmax().item()

    # =========================
    # Training Step
    # =========================
    def optimize_model(self):
        # Warmup
        if len(self.memory) < self.warmup_steps:
            return None

        # Train less frequently
        if self.step_count % self.train_freq != 0:
            return None

        # Sample
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        state_tensor = torch.tensor(states).to(self.device)
        action_tensor = torch.tensor(actions).unsqueeze(1).to(self.device)
        reward_tensor = torch.tensor(rewards).to(self.device)
        next_state_tensor = torch.tensor(next_states).to(self.device)
        done_tensor = torch.tensor(dones).to(self.device)

        # Current Q
        current_q = self.policy_net(state_tensor).gather(1, action_tensor).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.policy_net(next_state_tensor).argmax(1, keepdim=True)
            next_q = self.target_net(next_state_tensor).gather(1, next_actions).squeeze(1)
            target_q = reward_tensor + self.gamma * next_q * (1 - done_tensor)

        # Huber loss
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)

        self.optimizer.step()

        return loss.item()

    # =========================
    # Soft Target Update
    # =========================
    def update_target_network(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )