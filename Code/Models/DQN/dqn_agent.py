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
    21 -> 256 -> 256 -> 128 -> 4
    """

    def __init__(self, input_dim=21, output_dim=4):
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

    def push(self, state, action, reward, next_state, done, next_valid_mask=None, discount_pow=1.0):
        if next_valid_mask is None:
            next_valid_mask = np.ones(4, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done, next_valid_mask, discount_pow))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_valid_masks, discount_pows = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(next_valid_masks, dtype=np.float32),
            np.array(discount_pows, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =========================
# DQN Agent
# =========================
class DQNAgent:
    def __init__(
        self,
        input_dim=21,
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
        # Curriculum exploration-jolt state.
        self.epsilon_jolt_value = epsilon_end
        self.epsilon_jolt_until_step = 0

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
    def select_action(self, state, valid_actions=None, return_exploration=False):
        self.step_count += 1

        # Base schedule.
        scheduled_epsilon = max(
            self.epsilon_end,
            self.epsilon_end
            + (self.epsilon_start - self.epsilon_end)
            * np.exp(-self.step_count / self.epsilon_decay),
        )

        # Optional temporary floor from curriculum transitions.
        if self.step_count <= self.epsilon_jolt_until_step:
            self.epsilon = max(scheduled_epsilon, self.epsilon_jolt_value)
        else:
            self.epsilon = scheduled_epsilon

        candidate_actions = valid_actions if valid_actions else list(range(self.action_dim))

        exploring = random.random() < self.epsilon
        if exploring:
            action = random.choice(candidate_actions)
            return (action, True) if return_exploration else action

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze(0)

            # Mask invalid actions
            masked_q = torch.full_like(q_values, float("-inf"))
            for a in candidate_actions:
                masked_q[a] = q_values[a]

            action = masked_q.argmax().item()
            return (action, False) if return_exploration else action

    def apply_exploration_jolt(self, min_epsilon=0.2, duration_steps=50_000):
        """Temporarily enforce a minimum epsilon after curriculum transitions."""
        self.epsilon_jolt_value = max(self.epsilon_end, float(min_epsilon))
        self.epsilon_jolt_until_step = max(self.epsilon_jolt_until_step, self.step_count + int(duration_steps))

    # =========================
    # Training Step
    # =========================
    def optimize_model(self, batch_size=None):
        # Warmup
        if len(self.memory) < self.warmup_steps:
            return None

        # Train less frequently
        if self.step_count % self.train_freq != 0:
            return None

        # Sample
        bs = int(batch_size) if batch_size is not None else self.batch_size
        states, actions, rewards, next_states, dones, next_valid_masks, discount_pows = self.memory.sample(bs)

        state_tensor = torch.tensor(states).to(self.device)
        action_tensor = torch.tensor(actions).unsqueeze(1).to(self.device)
        reward_tensor = torch.tensor(rewards).to(self.device)
        next_state_tensor = torch.tensor(next_states).to(self.device)
        done_tensor = torch.tensor(dones).to(self.device)
        next_valid_mask_tensor = torch.tensor(next_valid_masks).to(self.device)
        discount_pow_tensor = torch.tensor(discount_pows).to(self.device)

        # Current Q
        current_q = self.policy_net(state_tensor).gather(1, action_tensor).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            policy_next_q = self.policy_net(next_state_tensor)
            target_next_q_all = self.target_net(next_state_tensor)

            valid_bool = next_valid_mask_tensor > 0.5
            all_invalid = ~valid_bool.any(dim=1)

            masked_policy_next_q = policy_next_q.masked_fill(~valid_bool, float("-inf"))
            next_actions = masked_policy_next_q.argmax(1, keepdim=True)

            masked_target_next_q = target_next_q_all.masked_fill(~valid_bool, float("-inf"))
            next_q = masked_target_next_q.gather(1, next_actions).squeeze(1)
            next_q = torch.where(all_invalid, torch.zeros_like(next_q), next_q)

            target_q = reward_tensor + discount_pow_tensor * next_q * (1 - done_tensor)

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