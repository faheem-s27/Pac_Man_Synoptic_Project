"""
dqn_agent.py
============
Improved DQN agent for the 27-dimensional egocentric Pac-Man environment.

Key improvements over the original 21D version:
  1. PrioritizedReplayBuffer — numpy circular buffer (O(1) indexing) with
     SumTree-based prioritized sampling and IS-weight correction.
     Replaces Python deque which had O(n) random access.
  2. DuelingQNetwork — separate Value and Advantage streams.
     Q = V(s) + (A(s,a) - mean_a A(s,a))
     Better at learning state-value independent of action choice.
  3. Gradient clipping reduced 10 → 1.0 for training stability, especially
     given the wide raw reward range (-500 to +1000).
  4. RunningMeanStd reward normalization — rewards are normalized to zero
     mean / unit variance at train time, stabilizing Q-value targets.
  5. LR scheduler — linear decay from 3e-4 to 1e-4 over 1M optimizer steps,
     then held constant. Allows aggressive early learning + fine-tuning later.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random

# Default observation size matching PacManEnv.OBS_SIZE
OBS_DIM = 29


# =============================================================================
# Running Mean/Std (Welford's online algorithm)
# Used to normalize rewards at train time for stable Q-value targets.
# =============================================================================
class RunningMeanStd:
    """Tracks running mean and variance of a scalar stream."""

    def __init__(self, epsilon: float = 1e-8):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        batch_mean  = float(np.mean(values))
        batch_var   = float(np.var(values))
        batch_count = len(values)

        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean += delta * batch_count / tot_count
        m_a = self.var   * self.count
        m_b = batch_var  * batch_count
        M2  = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var   = M2 / tot_count
        self.count = tot_count

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize to approx zero-mean unit-variance, clipped to [-10, 10]."""
        return np.clip(
            (values - self.mean) / (np.sqrt(self.var) + 1e-8),
            -10.0, 10.0,
        ).astype(np.float32)


# =============================================================================
# SumTree — O(log n) priority-proportional sampling
# =============================================================================
class SumTree:
    """Binary sum-tree backing the prioritized replay buffer."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        # Tree has (capacity - 1) internal nodes + capacity leaf nodes.
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write_ptr = 0
        self.size = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, idx: int, priority: float) -> None:
        """Update priority at leaf index idx (tree index, not data index)."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float) -> int:
        """Insert a new priority; returns the tree leaf index."""
        tree_idx = self.write_ptr + self.capacity - 1
        self.update(tree_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return tree_idx

    def get(self, s: float) -> tuple[int, int, float]:
        """Sample by cumulative sum s; returns (tree_idx, data_idx, priority)."""
        idx = 0
        while True:
            left  = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            # Descend toward left child unless s exceeds its sum.
            if s <= self.tree[left] or self.tree[right] == 0:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, data_idx, float(self.tree[idx])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        leaf_start = self.capacity - 1
        leaf_end   = leaf_start + max(1, self.size)
        return float(np.max(self.tree[leaf_start:leaf_end]))


# =============================================================================
# Prioritized Replay Buffer
# Stores transitions in pre-allocated numpy arrays for O(1) indexed access.
# =============================================================================
class PrioritizedReplayBuffer:
    """
    Experience replay buffer with priority-proportional sampling.

    Transitions with high TD error are sampled more often.
    Importance-sampling (IS) weights correct for the induced bias.

    Parameters
    ----------
    capacity    : Maximum number of transitions stored.
    obs_dim     : Observation vector size (must match env obs space).
    action_dim  : Number of discrete actions (for valid-mask storage).
    alpha       : Priority exponent — 0 = uniform, 1 = full prioritization.
    beta_start  : IS weight exponent at training start (annealed to beta_end).
    beta_end    : IS weight exponent at end of annealing.
    beta_steps  : Number of sample() calls over which beta is annealed.
    """

    def __init__(
        self,
        capacity:   int   = 200_000,
        obs_dim:    int   = OBS_DIM,
        action_dim: int   = 4,
        alpha:      float = 0.6,
        beta_start: float = 0.4,
        beta_end:   float = 1.0,
        beta_steps: int   = 2_000_000,
    ):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.beta_steps = beta_steps
        self._beta_step = 0

        self.tree = SumTree(capacity)

        # Pre-allocated storage arrays — no per-push memory allocation.
        self.states           = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions          = np.zeros(capacity,               dtype=np.int64)
        self.rewards          = np.zeros(capacity,               dtype=np.float32)
        self.next_states      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.dones            = np.zeros(capacity,               dtype=np.float32)
        self.next_valid_masks = np.ones((capacity, action_dim),  dtype=np.float32)
        self.discount_pows    = np.ones(capacity,                dtype=np.float32)

        self._write_ptr = 0
        self._size      = 0

    # ── Beta schedule ─────────────────────────────────────────────────────────

    @property
    def beta(self) -> float:
        frac = min(1.0, self._beta_step / max(1, self.beta_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    # ── Buffer operations ─────────────────────────────────────────────────────

    def push(
        self,
        state,
        action:         int,
        reward:         float,
        next_state,
        done:           bool,
        next_valid_mask = None,
        discount_pow:   float = 1.0,
    ) -> None:
        # New transitions receive max current priority so they're sampled
        # at least once before their TD error is known.
        priority = max(self.tree.max_priority, 1.0) ** self.alpha

        idx = self._write_ptr
        self.states[idx]      = state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.next_states[idx] = next_state
        self.dones[idx]       = float(done)
        self.discount_pows[idx] = discount_pow

        if next_valid_mask is not None:
            self.next_valid_masks[idx] = next_valid_mask
        else:
            self.next_valid_masks[idx] = 1.0

        self.tree.add(priority)
        self._write_ptr = (self._write_ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Returns a batch plus tree indices (for priority updates) and IS weights.
        """
        self._beta_step += 1
        n = self._size

        tree_indices  = np.zeros(batch_size, dtype=np.int64)
        data_indices  = np.zeros(batch_size, dtype=np.int64)
        priorities    = np.zeros(batch_size, dtype=np.float64)

        total = self.tree.total
        if total <= 0:
            # Fallback: uniform sampling (shouldn't happen in normal use).
            idxs = np.random.randint(0, n, size=batch_size)
            data_indices = idxs
            tree_indices = idxs + (self.tree.capacity - 1)
            priorities[:] = 1.0 / n
        else:
            segment = total / batch_size
            for i in range(batch_size):
                s = random.uniform(segment * i, segment * (i + 1))
                t_idx, d_idx, priority = self.tree.get(s)
                tree_indices[i] = t_idx
                data_indices[i] = max(0, min(d_idx, n - 1))
                priorities[i]   = max(priority, 1e-8)

        # IS weights: w_i = (1/N * 1/P(i))^beta, normalised by max weight.
        probs      = priorities / max(total, 1e-8)
        min_prob   = np.min(probs)
        max_weight = float((min_prob * n + 1e-8) ** (-self.beta))
        weights    = ((probs * n) ** (-self.beta) / max_weight).astype(np.float32)

        return (
            self.states[data_indices],
            self.actions[data_indices],
            self.rewards[data_indices],
            self.next_states[data_indices],
            self.dones[data_indices],
            self.next_valid_masks[data_indices],
            self.discount_pows[data_indices],
            tree_indices,
            weights,
        )

    def update_priorities(
        self,
        tree_indices: np.ndarray,
        td_errors:    np.ndarray,
        epsilon:      float = 1e-6,
    ) -> None:
        """Recompute priorities from |TD error| after each training step."""
        priorities = (np.abs(td_errors) + epsilon) ** self.alpha
        for idx, priority in zip(tree_indices, priorities):
            self.tree.update(int(idx), float(priority))

    def __len__(self) -> int:
        return self._size


# =============================================================================
# Dueling Q-Network
# =============================================================================
class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture (Wang et al., 2016).

    Architecture:
        Shared:    input → FC(256) → ReLU → FC(256) → ReLU
        Value:     → FC(128) → ReLU → FC(1)
        Advantage: → FC(128) → ReLU → FC(n_actions)
        Q(s,a)   = V(s) + (A(s,a) - mean_a A(s,a))

    Benefits over plain DQN for Pac-Man:
      - Value stream learns how good each state is independently of action.
      - Advantage stream identifies which actions are relatively better.
      - Reduces overestimation in corridor states where all actions are
        similarly valued.
    """

    def __init__(self, input_dim: int = OBS_DIM, output_dim: int = 4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared    = self.shared(x)
        value     = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        # Subtract mean advantage to ensure identifiability.
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


# =============================================================================
# DQN Agent
# =============================================================================
class DQNAgent:
    def __init__(
        self,
        input_dim:      int   = OBS_DIM,
        output_dim:     int   = 4,
        lr:             float = 3e-4,
        gamma:          float = 0.997,
        epsilon_start:  float = 1.0,
        epsilon_end:    float = 0.15,
        epsilon_decay:  int   = 600_000,
        use_amp:        bool | None = None,
    ):
        self.action_dim = output_dim
        self.gamma      = gamma

        # ── Epsilon schedule ──────────────────────────────────────────────────
        self.epsilon        = epsilon_start
        self.epsilon_start  = epsilon_start
        self.epsilon_end    = epsilon_end
        self.epsilon_decay  = epsilon_decay

        self.step_count            = 0
        self.epsilon_jolt_value    = epsilon_end
        self.epsilon_jolt_until_step = 0

        # ── Device ────────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_enabled = (self.device.type == "cuda") if use_amp is None else bool(use_amp and self.device.type == "cuda")
        self.grad_scaler = GradScaler(enabled=self.amp_enabled)

        # ── Networks (Dueling DQN) ────────────────────────────────────────────
        self.policy_net = DuelingQNetwork(input_dim, output_dim).to(self.device)
        self.target_net = DuelingQNetwork(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # ── Optimizer + LR scheduler ──────────────────────────────────────────
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # Linear decay: lr goes from lr → lr/3 over 1M optimizer steps,
        # then holds at lr/3 indefinitely. Allows aggressive early learning
        # and a stable fine-tuning phase.
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: max(1.0 / 3.0, 1.0 - step * (2.0 / 3.0) / 1_000_000),
        )

        # ── Prioritized replay buffer ─────────────────────────────────────────
        self.memory = PrioritizedReplayBuffer(
            capacity=200_000,
            obs_dim=input_dim,
            action_dim=output_dim,
        )

        # ── Reward normalizer ─────────────────────────────────────────────────
        # Rewards are normalized at train time (inside optimize_model).
        # Raw rewards remain in the buffer so the normalizer can adapt
        # as the reward distribution shifts across curriculum stages.
        self.reward_normalizer = RunningMeanStd()

        # ── Training hyper-parameters ─────────────────────────────────────────
        self.batch_size   = 256
        self.train_freq   = 4       # Optimize every N environment steps.
        self.warmup_steps = 10_000  # Don't train until buffer has this many entries.
        self.tau          = 0.003   # Soft target-update rate.

    # =========================================================================
    # Action selection
    # =========================================================================
    def select_action(
        self,
        state,
        valid_actions = None,
        return_exploration: bool = False,
    ):
        self.step_count += 1

        # Exponential epsilon decay with optional curriculum jolt floor.
        scheduled_epsilon = max(
            self.epsilon_end,
            self.epsilon_end
            + (self.epsilon_start - self.epsilon_end)
            * np.exp(-self.step_count / self.epsilon_decay),
        )

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
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_vals  = self.policy_net(state_t).squeeze(0)

            masked_q = torch.full_like(q_vals, float("-inf"))
            for a in candidate_actions:
                masked_q[a] = q_vals[a]

            action = int(masked_q.argmax().item())
            return (action, False) if return_exploration else action

    def apply_exploration_jolt(self, min_epsilon: float = 0.2, duration_steps: int = 50_000) -> None:
        """Temporarily enforce a minimum epsilon after curriculum transitions."""
        self.epsilon_jolt_value      = max(self.epsilon_end, float(min_epsilon))
        self.epsilon_jolt_until_step = max(
            self.epsilon_jolt_until_step,
            self.step_count + int(duration_steps),
        )

    # =========================================================================
    # Training step
    # =========================================================================
    def optimize_model(self, batch_size: int | None = None) -> float | None:
        if len(self.memory) < self.warmup_steps:
            return None

        if self.step_count % self.train_freq != 0:
            return None

        bs = int(batch_size) if batch_size is not None else self.batch_size
        (
            states, actions, rewards, next_states, dones,
            next_valid_masks, discount_pows,
            tree_indices, is_weights,
        ) = self.memory.sample(bs)

        # ── Reward normalization ──────────────────────────────────────────────
        # Update running statistics and normalize this batch.
        # Raw rewards stay in the buffer; normalization is applied at train time
        # so the distribution adapts as curriculum difficulty increases.
        self.reward_normalizer.update(rewards)
        rewards_norm = self.reward_normalizer.normalize(rewards)

        # ── Tensors ───────────────────────────────────────────────────────────
        state_t      = torch.tensor(states,           dtype=torch.float32).to(self.device)
        action_t     = torch.tensor(actions,          dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_t     = torch.tensor(rewards_norm,     dtype=torch.float32).to(self.device)
        next_state_t = torch.tensor(next_states,      dtype=torch.float32).to(self.device)
        done_t       = torch.tensor(dones,            dtype=torch.float32).to(self.device)
        mask_t       = torch.tensor(next_valid_masks, dtype=torch.float32).to(self.device)
        disc_t       = torch.tensor(discount_pows,    dtype=torch.float32).to(self.device)
        is_w_t       = torch.tensor(is_weights,       dtype=torch.float32).to(self.device)

        with autocast(enabled=self.amp_enabled):
            # ── Current Q-values ──────────────────────────────────────────────
            current_q = self.policy_net(state_t).gather(1, action_t).squeeze(1)

            # ── Double DQN target ─────────────────────────────────────────────
            with torch.no_grad():
                policy_next_q      = self.policy_net(next_state_t)
                target_next_q_all  = self.target_net(next_state_t)

                valid_bool  = mask_t > 0.5
                all_invalid = ~valid_bool.any(dim=1)

                # Policy net selects action, target net evaluates it.
                masked_policy = policy_next_q.masked_fill(~valid_bool, float("-inf"))
                next_actions  = masked_policy.argmax(1, keepdim=True)

                masked_target = target_next_q_all.masked_fill(~valid_bool, float("-inf"))
                next_q = masked_target.gather(1, next_actions).squeeze(1)
                next_q = torch.where(all_invalid, torch.zeros_like(next_q), next_q)

                target_q = reward_t + disc_t * next_q * (1.0 - done_t)

            # ── IS-weighted Huber loss ────────────────────────────────────────
            elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
            loss = (is_w_t * elementwise_loss).mean()

        # ── Per-sample TD errors for priority update ──────────────────────────
        with torch.no_grad():
            td_errors = (current_q - target_q).detach().float().cpu().numpy()
        self.memory.update_priorities(tree_indices, td_errors)

        self.optimizer.zero_grad(set_to_none=True)
        if self.amp_enabled:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

        self.scheduler.step()

        return float(loss.item())

    # =========================================================================
    # Soft target-network update
    # =========================================================================
    def update_target_network(self) -> None:
        for target_param, param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
