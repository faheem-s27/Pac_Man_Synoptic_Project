import numpy as np
import gymnasium as gym


class DQNActionMaskingWrapper(gym.Wrapper):
    """
    DQN-focused action masking helper.

    - Exposes valid/invalid actions and a binary mask.
    - Supports controlled invalid-action exploration so the agent can still
      learn wall-collision penalties without spamming illegal moves.
    """

    def __init__(self, env: gym.Env, invalid_explore_prob: float = 0.08):
        super().__init__(env)
        self.invalid_explore_prob = float(max(0.0, min(1.0, invalid_explore_prob)))

    def get_valid_actions(self) -> list[int]:
        valid = []
        if hasattr(self.env, "get_valid_actions"):
            valid = list(self.env.get_valid_actions())
        if not valid:
            n = int(getattr(self.action_space, "n", 0))
            return list(range(n))
        return valid

    def get_invalid_actions(self) -> list[int]:
        n = int(getattr(self.action_space, "n", 0))
        all_actions = list(range(n))
        valid_set = set(self.get_valid_actions())
        return [a for a in all_actions if a not in valid_set]

    def get_action_mask(self) -> np.ndarray:
        n = int(getattr(self.action_space, "n", 0))
        mask = np.zeros(n, dtype=np.float32)
        for a in self.get_valid_actions():
            if 0 <= int(a) < n:
                mask[int(a)] = 1.0
        return mask

    def pick_action(self, policy_action: int, exploring: bool) -> int:
        """
        Apply collision-aware exploration policy.

        - Exploitation path keeps policy_action (already masked by agent).
        - Exploration path occasionally samples invalid actions so collision
          penalties are learned, but mostly samples valid actions.
        """
        if not exploring:
            return int(policy_action)

        valid = self.get_valid_actions()
        invalid = self.get_invalid_actions()

        if invalid and np.random.random() < self.invalid_explore_prob:
            return int(np.random.choice(invalid))

        if valid:
            return int(np.random.choice(valid))

        return int(policy_action)

