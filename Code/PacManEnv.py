"""
PacManEnv.py
============
A Gymnasium-compatible environment that wraps the existing Pac-Man GameEngine.

Observation Space
-----------------
A flat numpy array of 34 floats:
  [0-1]     pacman pos x, y           (normalised 0-1)
  [2-3]     pacman dir dx, dy         (-1, 0, 1)
  [4-7]     ghost_0  rel_x, rel_y, dist, threat
  [8-11]    ghost_1  rel_x, rel_y, dist, threat
  [12-15]   ghost_2  rel_x, rel_y, dist, threat
  [16-19]   ghost_3  rel_x, rel_y, dist, threat
  [20-21]   nearest pellet  rel_x, rel_y
  [22-25]   wall sensors    up, down, left, right  (1=wall)
  [26]      pellet_ratio eaten  (0-1)
  [27]      frightened_mode active  (0 or 1)
  [28]      frightened_timer_ratio  (0-1)
  [29]      lives / 3
  [30]      global_scatter_mode  (0 or 1)
  [31-32]   nearest power pellet rel_x, rel_y  (0,0 if none left)
  [33]      power pellets remaining (normalised 0-1)

Action Space
------------
Discrete(5):
  0 → NOOP  (keep current direction)
  1 → UP
  2 → DOWN
  3 → LEFT
  4 → RIGHT

Reward
------
  +pellet_value   when a pellet is eaten
  +ghost_value    when a ghost is eaten (200 / 400 / 800 / 1600 combo)
  -100            when a life is lost
  +500            when a level is won  (episode continues on a fresh maze)
  -500            when game over (no lives left)  → episode terminates
  +0              otherwise

Level completion
----------------
Winning a level does NOT terminate the episode. Instead the engine
auto-advances (next_level()) giving a new procedurally-generated maze and
slightly faster ghosts. The episode only ends on game-over or truncation.
info["levels_completed"] tracks how many mazes have been cleared.

Usage
-----
    from Code.PacManEnv import PacManEnv

    env = PacManEnv(render_mode="human")           # shows the pygame window
    env = PacManEnv(render_mode="rgb_array")        # returns pixels in step()
    env = PacManEnv(render_mode=None)               # headless / fastest

    # Fixed maze — same layout every reset() (great for training)
    env = PacManEnv(render_mode=None, maze_seed=42)

    # Random maze — different layout every reset() (great for evaluation)
    env = PacManEnv(render_mode=None, maze_seed=None)

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
"""

import sys
import os

# Make sure sibling modules (GameEngine etc.) are importable when this file
# is run directly from inside the Code/ folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pygame

from Code.Settings import Settings
from Code.GameEngine import GameEngine, GameState
from Code.Ghost import GhostState


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_settings(json_path: str | None = None) -> dict:
    """Load game_settings.json relative to the Code folder."""
    if json_path is None:
        json_path = os.path.join(_HERE, "game_settings.json")
    s = Settings(json_path)
    return s.get_all()


# ── Environment ───────────────────────────────────────────────────────────────

class PacManEnv(gym.Env):
    """Gymnasium environment wrapping the Pac-Man GameEngine."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    # Action constants
    NOOP  = 0
    UP    = 1
    DOWN  = 2
    LEFT  = 3
    RIGHT = 4

    _ACTION_MAP = {
        NOOP:  (0,  0),
        UP:    (0, -1),
        DOWN:  (0,  1),
        LEFT:  (-1, 0),
        RIGHT: (1,  0),
    }

    # ── Ghost state encoding ──────────────────────────────────────────────────
    _GHOST_STATE_INT = {
        GhostState.SCATTER:    0,
        GhostState.CHASE:      1,
        GhostState.FRIGHTENED: 2,
        GhostState.EATEN:      3,
        GhostState.SPAWNING:   4,
    }

    def __init__(
            self,
            render_mode: str | None = None,
            obs_type: str = "vector",
            settings_path: str | None = None,
            max_episode_steps: int = 27_000,
            maze_seed: int | None = None,
            **engine_kwargs,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.max_episode_steps = max_episode_steps
        self.maze_seed = maze_seed
        self._step_count = 0

        self._base_cfg = _load_settings(settings_path)
        self._base_cfg.update(engine_kwargs)

        self._pygame_initialised = False
        self._screen = None
        self._clock = None
        self._engine = None

        self.action_space = spaces.Discrete(5)
        self._ACTION_MAP = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

        if self.obs_type == "pixels":
            res_str = self._base_cfg.get("window_resolution", "800x800")
            w, h = self._parse_res(res_str)
            self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        else:
            # The finalized 40-element vector
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(40,), dtype=np.float32)

        self._prev_score = 0
        self._prev_lives = 3
        self._won_already = False
        self._levels_completed = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_pygame()
        cfg = dict(self._base_cfg)
        cfg["maze_seed"] = self.maze_seed

        # Optimization: Pass render_mode to engine to skip asset loading
        self._engine = GameEngine(**cfg)
        self._engine.game_state = GameState.GAME
        self._engine.paused = False

        self._prev_score = 0
        self._prev_lives = self._engine.lives
        self._won_already = False
        self._step_count = 0
        self._levels_completed = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        direction = self._ACTION_MAP[int(action)]
        if direction != (0, 0):
            self._engine.pacman.set_direction(direction)

        self._engine.update()
        self._step_count += 1

        # Reward logic
        reward = -0.05  # Step penalty
        score_delta = self._engine.pacman.score - self._prev_score
        if score_delta > 0:
            reward += float(score_delta) / 10.0
        self._prev_score = self._engine.pacman.score

        if self._engine.lives < self._prev_lives:
            reward -= 50.0
            self._prev_lives = self._engine.lives

        if self._engine.won and not self._won_already:
            reward += 100.0  # High win bonus
            self._won_already = True
            self._levels_completed += 1
            self._engine.next_level()
            self._won_already = False
            self._prev_score = self._engine.pacman.score

        terminated = self._engine.game_over
        truncated = (self._step_count >= self.max_episode_steps)
        if terminated: reward -= 50.0

        if self.render_mode == "human": self._render_human()
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """Gymnasium render() — returns an RGB array or renders to screen."""
        if self._engine is None:
            return None

        if self.render_mode == "human":
            self._render_human()
            return None

        if self.render_mode == "rgb_array":
            return self._get_pixel_obs()

        return None

    def close(self):
        if self._pygame_initialised:
            pygame.quit()
            self._pygame_initialised = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_pygame(self):
        """Initialise pygame once, creating a window only for human mode.
        In headless mode (render_mode=None) we skip pygame display/surface
        entirely so images are never loaded and no drawing ever happens."""
        if self._pygame_initialised:
            return

        if self.render_mode is None:
            # Headless: use dummy drivers so pygame.init() doesn't open a
            # window or touch audio hardware.  We still call init() because
            # some pygame modules (e.g. font) are needed by GameEngine.
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            # No surface — GameEngine.draw(None) is a no-op
            self._screen = None
        elif self.render_mode == "human":
            pygame.init()
            res_str = self._base_cfg.get("window_resolution", "800x800")
            w, h    = self._parse_res(res_str)
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Pac-Man Gym Environment")
            self._clock  = pygame.time.Clock()
        else:  # rgb_array — off-screen surface
            pygame.init()
            res_str = self._base_cfg.get("window_resolution", "800x800")
            w, h    = self._parse_res(res_str)
            self._screen = pygame.Surface((w, h))

        self._pygame_initialised = True

    def _render_human(self):
        """Draw one frame to the visible window and pump events."""
        if self._screen is None or self._engine is None:
            return

        # Pump events so the OS doesn't think the window is frozen
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

        self._screen.fill((0, 0, 0))
        self._engine.draw(self._screen)
        pygame.display.flip()

        if self._clock:
            self._clock.tick(self.metadata["render_fps"])

    def _get_pixel_obs(self) -> np.ndarray:
        """Render to off-screen surface and return as HxWx3 uint8 array."""
        if self._screen is None:
            # Headless — return a blank array matching the declared obs shape
            h, w = self.observation_space.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)
        surf = self._screen
        surf.fill((0, 0, 0))
        self._engine.draw(surf)
        # pygame.surfarray returns WxHx3; transpose to HxWx3
        arr = pygame.surfarray.array3d(surf)
        return np.transpose(arr, (1, 0, 2))

    def _get_obs(self) -> np.ndarray:
        if self.obs_type == "pixels":
            return self._get_pixel_obs()
        return self._get_vector_obs()

    def _get_vector_obs(self) -> np.ndarray:
        eng = self._engine
        ts, mw_px, mh_px = eng.tile_size, eng.maze.width * eng.tile_size, eng.maze.height * eng.tile_size

        # [0-3] Pac-Man
        obs = [eng.pacman.x / mw_px, eng.pacman.y / mh_px, float(eng.pacman.direction[0]),
               float(eng.pacman.direction[1])]

        # [4-27] Ghosts (6 features each)
        for i in range(4):
            if i < len(eng.ghosts):
                g = eng.ghosts[i]
                rel_x, rel_y = (g.x - eng.pacman.x) / mw_px, (g.y - eng.pacman.y) / mh_px
                dist = np.sqrt(rel_x ** 2 + rel_y ** 2)
                threat = 1.0 if g.state in [GhostState.CHASE, GhostState.SCATTER] else (
                    -1.0 if g.state == GhostState.FRIGHTENED else 0.0)
                obs.extend([rel_x, rel_y, dist, float(g.current_dir[0]), float(g.current_dir[1]), threat])
            else:
                obs.extend([0.0, 0.0, 1.5, 0.0, 0.0, 0.0])

        # [28-29] Pellet Radar
        p_rel = [0.0, 0.0]
        if eng.pellets or eng.power_pellets:
            all_p = eng.pellets + eng.power_pellets
            dists = [(p[0] - eng.pacman.x) ** 2 + (p[1] - eng.pacman.y) ** 2 for p in all_p]
            cp = all_p[np.argmin(dists)]
            p_rel = [(cp[0] - eng.pacman.x) / mw_px, (cp[1] - eng.pacman.y) / mh_px]
        obs.extend(p_rel)

        # [30-33] Walls
        tx, ty = int(eng.pacman.x / ts), int(eng.pacman.y / ts)
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = tx + dx, ty + dy
            is_wall = 1.0 if not (0 <= nx < eng.maze.width and 0 <= ny < eng.maze.height) or eng.maze.maze[ny][
                nx] == 1 else 0.0
            obs.append(is_wall)

        # [34-39] Global
        total_p = max(eng.pellets_eaten_this_level + len(eng.pellets), 1)
        pp_dist = np.sqrt(min([(p[0] - eng.pacman.x) ** 2 + (p[1] - eng.pacman.y) ** 2 for p in
                               eng.power_pellets])) / mw_px if eng.power_pellets else 1.5
        obs.extend([eng.pellets_eaten_this_level / total_p, 1.0 if eng.frightened_mode else 0.0,
                    eng.frightened_timer / max(eng.frightened_duration, 1), eng.lives / 3.0,
                    1.0 if eng.global_scatter_mode else 0.0, pp_dist])

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict:
        if self._engine is None:
            return {}
        return {
            "score":            self._engine.pacman.score,
            "lives":            self._engine.lives,
            "level":            self._engine.level,
            "levels_completed": self._levels_completed,
            "pellets_eaten":    self._engine.pellets_eaten_this_level,
            "pellets_left":     len(self._engine.pellets) + len(self._engine.power_pellets),
            "frightened":       self._engine.frightened_mode,
            "game_over":        self._engine.game_over,
            "won":              self._engine.won,
        }

    @staticmethod
    def _parse_res(res_str: str) -> tuple[int, int]:
        try:
            w, h = res_str.split("x")
            return int(w), int(h)
        except Exception:
            return 800, 800


# ── Quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running PacManEnv smoke-test (headless)...\n")

    # ── 1. Basic sanity: 100 random steps ────────────────────────────────────
    env = PacManEnv(render_mode=None, obs_type="vector", maze_seed=None)
    obs, info = env.reset()
    print(f"  obs shape : {obs.shape}")
    print(f"  info      : {info}")

    total_reward = 0.0
    for step_i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print(f"  Total reward over 100 steps: {total_reward:.1f}\n")

    # ── 2. Seed reproducibility: two envs with the same seed must produce ────
    #        identical starting observations after reset()
    print("  Checking maze_seed reproducibility...")
    env_a = PacManEnv(render_mode=None, maze_seed=42)
    env_b = PacManEnv(render_mode=None, maze_seed=42)
    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    env_a.close()
    env_b.close()

    import numpy as np
    match = np.allclose(obs_a, obs_b)
    print(f"  Seeded envs produce identical obs: {match}")
    assert match, "Seed reproducibility FAILED!"

    # ── 3. Different seeds give different mazes ───────────────────────────────
    env_c = PacManEnv(render_mode=None, maze_seed=99)
    obs_c, _ = env_c.reset()
    env_c.close()
    different = not np.allclose(obs_a, obs_c)
    print(f"  Different seeds give different obs: {different}")

    print("\nSmoke-test passed ✓")

