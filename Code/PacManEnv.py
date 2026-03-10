"""
PacManEnv.py
============
A Gymnasium-compatible environment that wraps the existing Pac-Man GameEngine.

Observation Space
-----------------
A flat numpy array of 40 floats (normalised between -1.0 and 1.0):
  [0-3]     Pac-Man State: pos_x, pos_y, dir_dx, dir_dy
  [4-27]    Ghosts (x4): rel_x, rel_y, dist, dir_dx, dir_dy, threat_level
  [28-29]   Pellet Radar: nearest_pellet_rel_x, nearest_pellet_rel_y
  [30-33]   Wall Sensors: up, down, left, right (1.0 if wall, 0.0 otherwise)
  [34-39]   Global State: pellet_ratio, frightened_active, frightened_timer, lives_ratio, scatter_mode, pp_dist

Action Space
------------
Discrete(5):
  0 → NOOP  (keep current direction)
  1 → UP
  2 → DOWN
  3 → LEFT
  4 → RIGHT

Reward Architecture (Scaled for Network Stability)
--------------------------------------------------
  + (score_delta / 2.0)  : Reward for eating pellets, power pellets, and ghosts.
  - 0.05                 : Step penalty to encourage movement efficiency.
  - 50.0                 : Life lost penalty.
  - 50.0                 : Game over penalty (terminal state).
  + 100.0                : Level cleared bonus.

Level Completion
----------------
Winning a level does NOT terminate the episode. Instead, the engine auto-advances
(next_level()), generating a new procedurally-generated maze to enforce generalisation.
The episode only ends on game-over or truncation.
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

def _load_settings(json_path: str | dict | None = None) -> dict:
    """Load game_settings.json relative to the Code folder, or return the dict provided."""
    # If a dictionary is passed directly (e.g. from CurriculumManager), use it.
    if isinstance(json_path, dict):
        return json_path

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
    # NOOP  = 0
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3

    _ACTION_MAP = {
        # NOOP:  (0,  0),
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
            settings: dict | None = None,
            settings_path: str | None = None, # Legacy: kept for compatibility
            max_episode_steps: int = 27_000,
            maze_seed: int | None = None,
            maze_algorithm: str = "recursive_backtracking",
            **engine_kwargs,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.max_episode_steps = max_episode_steps
        self.maze_seed = maze_seed
        self._step_count = 0

        # Load settings: try 'settings' dict first, then 'settings_path', then default file
        self._base_cfg = _load_settings(settings if settings else settings_path)
        self._base_cfg.update(engine_kwargs)

        self._pygame_initialised = False
        self._screen = None
        self._clock = None
        self._engine = None

        self.action_space = spaces.Discrete(4)

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

        # Ensure pygame initialized if needed
        self._ensure_pygame()

        # Merge base config
        cfg = self._base_cfg.copy()

        # Apply runtime seed override if provided
        if seed is not None:
            cfg["maze_seed"] = seed

        # Initialize Engine with merged config
        # GameEngine receives active_ghost_count via **kwargs if present in cfg
        self._engine = GameEngine(**cfg)

        self._engine.game_state = GameState.GAME
        self._engine.paused = False

        self._prev_score = 0
        self._prev_lives = self._engine.lives
        self._won_already = False
        self._step_count = 0
        self._levels_completed = 0

        # Return first observation
        return self._get_obs(), {}

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
            reward += float(score_delta) / 2.0
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
        """Initialise pygame once, creating a window only for human mode."""
        if self._pygame_initialised:
            return

        if self.render_mode is None:
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            self._screen = None
        elif self.render_mode == "human":
            pygame.init()
            res_str = self._base_cfg.get("window_resolution", "800x800")
            w, h    = self._parse_res(res_str)
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Pac-Man Gym Environment")
            self._clock  = pygame.time.Clock()
        else:
            pygame.init()
            res_str = self._base_cfg.get("window_resolution", "800x800")
            w, h    = self._parse_res(res_str)
            self._screen = pygame.Surface((w, h))

        self._pygame_initialised = True

    def _draw_debug_sensors(self):
        """Visualizes the AI's 40-element input vector strictly using center-mass math."""
        if self._screen is None or self._engine is None:
            return

        eng = self._engine
        ts = eng.tile_size

        # ACTION: Strict Center Mass Anchor (Top-left origin is banned)
        pac_cx = int(eng.pacman.x + (ts / 2))
        pac_cy = int(eng.pacman.y + (ts / 2))
        pac_center = (pac_cx, pac_cy)

        # 1. Pellet Radar (Green)
        if eng.pellets or eng.power_pellets:
            all_p = eng.pellets + eng.power_pellets
            # PELLETS are already PRE-CALCULATED as center-mass pixel coordinates in GameEngine
            px_coords = all_p
            dists = [(px[0] - pac_cx) ** 2 + (px[1] - pac_cy) ** 2 for px in px_coords]

            target_px = px_coords[np.argmin(dists)]
            # Draw GREEN line using purely centered, scaled pixels
            pygame.draw.line(self._screen, (0, 255, 0), pac_center, target_px, 2)

        # 2. Ghost Tracking
        for g in eng.ghosts:
            color = (255, 0, 0) if g.state in [GhostState.CHASE, GhostState.SCATTER] else (0, 255, 255)
            g_center = (int(g.x + (ts / 2)), int(g.y + (ts / 2)))
            pygame.draw.line(self._screen, color, pac_center, g_center, 1)

        # 3. Wall Sensors
        for dx, dy in [(0, -ts), (0, ts), (-ts, 0), (ts, 0)]:
            sensor_px = (pac_cx + dx, pac_cy + dy)
            tx, ty = int(sensor_px[0] / ts), int(sensor_px[1] / ts)
            if not (0 <= tx < eng.maze.width and 0 <= ty < eng.maze.height) or eng.maze.maze[ty][tx] == 1:
                pygame.draw.circle(self._screen, (255, 255, 0), sensor_px, 5, 1)

    def _render_human(self):
        """Draw one frame to the visible window and pump events."""
        if self._screen is None or self._engine is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

        self._screen.fill((0, 0, 0))
        self._engine.draw(self._screen)

        # Active overlay
        self._draw_debug_sensors()
        pygame.display.flip()

        if self._clock:
            self._clock.tick(self.metadata["render_fps"])

    def _get_pixel_obs(self) -> np.ndarray:
        if self._screen is None:
            h, w = self.observation_space.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)
        surf = self._screen
        surf.fill((0, 0, 0))
        self._engine.draw(surf)
        arr = pygame.surfarray.array3d(surf)
        return np.transpose(arr, (1, 0, 2))

    def _get_obs(self) -> np.ndarray:
        if self.obs_type == "pixels":
            return self._get_pixel_obs()
        return self._get_vector_obs()

    def _get_vector_obs(self) -> np.ndarray:
        eng = self._engine
        ts = eng.tile_size
        mw_px, mh_px = eng.maze.width * ts, eng.maze.height * ts

        # ACTION: Establish strict center-mass for neural network inputs
        pac_cx = eng.pacman.x + (ts / 2.0)
        pac_cy = eng.pacman.y + (ts / 2.0)

        # [0-3] Pac-Man
        obs = [pac_cx / mw_px, pac_cy / mh_px, float(eng.pacman.direction[0]), float(eng.pacman.direction[1])]

        # [4-27] Ghosts
        for i in range(4):
            if i < len(eng.ghosts):
                g = eng.ghosts[i]
                g_cx = g.x + (ts / 2.0)
                g_cy = g.y + (ts / 2.0)

                rel_x = (g_cx - pac_cx) / mw_px
                rel_y = (g_cy - pac_cy) / mh_px
                dist = np.sqrt(rel_x ** 2 + rel_y ** 2)

                threat = 1.0 if g.state in [GhostState.CHASE, GhostState.SCATTER] else (
                    -1.0 if g.state == GhostState.FRIGHTENED else 0.0)
                obs.extend([rel_x, rel_y, dist, float(g.current_dir[0]), float(g.current_dir[1]), threat])
            else:
                obs.extend([0.0, 0.0, 1.5, 0.0, 0.0, 0.0])

        # [28-29] Pellet Radar (Center Mass Distance & Grid-to-Pixel Conversion)
        p_rel = [0.0, 0.0]
        if eng.pellets or eng.power_pellets:
            all_p = eng.pellets + eng.power_pellets
            # PELLETS are already PRE-CALCULATED as center-mass pixel coordinates in GameEngine
            px_coords = all_p
            dists = [(px[0] - pac_cx) ** 2 + (px[1] - pac_cy) ** 2 for px in px_coords]

            target_px = px_coords[int(np.argmin(dists))]
            p_rel = [(target_px[0] - pac_cx) / mw_px, (target_px[1] - pac_cy) / mh_px]
        obs.extend(p_rel)

        # [30-33] Walls (Center Mass Boundary Checking)
        tx, ty = int(pac_cx / ts), int(pac_cy / ts)
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = tx + dx, ty + dy
            is_wall = 1.0 if not (0 <= nx < eng.maze.width and 0 <= ny < eng.maze.height) or eng.maze.maze[ny][
                nx] == 1 else 0.0
            obs.append(is_wall)

        # [34-39] Global (Power Pellet Center Mass & Grid-to-Pixel Conversion)
        total_p = max(eng.pellets_eaten_this_level + len(eng.pellets), 1)
        pp_dist = 1.5
        if eng.power_pellets:
            # PELLETS are already PRE-CALCULATED as center-mass pixel coordinates in GameEngine
            pp_coords = eng.power_pellets
            pp_dists = [(px[0] - pac_cx) ** 2 + (px[1] - pac_cy) ** 2 for px in pp_coords]
            pp_dist = np.sqrt(min(pp_dists)) / mw_px

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

    env = PacManEnv(render_mode=None, obs_type="vector", maze_seed=None)
    obs, info = env.reset()
    print(f"  obs shape : {obs.shape}")

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