"""
PacManEnv.py
============
A Gymnasium-compatible environment that wraps the existing Pac-Man GameEngine.

Observation Space
-----------------
A flat numpy array of 16 floats (normalised between -1.0 and 1.0):
  [0-1]   Pac-Man Direction: dir_dx, dir_dy
  [2-4]   Closest Ghost 1: rel_x, rel_y, threat_level
  [5-7]   Closest Ghost 2: rel_x, rel_y, threat_level
  [8-9]   Nearest Pellet: rel_x, rel_y
  [10-13] Wall Sensors: up, down, left, right (1.0 if wall, 0.0 if empty)
  [14-15] Nearest Power Pellet: rel_x, rel_y

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
import torch

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
from Code.Pathfinding import Pathfinding


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
            max_episode_steps: int = 500,
            maze_seed: int | None = None,
            maze_algorithm: str = "recursive_backtracking",
            **engine_kwargs,
    ):
        super().__init__()
        self._last_action = None
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.maze_seed = maze_seed
        self._step_count = 0

        # Load settings: try 'settings' dict first, then 'settings_path', then default file
        self._base_cfg = _load_settings(settings if settings else settings_path)
        self._base_cfg.update(engine_kwargs)

        self.max_episode_steps = self._base_cfg.get("max_episode_steps", max_episode_steps)
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
            # New compact 16-element vector observation
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)

        self._prev_score = 0
        self._prev_lives = 3
        self._won_already = False
        self._levels_completed = 0

        self._stuck_frames = 0

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

        self._last_action = None
        self._stuck_frames = 0

        # Return first observation
        return self._get_obs(), {}

    def step(self, action: int):
        new_direction = self._ACTION_MAP[int(action)]
        current_direction = self._engine.pacman.direction

        # ACTION: The Intent Buffer
        # If the AI wants to turn, check if the turn is valid.
        if new_direction != current_direction:
            # We use next_direction to queue the turn. The GameEngine will execute
            # it when the grid alignment allows, preventing sub-pixel wall snags.
            self._engine.pacman.next_direction = new_direction

            # If Pac-Man is currently stopped (e.g. at a dead end), try to force the move
            if current_direction == (0, 0):
                self._engine.pacman.set_direction(new_direction)

        # Step the engine
        self._engine.update()
        self._step_count += 1

        # Reward logic
        reward = -0.01  # Step penalty

        # ACTION: Apply the 180-degree Reversal Penalty
        if self._last_action is not None:
            # Reversal pairs: 0-1 (UP-DOWN), 2-3 (LEFT-RIGHT)
            is_reversal = False
            if self._last_action == 0 and action == 1: is_reversal = True  # UP to DOWN
            if self._last_action == 1 and action == 0: is_reversal = True  # DOWN to UP
            if self._last_action == 2 and action == 3: is_reversal = True  # LEFT to RIGHT
            if self._last_action == 3 and action == 2: is_reversal = True  # RIGHT to LEFT

            if is_reversal:
                reward -= 2.0  # Significant tax on vibrating behavior

        # Update the action memory for the next frame
        self._last_action = action

        # If Pac-Man is stationary (hit a wall and didn't provide a valid turn)
        if self._engine.pacman.direction == (0, 0):
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0  # Reset immediately upon moving

        # If stuck for 30 consecutive frames, kill it
        if self._stuck_frames >= 30:
            reward -= 50.0
            self._engine.game_over = True

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
            # ACTION: Infinite loop amputated. We DO NOT call next_level() here.

        # ACTION: Terminate episode immediately if Pac-Man dies OR wins
        terminated = self._engine.game_over or self._engine.won
        truncated = (self._step_count >= self.max_episode_steps)

        # ACTION: Apply death penalty strictly upon death (not winning)
        if self._engine.game_over:
            reward -= 50.0

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
            # Remove headless driver overrides so a real window can be created,
            # even if a previous headless environment set these variables.
            for _var in ("SDL_VIDEODRIVER", "SDL_AUDIODRIVER"):
                if os.environ.get(_var) == "dummy":
                    del os.environ[_var]
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
        """Visualizes the current 16-element observation (Pure Euclidean straight-lines)."""
        if self._screen is None or self._engine is None:
            return

        eng = self._engine
        ts = eng.tile_size

        # Pac-Man center (anchor for all sensors)
        pac_cx = int(eng.pacman.x + (ts / 2))
        pac_cy = int(eng.pacman.y + (ts / 2))
        pac_center = (pac_cx, pac_cy)

        # ----- Ghost sensors (Euclidean) -----
        active_ghosts = []
        for g in eng.ghosts:
            if g.state in (GhostState.EATEN, GhostState.SPAWNING):
                continue
            g_cx = g.x + (ts / 2.0)
            g_cy = g.y + (ts / 2.0)
            dx = g_cx - pac_cx
            dy = g_cy - pac_cy
            dist_sq = dx * dx + dy * dy
            active_ghosts.append((dist_sq, g_cx, g_cy))

        active_ghosts.sort(key=lambda t: t[0])

        ghost_colors = [(255, 0, 0), (255, 128, 0)]  # closest, second-closest
        for idx in range(min(2, len(active_ghosts))):
            _, g_cx, g_cy = active_ghosts[idx]
            pygame.draw.line(self._screen, ghost_colors[idx], pac_center, (int(g_cx), int(g_cy)), 2)

        # ----- Nearest normal pellet (Euclidean, Green) -----
        if eng.pellets:
            dists = []
            for px, py in eng.pellets:
                dx = px - pac_cx
                dy = py - pac_cy
                dists.append((dx * dx + dy * dy, px, py))
            dists.sort(key=lambda t: t[0])
            _, nearest_px, nearest_py = dists[0]
            pygame.draw.line(self._screen, (0, 255, 0), pac_center, (int(nearest_px), int(nearest_py)), 2)

        # ----- Nearest power pellet (Euclidean, Blue) -----
        if eng.power_pellets:
            dists = []
            for px, py in eng.power_pellets:
                dx = px - pac_cx
                dy = py - pac_cy
                dists.append((dx * dx + dy * dy, px, py))
            dists.sort(key=lambda t: t[0])
            _, nearest_px, nearest_py = dists[0]
            pygame.draw.line(self._screen, (0, 128, 255), pac_center, (int(nearest_px), int(nearest_py)), 2)

        # ----- Wall sensors (Strictly local, Yellow circles) -----
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
        """Return the 16-element dense Euclidean observation vector."""
        eng = self._engine
        ts = eng.tile_size
        mw_px, mh_px = eng.maze.width * ts, eng.maze.height * ts

        pac_cx = eng.pacman.x + (ts / 2.0)
        pac_cy = eng.pacman.y + (ts / 2.0)
        dir_dx, dir_dy = eng.pacman.direction

        obs: list[float] = []

        # [0-1] Pac-Man direction
        obs.append(float(dir_dx))
        obs.append(float(dir_dy))

        # 2) K-nearest ghosts (Pure Euclidean + State Threat)
        active_ghosts = []
        for g in eng.ghosts:
            if g.state in (GhostState.EATEN, GhostState.SPAWNING):
                continue
            g_cx = g.x + (ts / 2.0)
            g_cy = g.y + (ts / 2.0)
            dx = g_cx - pac_cx
            dy = g_cy - pac_cy
            dist_sq = dx * dx + dy * dy
            active_ghosts.append((dist_sq, g_cx, g_cy, g.state))

        active_ghosts.sort(key=lambda t: t[0])

        def _encode_ghost(g_cx: float, g_cy: float, state) -> list[float]:
            rel_x = (g_cx - pac_cx) / mw_px
            rel_y = (g_cy - pac_cy) / mh_px
            if state in (GhostState.CHASE, GhostState.SCATTER):
                threat = 1.0
            elif state == GhostState.FRIGHTENED:
                threat = -1.0
            else:
                threat = 0.0
            return [rel_x, rel_y, float(threat)]

        # [2-4] Closest Ghost 1, [5-7] Closest Ghost 2
        for k in range(2):
            if k < len(active_ghosts):
                _, g_cx, g_cy, g_state = active_ghosts[k]
                obs.extend(_encode_ghost(g_cx, g_cy, g_state))
            else:
                obs.extend([0.0, 0.0, 0.0])

        # 3) Nearest normal pellet (Pure Euclidean)
        pellet_rel = [0.0, 0.0]
        if eng.pellets:
            dists = []
            for px, py in eng.pellets:
                dx = px - pac_cx
                dy = py - pac_cy
                dists.append((dx * dx + dy * dy, px, py))
            dists.sort(key=lambda t: t[0])
            _, px, py = dists[0]
            pellet_rel = [(px - pac_cx) / mw_px, (py - pac_cy) / mh_px]
        # [8-9]
        obs.extend(pellet_rel)

        # 4) Wall sensors: up, down, left, right (Strictly Local)
        tx = int(pac_cx / ts)
        ty = int(pac_cy / ts)
        wall_vals: list[float] = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = tx + dx, ty + dy
            if not (0 <= nx < eng.maze.width and 0 <= ny < eng.maze.height):
                wall_vals.append(1.0)
            else:
                wall_vals.append(1.0 if eng.maze.maze[ny][nx] == 1 else 0.0)
        # [10-13]
        obs.extend(wall_vals)

        # 5) Nearest power pellet (Pure Euclidean)
        pp_rel = [0.0, 0.0]
        if eng.power_pellets:
            dists = []
            for px, py in eng.power_pellets:
                dx = px - pac_cx
                dy = py - pac_cy
                dists.append((dx * dx + dy * dy, px, py))
            dists.sort(key=lambda t: t[0])
            _, px, py = dists[0]
            pp_rel = [(px - pac_cx) / mw_px, (py - pac_cy) / mh_px]
        # [14-15]
        obs.extend(pp_rel)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
