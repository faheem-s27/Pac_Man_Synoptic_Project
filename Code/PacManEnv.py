"""
PacManEnv.py
============
A Gymnasium-compatible environment that wraps the existing Pac-Man GameEngine.

Observation Space
-----------------
A flat numpy array containing:
  [0]       pacman_tile_x   (int, normalised 0-1)
  [1]       pacman_tile_y   (int, normalised 0-1)
  [2]       pacman_dir_x    (-1, 0, 1)
  [3]       pacman_dir_y    (-1, 0, 1)
  [4..7]    ghost_0 tile_x, tile_y, state, is_frightened  (normalised)
  [8..11]   ghost_1 ...
  [12..15]  ghost_2 ...
  [16..19]  ghost_3 ...
  [20]      lives (normalised 0-1 over max_lives=3)
  [21]      pellets_eaten_ratio  (0-1)
  [22]      frightened_mode_active (0 or 1)
  [23]      frightened_timer_ratio (0-1)
  [24]      global_scatter_mode (0 or 1)

Additionally the environment can optionally provide a pixel render of the game
as the observation by setting `obs_type="pixels"` in the constructor.

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
        obs_type: str = "vector",           # "vector" or "pixels"
        settings_path: str | None = None,
        max_episode_steps: int = 27_000,    # 7.5 minutes at 60 FPS
        maze_seed: int | None = None,       # fixed int → same maze every reset(); None → random
        **engine_kwargs,                    # override any GameEngine param
    ):
        super().__init__()

        assert render_mode in (None, "human", "rgb_array"), \
            f"Unsupported render_mode: {render_mode!r}"
        assert obs_type in ("vector", "pixels"), \
            f"Unsupported obs_type: {obs_type!r}"

        self.render_mode       = render_mode
        self.obs_type          = obs_type
        self.max_episode_steps = max_episode_steps
        self.maze_seed         = maze_seed   # None = random each reset
        self._step_count       = 0

        # Load base config from JSON then override with any kwargs
        self._base_cfg = _load_settings(settings_path)
        self._base_cfg.update(engine_kwargs)

        # We need pygame running even in headless mode (for GameEngine drawing),
        # but we only open a visible window in "human" mode.
        self._pygame_initialised = False
        self._screen = None
        self._clock  = None

        # Will be created in reset()
        self._engine: GameEngine | None = None

        # ── Observation & action spaces ───────────────────────────────────────
        self.action_space = spaces.Discrete(5)

        if self.obs_type == "pixels":
            # Parse resolution from settings
            res_str = self._base_cfg.get("window_resolution", "800x800")
            w, h    = self._parse_res(res_str)
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(h, w, 3), dtype=np.uint8
            )
        else:
            # Vector obs: 35 floats in [0, 1] (or small integers)
            # See module docstring for layout
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(31,), dtype=np.float32
            )

        # Reward tracking (to compute step reward)
        self._prev_score       = 0
        self._prev_lives       = 3
        self._won_already      = False
        self._levels_completed = 0  # How many mazes cleared this episode

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._ensure_pygame()

        # Build a fresh GameEngine each episode
        cfg = dict(self._base_cfg)

        # Inject maze seed — overrides anything in the JSON
        # If maze_seed is set it pins the starting maze; None lets MazeGenerator
        # pick a random layout on each reset.
        cfg["maze_seed"] = self.maze_seed

        # Force headless-friendly settings when not rendering
        if self.render_mode is None:
            cfg.setdefault("enable_ghosts", True)

        self._engine = GameEngine(**cfg)
        # Jump straight to GAME state (skip audio intro)
        self._engine.game_state = GameState.GAME
        self._engine.paused     = False

        self._prev_score       = 0
        self._prev_lives       = self._engine.lives
        self._won_already      = False
        self._step_count       = 0
        self._levels_completed = 0

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self._engine is not None, "Call reset() before step()."

        # Apply action
        direction = self._ACTION_MAP[int(action)]
        if direction != (0, 0):
            self._engine.pacman.set_direction(direction)

        # Tick the engine one frame
        self._engine.update()
        self._step_count += 1

        # ── Reward ───────────────────────────────────────────────────────────
        reward = -0.1

        # Score delta (pellets, ghosts eaten)
        score_delta = self._engine.pacman.score - self._prev_score
        reward += float(score_delta)
        self._prev_score = self._engine.pacman.score

        # Life lost penalty
        if self._engine.lives < self._prev_lives:
            reward -= 500.0
            self._prev_lives = self._engine.lives

        # Win bonus — advance to a new maze rather than ending the episode
        if self._engine.won and not self._won_already:
            reward += 500.0
            self._won_already = True
            self._levels_completed += 1
            print(f"[Env] Level cleared! Advancing to maze {self._levels_completed + 1} "
                  f"(engine level {self._engine.level + 1})")
            self._engine.next_level()
            # next_level() resets won → False, so the flag must be cleared too
            self._won_already = False
            # Keep score accumulation continuous across levels
            self._prev_score = self._engine.pacman.score

        # ── Termination / Truncation ──────────────────────────────────────────
        # Only game-over (no lives left) terminates the episode.
        # Winning a level keeps it alive on the new maze.
        terminated = self._engine.game_over
        truncated  = (self._step_count >= self.max_episode_steps)

        if self._engine.game_over:
            reward -= 500.0  # Extra penalty for full game over

        obs  = self._get_obs()
        info = self._get_info()

        # Render if in human mode
        if self.render_mode == "human":
            self._render_human()

        return obs, reward, terminated, truncated, info

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
        ts = eng.tile_size
        mw_px = eng.maze.width * ts
        mh_px = eng.maze.height * ts

        # 1. Pac-Man State (4 values)
        pac_x_norm = eng.pacman.x / mw_px
        pac_y_norm = eng.pacman.y / mh_px
        pac_dx, pac_dy = eng.pacman.direction
        obs = [pac_x_norm, pac_y_norm, float(pac_dx), float(pac_dy)]

        # 2. Ghost Proximity Sensors (4 ghosts × 4 values = 16)
        for i in range(4):
            if i < len(eng.ghosts):
                g = eng.ghosts[i]
                # Calculate relative distance (Vector from Pac-Man to Ghost)
                rel_x = (g.x - eng.pacman.x) / mw_px
                rel_y = (g.y - eng.pacman.y) / mh_px

                # Binary state: 1.0 if Dangerous, -1.0 if Edible (Frightened), 0.0 if Eaten/Spawning
                g_threat = 0.0
                if g.state == GhostState.CHASE or g.state == GhostState.SCATTER:
                    g_threat = 1.0
                elif g.state == GhostState.FRIGHTENED:
                    g_threat = -1.0

                obs.extend([rel_x, rel_y, np.sqrt(rel_x ** 2 + rel_y ** 2), g_threat])
            else:
                obs.extend([0.0, 0.0, 1.0, 0.0])  # Placeholder for absent ghosts

        # 3. Global Game State (5 values)
        pellet_ratio = eng.pellets_eaten_this_level / max((len(eng.pellets) + eng.pellets_eaten_this_level), 1)
        frit_active = 1.0 if eng.frightened_mode else 0.0
        frit_time = eng.frightened_timer / eng.frightened_duration

        # 4. Nearest Pellet "Radar" (2 values)
        if eng.pellets or eng.power_pellets:
            all_pellets = eng.pellets + eng.power_pellets
            # Calculate distances to all pellets to find the closest one
            distances = [np.sqrt((p[0] - eng.pacman.x) ** 2 + (p[1] - eng.pacman.y) ** 2) for p in all_pellets]
            closest_idx = np.argmin(distances)
            closest_p = all_pellets[closest_idx]

            # Directional vector to the closest pellet
            pellet_rel_x = (closest_p[0] - eng.pacman.x) / mw_px
            pellet_rel_y = (closest_p[1] - eng.pacman.y) / mh_px
            obs.extend([pellet_rel_x, pellet_rel_y])
        else:
            obs.extend([0.0, 0.0])  # No pellets left

        # 5. Wall Sensors (4 values)
        # Check if the tile in each direction is a wall (1 = Wall, 0 = Path)
        current_tx = int(eng.pacman.x / ts)
        current_ty = int(eng.pacman.y / ts)
        wall_sensors = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            tx, ty = current_tx + dx, current_ty + dy
            if 0 <= tx < eng.maze.width and 0 <= ty < eng.maze.height:
                wall_sensors.append(1.0 if eng.maze.maze[ty][tx] == 1 else 0.0)
            else:
                wall_sensors.append(1.0)  # Out of bounds is a wall
        obs.extend(wall_sensors)

        obs.extend([pellet_ratio, frit_active, frit_time, float(eng.lives / 3), float(eng.global_scatter_mode)])

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

