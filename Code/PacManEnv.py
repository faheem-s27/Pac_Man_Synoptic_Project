"""
PacManEnv.py
============
Fully observable, dense local-grid version.

Observation (float32):
    - 11x11 absolute grid slice centred on Pac-Man's current tile (flattened to 121 floats).
        Encoding per cell (single channel):
            -1.0 : Wall or out-of-bounds
             0.0 : Empty walkable floor
             0.5 : Regular pellet
             1.0 : Power pellet
            -0.8 : Lethal ghost (CHASE / SCATTER / other non-frightened, non-eaten)
             0.8 : Edible ghost (FRIGHTENED)
        Priority (if multiple entities share a tile):
            Pac-Man > Ghost (lethal/edible) > Power pellet > Regular pellet > Empty.
            Pac-Man uses the underlying floor/maze code (no separate value).
    - 1 scalar: normalized remaining fright time in [0, 1].

Total observation size: 122 floats.

Action Space: Discrete(4) — absolute cardinal directions
    0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
Each env.step() advances the GameEngine by exactly one update tick.
"""

import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.Settings import Settings
from Code.GameEngine import GameEngine, GameState
from Code.Ghost import GhostState

def _load_settings(json_path: str | dict | None = None) -> dict:
    if isinstance(json_path, dict): return json_path
    if json_path is None: json_path = os.path.join(_HERE, "game_settings.json")
    return Settings(json_path).get_all()

class PacManEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    UP      = 0
    DOWN    = 1
    LEFT_C  = 2
    RIGHT_C = 3

    # Absolute cardinal index -> direction vector (in tile coordinates)
    _CARDINAL_TO_VEC = {UP: (0, -1), DOWN: (0, 1), LEFT_C: (-1, 0), RIGHT_C: (1, 0)}
    _CARDINAL_OPPOSITE = {UP: DOWN, DOWN: UP, LEFT_C: RIGHT_C, RIGHT_C: LEFT_C}

    def __init__(self, render_mode: str | None = None, obs_type: str = "vector", settings: dict | None = None, settings_path: str | None = None, max_episode_steps: int = 10000, maze_seed: int | None = None, **engine_kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.maze_seed = maze_seed
        self._step_count = 0

        self._base_cfg = _load_settings(settings if settings else settings_path)
        self._base_cfg.update(engine_kwargs)

        self.max_episode_steps = self._base_cfg.get("max_episode_steps", max_episode_steps)

        self.starvation_limit = 30 * 60
        self._ticks_since_food = 0

        self._pygame_initialised = False
        self._screen = None
        self._clock = None
        self.engine = None
        self._engine = None

        # Actions: absolute cardinals
        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.action_space = spaces.Discrete(4)

        # Observation: 11x11 local grid (flattened) + 1 fright-time scalar
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(122,), dtype=np.float32)

        self._prev_score = 0
        self._prev_lives = 3
        self._won_already = False
        self._levels_completed = 0
        self._last_action = None
        # Last committed direction in cardinal index space; defaults to UP
        self._last_cardinal_dir = self.UP
        self._recent_tiles = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_pygame()

        cfg = self._base_cfg.copy()
        if seed is not None: cfg["maze_seed"] = seed

        self.engine = GameEngine(**cfg)
        self._engine = self.engine
        self.engine.game_state = GameState.GAME
        self.engine.paused = False

        self._prev_score = 0
        self._prev_lives = self.engine.lives
        self._won_already = False
        self._step_count = 0
        self._levels_completed = 0
        self._last_action = None
        self._last_cardinal_dir = self.UP

        self._recent_tiles = []
        self._ticks_since_food = 0

        return self._get_obs(), {}

    def step(self, action: int):
        # Map discrete action (0..3) directly to absolute cardinal direction
        assert self.action_space.contains(action), f"Invalid action {action} for Discrete(4)"

        # --- Snapshot pre-step game state for event detection ---
        pre_lives = self.engine.lives
        pre_won = self.engine.won

        # Entity counts BEFORE update for direct reward querying
        pre_pellets = len(self.engine.pellets)
        pre_power = len(self.engine.power_pellets)
        pre_eaten = sum(1 for g in self.engine.ghosts if g.state == GhostState.EATEN)

        # Interpret action as absolute cardinal
        target_dir = action  # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT

        # No auto-correction: pass action directly. If it points into a wall,
        # the engine's movement/collision logic will prevent motion and Pac-Man
        # will effectively not move this tick.
        self._last_cardinal_dir = target_dir
        self.engine.pacman.next_direction = self._CARDINAL_TO_VEC[target_dir]
        self._last_action = action

        # --- Single-tick engine advance ---
        self.engine.update()
        self._step_count += 1
        self._ticks_since_food += 1

        # --- Normalized reward calculation (event-based only) ---
        reward = 0.0

        # Time tax every step
        reward -= 0.01

        # Entity counts AFTER update
        cur_pellets = len(self.engine.pellets)
        cur_power = len(self.engine.power_pellets)
        cur_eaten = sum(1 for g in self.engine.ghosts if g.state == GhostState.EATEN)

        # Pellet consumption
        pellets_eaten = max(0, pre_pellets - cur_pellets)
        reward += 0.1 * float(pellets_eaten)

        # Power pellet consumption
        power_eaten = max(0, pre_power - cur_power)
        reward += 1.0 * float(power_eaten)

        # Vulnerable ghosts eaten (transition into EATEN state)
        ghosts_eaten = max(0, cur_eaten - pre_eaten)
        reward += 5.0 * float(ghosts_eaten)

        # Level completed (win state this tick only)
        if self.engine.won and not pre_won:
            reward += 10.0

        # Lethal ghost collision / life lost (death this tick only)
        if self.engine.lives < pre_lives:
            reward -= 10.0

        # Starvation condition used for termination AND penalty
        starved = (self._ticks_since_food >= self.starvation_limit)
        if starved:
            reward -= 10.0

        # Time-limit truncation
        truncated = self._step_count >= self.max_episode_steps
        terminated = self.engine.game_over or self.engine.won or starved


        if self.render_mode == "human":
            self._render_human()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Return the current observation as a flattened 11x11 local grid + fright time."""
        return self._get_vector_obs()

    def _get_vector_obs(self) -> np.ndarray:
        """Dense 11x11 local grid centred on Pac-Man's tile, flattened to 121 values
        plus 1 normalized fright-time scalar.
        """
        eng = self.engine
        ts = eng.tile_size

        # Centre of Pac-Man in pixel coordinates
        px = eng.pacman.x + ts / 2.0
        py = eng.pacman.y + ts / 2.0

        # Current tile indices for Pac-Man (tile centred)
        tx = int(px // ts)
        ty = int(py // ts)

        # Precompute entity tile sets
        food_tiles: set[tuple[int, int]] = set()
        for x, y in eng.pellets:
            gx = int(x // ts)
            gy = int(y // ts)
            food_tiles.add((gx, gy))

        power_tiles: set[tuple[int, int]] = set()
        for x, y in eng.power_pellets:
            gx = int(x // ts)
            gy = int(y // ts)
            power_tiles.add((gx, gy))

        lethal_ghost_tiles: set[tuple[int, int]] = set()
        edible_ghost_tiles: set[tuple[int, int]] = set()
        for g in eng.ghosts:
            gx = int((g.x + ts / 2.0) // ts)
            gy = int((g.y + ts / 2.0) // ts)
            if g.state == GhostState.FRIGHTENED:
                edible_ghost_tiles.add((gx, gy))
            elif g.state != GhostState.EATEN:
                # Treat all non-frightened, non-eaten ghosts as lethal
                lethal_ghost_tiles.add((gx, gy))

        width, height = eng.maze.width, eng.maze.height

        grid_values: list[float] = []
        radius = 5

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                gx = tx + dx
                gy = ty + dy

                # Out of bounds -> wall
                if not (0 <= gx < width and 0 <= gy < height):
                    grid_values.append(-1.0)
                    continue

                tile = eng.maze.maze[gy][gx]

                # Base encoding from maze layout
                if tile == 1:
                    value = -1.0  # wall
                else:
                    value = 0.0  # empty floor

                # Overlay entities in priority order.
                # 1) Ghosts (lethal / edible)
                if (gx, gy) in lethal_ghost_tiles:
                    value = -0.8
                elif (gx, gy) in edible_ghost_tiles:
                    value = 0.8

                # 2) Power pellet
                if (gx, gy) in power_tiles and value == 0.0:
                    value = 1.0

                # 3) Regular pellet
                if (gx, gy) in food_tiles and value == 0.0:
                    value = 0.5

                grid_values.append(value)

        # Normalized remaining fright time feature
        fright_fraction = 0.0
        if getattr(eng, "frightened_mode", False):
            duration = max(1, int(getattr(eng, "frightened_duration", 1)))
            timer = int(getattr(eng, "frightened_timer", 0))
            # Clamp to [0, 1]
            fright_fraction = max(0.0, min(1.0, (duration - timer) / float(duration)))

        grid_values.append(fright_fraction)

        return np.array(grid_values, dtype=np.float32)

    def _get_info(self) -> dict:
        return {"score": self.engine.pacman.score, "frightened": self.engine.frightened_mode}

    def _ensure_pygame(self):
        if self._pygame_initialised: return
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy") if self.render_mode is None else None
        pygame.init()
        if self.render_mode == "human":
            res = self._base_cfg.get("window_resolution", "800x800")
            w, h = map(int, res.split('x'))
            self._screen = pygame.display.set_mode((w, h))
            self._clock = pygame.time.Clock()
        self._pygame_initialised = True

    def _draw_debug_sensors(self): return
    def _render_human(self):
        if not self._screen: return
        pygame.event.pump()
        self._screen.fill((0,0,0))
        self.engine.draw(self._screen)
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._pygame_initialised: pygame.quit(); self._pygame_initialised = False

    @staticmethod
    def _parse_res(res_str):
        w, h = res_str.split('x')
        return int(w), int(h)