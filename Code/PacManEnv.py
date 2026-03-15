"""
PacManEnv.py
============
Egocentric Raycast Version.

Observation (float32):
    - 8 global raycasts: for each direction, 3 floats:
        [1/d_wall, 1/d_food, signed_ghost]
      where signed_ghost > 0 means lethal ghost, < 0 means frightened ghost, 0 = none.
      Directions (global frame):
        0: UP, 1: DOWN, 2: LEFT, 3: RIGHT,
        4: UP-LEFT, 5: UP-RIGHT, 6: DOWN-LEFT, 7: DOWN-RIGHT.
    - Rays are re-ordered into Pac-Man's egocentric frame based on current heading.
    - 1 scalar frightened flag (1.0 if frightened mode active, else 0.0).

Total observation size: 25 floats (8 * 3 + 1).

Action Space: Discrete(3) — egocentric relative actions
    0: FORWARD, 1: LEFT, 2: RIGHT

Each env.step() fast-forwards physics via an internal while-loop and only
returns when Pac-Man reaches a topological node (intersection or corner)
or the episode terminates / truncates. Rewards are accumulated across
all internal ticks.
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

    # Egocentric actions
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    BACKWARD = 3

    # Cardinal directions
    UP = 0
    DOWN = 1
    LEFT_C = 2
    RIGHT_C = 3

    # Absolute cardinal index -> direction vector (in tile coordinates)
    _CARDINAL_TO_VEC = {UP: (0, -1), DOWN: (0, 1), LEFT_C: (-1, 0), RIGHT_C: (1, 0)}
    _CARDINAL_OPPOSITE = {UP: DOWN, DOWN: UP, LEFT_C: RIGHT_C, RIGHT_C: LEFT_C}

    # 8 global ray directions (dx, dy) in tile coordinates
    _GLOBAL_RAY_DIRS = [
        (0, -1),   # UP
        (0, 1),    # DOWN
        (-1, 0),   # LEFT
        (1, 0),    # RIGHT
        (-1, -1),  # UP-LEFT
        (1, -1),   # UP-RIGHT
        (-1, 1),   # DOWN-LEFT
        (1, 1),    # DOWN-RIGHT
    ]

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

        # Actions: egocentric relative
        # 0: FORWARD, 1: LEFT, 2: RIGHT, 3: BACKWARD
        self.action_space = spaces.Discrete(4)

        # Observation: 8 rays * (wall, food, ghost) + frightened flag = 25
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(25,), dtype=np.float32)

        self._prev_score = 0
        self._prev_lives = 3
        self._won_already = False
        self._levels_completed = 0
        self._last_action = None
        # Last committed direction in cardinal index space; defaults to UP
        self._last_cardinal_dir = self.UP

        # Episode-level telemetry counters
        self.pellets_eaten_this_episode = 0
        self.ghosts_eaten_this_episode = 0
        self.power_pellets_eaten_this_episode = 0

        # Exploration tracking within a maze
        self._visited_tiles: set[tuple[int, int]] = set()
        self._total_explorable_tiles: int = 0

        # Optional external telemetry fields (can be set by trainers)
        self.current_stage = None   # to be provided by curriculum if desired
        self.current_epsilon = None # to be provided by agent if desired


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

        self._ticks_since_food = 0

        # Reset episode-level telemetry counters
        self.pellets_eaten_this_episode = 0
        self.ghosts_eaten_this_episode = 0
        self.power_pellets_eaten_this_episode = 0

        # Reset exploration tracking
        self._visited_tiles.clear()
        self._total_explorable_tiles = 0

        # Precompute total explorable tiles (non-wall tiles) for this maze
        maze = self.engine.maze
        for y in range(maze.height):
            for x in range(maze.width):
                if maze.maze[y][x] == 0:
                    self._total_explorable_tiles += 1

        return self._get_obs(), {}

    # ----------------- Helpers -----------------

    def _get_pacman_heading_cardinal(self) -> int:
        dx, dy = self.engine.pacman.direction
        if (dx, dy) == (0, 0):
            return self._last_cardinal_dir
        if dx == 0 and dy < 0:
            return self.UP
        if dx == 0 and dy > 0:
            return self.DOWN
        if dx < 0 and dy == 0:
            return self.LEFT_C
        return self.RIGHT_C

    # ----------------- Observation -----------------

    def _get_obs(self) -> np.ndarray:
        """25-float egocentric ray observation."""
        return self._get_vector_obs()

    def _get_vector_obs(self) -> np.ndarray:
        """Compute 8 global raycasts (wall, food, ghost) and rotate into egocentric frame,
        then append frightened flag: total 25 floats.
        """
        eng = self.engine
        ts = eng.tile_size

        # Center of Pac-Man in tile coordinates
        px = eng.pacman.x + ts / 2.0
        py = eng.pacman.y + ts / 2.0
        tx = int(px // ts)
        ty = int(py // ts)

        # Precompute discrete sets
        food_tiles = set()
        for x, y in eng.pellets + eng.power_pellets:
            gx = int(x // ts)
            gy = int(y // ts)
            food_tiles.add((gx, gy))

        lethal_ghost_tiles = set()
        edible_ghost_tiles = set()
        for g in eng.ghosts:
            gx = int((g.x + ts / 2.0) // ts)
            gy = int((g.y + ts / 2.0) // ts)
            if g.state == GhostState.FRIGHTENED:
                edible_ghost_tiles.add((gx, gy))
            elif g.state != GhostState.EATEN:
                lethal_ghost_tiles.add((gx, gy))

        global_rays: list[float] = []

        # For each global direction, raycast until wall or out-of-bounds
        for dx, dy in self._GLOBAL_RAY_DIRS:
            d = 0
            w_d = 0.0
            f_d = 0.0
            g_d = 0.0

            cx, cy = tx, ty
            while True:
                cx += dx
                cy += dy
                d += 1

                # Out of bounds or wall ends the ray
                if not (0 <= cx < eng.maze.width and 0 <= cy < eng.maze.height) or eng.maze.maze[cy][cx] == 1:
                    w_d = 1.0 / d
                    break

                # First food tile along this ray
                if f_d == 0.0 and (cx, cy) in food_tiles:
                    f_d = 1.0 / d

                # First ghost tile along this ray
                if g_d == 0.0:
                    if (cx, cy) in lethal_ghost_tiles:
                        g_d = 1.0 / d
                    elif (cx, cy) in edible_ghost_tiles:
                        g_d = -1.0 / d

            global_rays.extend([w_d, f_d, g_d])

        # Rotate into egocentric frame based on heading
        heading = self._get_pacman_heading_cardinal()
        heading_mapping = {
            # rays index: [UP, DOWN, LEFT, RIGHT, UP-LEFT, UP-RIGHT, DOWN-LEFT, DOWN-RIGHT]
            # egocentric order: FRONT, BACK, LEFT, RIGHT, FR-L, FR-R, BK-L, BK-R
            self.UP:      [0, 1, 2, 3, 4, 5, 6, 7],
            self.DOWN:    [1, 0, 3, 2, 7, 6, 5, 4],
            self.LEFT_C:  [2, 3, 1, 0, 6, 4, 7, 5],
            self.RIGHT_C: [3, 2, 0, 1, 5, 7, 4, 6],
        }
        order = heading_mapping.get(heading, heading_mapping[self.UP])

        obs = []
        for idx in order:
            obs.extend(global_rays[idx * 3 : idx * 3 + 3])

        # Frightened timer flag: 1.0 at start of frightened mode, down to 0.0 as it expires
        max_ft = getattr(eng, "max_frightened_time", None)
        cur_ft = getattr(eng, "frightened_time", 0)
        if max_ft is None or max_ft <= 0:
            frightened_val = 1.0 if eng.frightened_mode else 0.0
        else:
            frightened_val = max(0.0, min(1.0, float(cur_ft) / float(max_ft)))
        obs.append(frightened_val)

        return np.array(obs, dtype=np.float32)

    # ----------------- Step (Node-based fast-forward) -----------------

    def step(self, action: int):
        """Egocentric 4-action interface with single-tick step.

        If Pac-Man is between tile centres, we ignore the new action and
        keep his previous direction (i.e. keep moving along the current edge).
        Only when centred on a tile do we apply the new egocentric action.
        """
        assert self.action_space.contains(action), f"Invalid action {action} for Discrete(4)"

        eng = self.engine
        ts = eng.tile_size

        # Check if Pac-Man is near the centre of a tile using a tolerance,
        # mirroring PacMan.is_aligned_to_tile() behaviour.
        px = eng.pacman.x + eng.pacman.size // 2
        py = eng.pacman.y + eng.pacman.size // 2
        tile_offset_x = px % ts
        tile_offset_y = py % ts
        tolerance = max(eng.pacman.speed, ts // 10, 3)
        is_centered = (
            abs(tile_offset_x - ts // 2) < tolerance
            and abs(tile_offset_y - ts // 2) < tolerance
        )

        # Also detect if continuing in the current heading is blocked by a wall.
        heading = self._get_pacman_heading_cardinal()
        dx_h, dy_h = self._CARDINAL_TO_VEC[heading]
        gx = int(px // ts)
        gy = int(py // ts)
        ahead_x = gx + dx_h
        ahead_y = gy + dy_h
        blocked = not (
            0 <= ahead_x < eng.maze.width
            and 0 <= ahead_y < eng.maze.height
            and eng.maze.maze[ahead_y][ahead_x] == 0
        )

        # We allow applying a new egocentric action whenever Pac-Man is
        # approximately centred OR his current heading is blocked.
        if is_centered or blocked:
            left_map = {
                self.UP: self.LEFT_C,
                self.DOWN: self.RIGHT_C,
                self.LEFT_C: self.DOWN,
                self.RIGHT_C: self.UP,
            }
            right_map = {
                self.UP: self.RIGHT_C,
                self.DOWN: self.LEFT_C,
                self.LEFT_C: self.UP,
                self.RIGHT_C: self.DOWN,
            }
            backward_map = {
                self.UP: self.DOWN,
                self.DOWN: self.UP,
                self.LEFT_C: self.RIGHT_C,
                self.RIGHT_C: self.LEFT_C,
            }

            if action == self.FORWARD:
                target_dir = heading
            elif action == self.LEFT:
                target_dir = left_map[heading]
            elif action == self.RIGHT:
                target_dir = right_map[heading]
            else:  # BACKWARD
                target_dir = backward_map[heading]

            self._last_cardinal_dir = target_dir
            eng.pacman.next_direction = self._CARDINAL_TO_VEC[target_dir]
            self._last_action = action
        else:
            # Between tiles with free space ahead: keep moving along previous direction
            eng.pacman.next_direction = self._CARDINAL_TO_VEC[self._last_cardinal_dir]

        # --- Single physics tick ---
        pre_lives = eng.lives
        pre_won = eng.won
        pre_pellets = len(eng.pellets)
        pre_power = len(eng.power_pellets)
        pre_eaten = sum(1 for g in eng.ghosts if g.state == GhostState.EATEN)

        eng.update()
        self._step_count += 1
        self._ticks_since_food += 1

        # Update exploration: mark current tile as visited
        cur_tx = int((eng.pacman.x + ts / 2.0) // ts)
        cur_ty = int((eng.pacman.y + ts / 2.0) // ts)
        if 0 <= cur_tx < eng.maze.width and 0 <= cur_ty < eng.maze.height and eng.maze.maze[cur_ty][cur_tx] == 0:
            self._visited_tiles.add((cur_tx, cur_ty))

        reward = -0.05  # time tax

        cur_pellets = len(eng.pellets)
        cur_power = len(eng.power_pellets)
        cur_eaten = sum(1 for g in eng.ghosts if g.state == GhostState.EATEN)

        pellets_eaten = max(0, pre_pellets - cur_pellets)
        if pellets_eaten > 0:
            reward += 1.0 * float(pellets_eaten)
            self._ticks_since_food = 0  # Reset starvation clock when food is eaten
            self.pellets_eaten_this_episode += pellets_eaten

        power_eaten = max(0, pre_power - cur_power)
        if power_eaten > 0:
            reward += 1.0 * float(power_eaten)
            self._ticks_since_food = 0  # Reset for power pellets too
            self.power_pellets_eaten_this_episode += power_eaten

        ghosts_eaten = max(0, cur_eaten - pre_eaten)
        if ghosts_eaten > 0:
            reward += 10.0 * float(ghosts_eaten)
            self.ghosts_eaten_this_episode += ghosts_eaten

        if eng.won and not pre_won:
            reward += 100.0

        if eng.lives < pre_lives:
            reward -= 200.0

        starved = self._ticks_since_food >= self.starvation_limit
        if starved:
            reward -= 50.0

        # Termination / truncation flags
        truncated = self._step_count >= self.max_episode_steps
        terminated = eng.game_over or eng.won or starved

        # Determine cause of episode outcome (or NONE if still ongoing).
        # Priority: STARVATION > WIN > GHOST > MAX_STEPS > NONE
        if starved:
            death_cause = "STARVATION"
        elif eng.won:
            death_cause = "WIN"
        elif eng.game_over:
            death_cause = "GHOST"
        elif truncated:
            death_cause = "MAX_STEPS"
        else:
            death_cause = "NONE"

        if self.render_mode == "human":
            self._render_human()

        info = self._get_info()
        info["death_cause"] = death_cause

        return self._get_obs(), reward, terminated, truncated, info

    def _get_info(self) -> dict:
        """Return base info dictionary for telemetry.

        Note: step() augments this with fields like "death_cause" before returning.
        """
        return {
            "score": self.engine.pacman.score,
            "frightened": self.engine.frightened_mode,
            "stage": self.current_stage,
            "epsilon": self.current_epsilon,
            "pellets": self.pellets_eaten_this_episode,
            "ghosts": self.ghosts_eaten_this_episode,
            "power_pellets": self.power_pellets_eaten_this_episode,
            "maze_seed": getattr(self.engine, "maze_seed", None),
            "explored_tiles": len(self._visited_tiles),
            "total_explorable_tiles": self._total_explorable_tiles,
            "explore_rate": (len(self._visited_tiles) / self._total_explorable_tiles) if self._total_explorable_tiles > 0 else 0.0,
        }

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