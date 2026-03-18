"""
PacManEnv.py
============
Egocentric Raycast Version.

Observation (float32):
    - 8 global raycasts: for each direction, 4 floats:
        [1/d_wall, 1/d_food, 1/d_power_pellet, signed_ghost]
      where signed_ghost > 0 means lethal ghost, < 0 means frightened ghost, 0 = none.
      Directions (global frame):
        0: UP, 1: DOWN, 2: LEFT, 3: RIGHT,
        4: UP-LEFT, 5: UP-RIGHT, 6: DOWN-LEFT, 7: DOWN-RIGHT.
    - Rays are re-ordered into Pac-Man's egocentric frame based on current heading.
    - 1 scalar frightened flag / timer in [0,1].

Total observation size: 33 floats (8 * 4 + 1).

Action Space: Discrete(4) — egocentric relative actions
    0: FORWARD, 1: LEFT, 2: RIGHT, 3: BACKWARD

Each env.step() fast-forwards physics via an internal while-loop and only
returns when Pac-Man reaches the centre of a new tile (next decision point)
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

        self.action_space = spaces.Discrete(4)

        # Observation: 8 rays * (wall, food, power_pellet, ghost) + frightened flag = 33
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(33,), dtype=np.float32)

        self._prev_score = 0
        self._prev_lives = 3
        self._won_already = False
        self._levels_completed = 0
        self._last_action = None
        self._last_cardinal_dir = self.UP

        self.pellets_eaten_this_episode = 0
        self.ghosts_eaten_this_episode = 0
        self.power_pellets_eaten_this_episode = 0

        self._visited_tiles: set[tuple[int, int]] = set()
        self._total_explorable_tiles: int = 0

        self.current_stage = None
        self.current_epsilon = None

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
        dx, dy = self.engine.pacman.direction
        if (dx, dy) == (0, 0):
            self._last_cardinal_dir = self.RIGHT_C
        else:
            self._last_cardinal_dir = self._get_pacman_heading_cardinal()

        self._ticks_since_food = 0

        self.pellets_eaten_this_episode = 0
        self.ghosts_eaten_this_episode = 0
        self.power_pellets_eaten_this_episode = 0

        self._visited_tiles.clear()
        self._total_explorable_tiles = 0

        maze = self.engine.maze
        for y in range(maze.height):
            for x in range(maze.width):
                if maze.maze[y][x] == 0:
                    self._total_explorable_tiles += 1

        return self._get_obs(), {}

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

    def _get_obs(self) -> np.ndarray:
        return self._get_vector_obs()

    def _get_vector_obs(self) -> np.ndarray:
        eng = self.engine
        ts = eng.tile_size

        px = eng.pacman.x + ts / 2.0
        py = eng.pacman.y + ts / 2.0
        tx = int(px // ts)
        ty = int(py // ts)

        food_tiles = set()
        power_tiles = set()
        for x, y in eng.pellets:
            food_tiles.add((int(x // ts), int(y // ts)))
        for x, y in eng.power_pellets:
            power_tiles.add((int(x // ts), int(y // ts)))

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

        for dx, dy in self._GLOBAL_RAY_DIRS:
            d = 0
            w_d = 0.0
            f_d = 0.0
            p_d = 0.0
            g_d = 0.0

            cx, cy = tx, ty
            while True:
                cx += dx
                cy += dy
                d += 1

                if not (0 <= cx < eng.maze.width and 0 <= cy < eng.maze.height) or eng.maze.maze[cy][cx] == 1:
                    w_d = 1.0 / d
                    break

                if f_d == 0.0 and (cx, cy) in food_tiles:
                    f_d = 1.0 / d

                if p_d == 0.0 and (cx, cy) in power_tiles:
                    p_d = 1.0 / d

                if g_d == 0.0:
                    if (cx, cy) in lethal_ghost_tiles:
                        g_d = 1.0 / d
                    elif (cx, cy) in edible_ghost_tiles:
                        g_d = -1.0 / d

            global_rays.extend([w_d, f_d, p_d, g_d])

        heading = self._get_pacman_heading_cardinal()
        heading_mapping = {
            self.UP:      [0, 1, 2, 3, 4, 5, 6, 7],
            self.DOWN:    [1, 0, 3, 2, 7, 6, 5, 4],
            self.LEFT_C:  [2, 3, 1, 0, 6, 4, 7, 5],
            self.RIGHT_C: [3, 2, 0, 1, 5, 7, 4, 6],
        }
        order = heading_mapping.get(heading, heading_mapping[self.UP])

        obs = []
        for idx in order:
            base = idx * 4
            obs.extend(global_rays[base : base + 4])

        max_ft = getattr(eng, "max_frightened_time", None)
        cur_ft = getattr(eng, "frightened_time", 0)
        if max_ft is None or max_ft <= 0:
            frightened_val = 1.0 if eng.frightened_mode else 0.0
        else:
            frightened_val = max(0.0, min(1.0, float(cur_ft) / float(max_ft)))
        obs.append(frightened_val)

        return np.array(obs, dtype=np.float32)

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        eng = self.engine
        ts = eng.tile_size

        # 1. Map egocentric action to cardinal direction ONCE
        heading = self._get_pacman_heading_cardinal()

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
        else:
            target_dir = backward_map[heading]

        self._last_cardinal_dir = target_dir
        eng.pacman.next_direction = self._CARDINAL_TO_VEC[target_dir]

        accumulated_reward = 0.0

        # --- Safe Distance Calculation Helpers ---
        def get_min_dist(pac_tx, pac_ty, entities):
            if not entities: return float('inf')
            return min(abs(pac_tx - int(e[0] // ts)) + abs(pac_ty - int(e[1] // ts)) for e in entities)

        def get_min_ghost_dist(pac_tx, pac_ty, frightened=False):
            if not eng.ghosts: return float('inf')
            target_ghosts = []
            for g in eng.ghosts:
                if g.state == GhostState.EATEN: continue
                if frightened and g.state == GhostState.FRIGHTENED:
                    target_ghosts.append(g)
                elif not frightened and g.state != GhostState.FRIGHTENED:
                    target_ghosts.append(g)
            if not target_ghosts: return float('inf')
            return min(abs(pac_tx - int(g.x // ts)) + abs(pac_ty - int(g.y // ts)) for g in target_ghosts)

        # Pre-loop distances and Topo-Lock tracking
        px_start = eng.pacman.x + eng.pacman.size // 2
        py_start = eng.pacman.y + eng.pacman.size // 2
        start_tx = int(px_start // ts)
        start_ty = int(py_start // ts)

        prev_food_dist = get_min_dist(start_tx, start_ty, eng.pellets)
        prev_frightened_dist = get_min_ghost_dist(start_tx, start_ty, frightened=True)

        # 2. THE TILE-LOCK LOOP
        #
        # This loop advances the underlying game until one of the following
        # happens:
        #   * the episode terminates / truncates, or
        #   * Pac-Man leaves the starting tile and reaches the geometric
        #     centre of the *next* tile (our RL decision point).
        #
        # No further high-level actions are read inside this loop; the only
        # control input is the "next_direction" we set above.
        internal_ticks = 0
        # Guard against long spins if movement is blocked but direction stays non-zero.
        no_progress_ticks = 0
        max_no_progress_ticks = max(2, ts // max(1, eng.pacman.speed))
        starved = False
        terminated = False
        truncated = False
        while True:
            pre_lives = eng.lives
            pre_won = eng.won
            pre_pellets = len(eng.pellets)
            pre_power = len(eng.power_pellets)
            pre_eaten = sum(1 for g in eng.ghosts if g.state == GhostState.EATEN)

            pre_pac_x = eng.pacman.x
            pre_pac_y = eng.pacman.y
            eng.update()
            internal_ticks += 1
            self._step_count += 1
            self._ticks_since_food += 1

            reward_tick = -0.02  # stronger time pressure

            # Pellets
            pellets_eaten = max(0, pre_pellets - len(eng.pellets))
            if pellets_eaten > 0:
                reward_tick += 1.0 * pellets_eaten
                self._ticks_since_food = 0
                self.pellets_eaten_this_episode += pellets_eaten

            # Power pellets
            power_eaten = max(0, pre_power - len(eng.power_pellets))
            if power_eaten > 0:
                reward_tick += 5.0 * power_eaten
                self._ticks_since_food = 0
                self.power_pellets_eaten_this_episode += power_eaten

            # Ghosts
            ghosts_eaten = max(0, sum(1 for g in eng.ghosts if g.state == GhostState.EATEN) - pre_eaten)
            if ghosts_eaten > 0:
                reward_tick += 10.0 * ghosts_eaten
                self.ghosts_eaten_this_episode += ghosts_eaten

            # Terminal events
            if eng.won and not pre_won:
                reward_tick += 100.0

            if eng.lives < pre_lives:
                reward_tick -= 75.0  # stronger penalty

            # Starvation
            starved = self._ticks_since_food >= self.starvation_limit
            if starved:
                reward_tick -= 25.0  # reduced (less overlap with time penalty)

            accumulated_reward += reward_tick

            terminated = eng.game_over or eng.won or starved
            truncated = self._step_count >= self.max_episode_steps

            if terminated or truncated:
                break

            # Break Condition (Topological Node / Tile-Centre Fix)
            px = eng.pacman.x + eng.pacman.size // 2
            py = eng.pacman.y + eng.pacman.size // 2
            cur_tx = int(px // ts)
            cur_ty = int(py // ts)

            offset_x = px % ts
            offset_y = py % ts
            tolerance = max(eng.pacman.speed, 2)

            # Only treat this as a new decision point once we've actually
            # *left* the starting tile and are very close to the new tile
            # centre. We deliberately do NOT update (start_tx, start_ty)
            # inside the loop so that exactly one break occurs per tile
            # transition.
            if (cur_tx != start_tx) or (cur_ty != start_ty):
                if abs(offset_x - ts // 2) <= tolerance and abs(offset_y - ts // 2) <= tolerance:
                    # Snap perfectly to the NEW tile's centre to remove any
                    # accumulated sub-tile drift.
                    eng.pacman.x = (cur_tx * ts) + (ts // 2) - (eng.pacman.size // 2)
                    eng.pacman.y = (cur_ty * ts) + (ts // 2) - (eng.pacman.size // 2)
                    break

            # If no movement happened this tick, count it and fail safe after
            # a short window to return control to the policy.
            if eng.pacman.x == pre_pac_x and eng.pacman.y == pre_pac_y:
                no_progress_ticks += 1
            else:
                no_progress_ticks = 0

            if no_progress_ticks >= max_no_progress_ticks:
                eng.pacman.x = (cur_tx * ts) + (ts // 2) - (eng.pacman.size // 2)
                eng.pacman.y = (cur_ty * ts) + (ts // 2) - (eng.pacman.size // 2)
                break

            # Safety catch: if movement stalls, return control to the policy
            # from this centred tile instead of auto-picking a direction.
            if eng.pacman.direction == (0, 0):
                eng.pacman.x = (cur_tx * ts) + (ts // 2) - (eng.pacman.size // 2)
                eng.pacman.y = (cur_ty * ts) + (ts // 2) - (eng.pacman.size // 2)
                break

        # 3. Post-Loop Dense Rewards (Applied once per Tile)
        px = eng.pacman.x + eng.pacman.size // 2
        py = eng.pacman.y + eng.pacman.size // 2
        cur_tx = int(px // ts)
        cur_ty = int(py // ts)

        # Normalize reward to remove variable tick bias
        accumulated_reward /= max(1, internal_ticks)

        # --- Exploration (decaying) ---
        if (cur_tx, cur_ty) not in self._visited_tiles:
            decay = 1.0 - (self._step_count / self.max_episode_steps)
            accumulated_reward += 0.05 * max(0.0, decay)
            self._visited_tiles.add((cur_tx, cur_ty))

        # --- Food shaping ---
        curr_food_dist = get_min_dist(cur_tx, cur_ty, eng.pellets)
        if curr_food_dist < prev_food_dist and curr_food_dist != float('inf'):
            accumulated_reward += 0.05

        # --- Ghost avoidance ---
        if not eng.frightened_mode:
            ghost_dist = get_min_ghost_dist(cur_tx, cur_ty, frightened=False)
            if ghost_dist < 5:
                accumulated_reward -= 2.0 / max(1, ghost_dist)

        # --- Ghost chasing (frightened mode) ---
        if eng.frightened_mode:
            curr_frightened_dist = get_min_ghost_dist(cur_tx, cur_ty, frightened=True)
            if curr_frightened_dist < prev_frightened_dist and curr_frightened_dist != float('inf'):
                accumulated_reward += 0.1

        # 4. Post-Loop Cleanup
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

        info = self._get_info()
        info.update({
            "death_cause": death_cause,
            # Debug/analysis fields
            "internal_ticks": internal_ticks,
            "tile_center": (cur_tx, cur_ty),
            # Total high-level environment steps taken so far in this episode
            "steps": int(self._step_count),
        })

        if self.render_mode == "human": self._render_human()
        return self._get_obs(), accumulated_reward, terminated, truncated, info

    def _get_info(self) -> dict:
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

    def render(self):
        """Standard Gymnasium render hook."""
        if self.render_mode == "rgb_array":
            self._ensure_pygame()
            width = self.engine.maze.width * self.engine.tile_size
            height = self.engine.maze.height * self.engine.tile_size
            temp_surface = pygame.Surface((width, height))
            temp_surface.fill((0, 0, 0))
            self.engine.draw(temp_surface)
            rgb_array = np.transpose(pygame.surfarray.array3d(temp_surface), (1, 0, 2))
            return rgb_array
        elif self.render_mode == "human":
            self._render_human()
            return None

    def _render_human(self):
        if not self._screen: return
        pygame.event.pump()
        self._screen.fill((0,0,0))
        self.engine.draw(self._screen)
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._pygame_initialised:
            pygame.quit()
            self._pygame_initialised = False

    @staticmethod
    def _parse_res(res_str):
        w, h = res_str.split('x')
        return int(w), int(h)