"""
PacManEnv.py
============
Egocentric Raycast Version.

Observation (float32):
    - 4 global cardinal raycasts: for each direction, 4 floats:
        [ray_wall, ray_food, ray_power_pellet, ghost_signal]
      where ghost_signal is signed in [-1,1]:
      > 0 lethal ghost, < 0 frightened ghost, 0 = none.
      Directions (global frame):
        0: UP, 1: DOWN, 2: LEFT, 3: RIGHT.
    - Rays are re-ordered into Pac-Man's egocentric frame based on current heading.
    - 3 BFS nearest-entity features (global awareness):
        [d_nearest_food, d_nearest_ghost_dangerous, d_nearest_ghost_edible]
      Distances are 1 / (1 + clipped_shortest_path_distance), with clipping controlled
      by obs_distance_horizon (default 20).
    - 2 power-state scalars:
        [is_powered, power_time_remaining_normalized].
Total observation size: 21 floats (4 * 4 + 3 + 2).

Action Space: Discrete(4) — egocentric relative actions
    0: FORWARD, 1: LEFT, 2: RIGHT, 3: BACKWARD

Each env.step() fast-forwards physics via an internal while-loop and only
returns when Pac-Man reaches the centre of a new tile (next decision point)
or the episode terminates / truncates. Rewards are accumulated across
all internal ticks.
"""

import sys
import os
from collections import deque
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

    # 4 global cardinal ray directions (dx, dy) in tile coordinates
    _GLOBAL_RAY_DIRS = [
        (0, -1),   # UP
        (0, 1),    # DOWN
        (-1, 0),   # LEFT
        (1, 0),    # RIGHT
    ]

    def __init__(self, render_mode: str | None = None, obs_type: str = "vector", settings: dict | None = None, settings_path: str | None = None, max_episode_steps: int | None = 10000, maze_seed: int | None = None, **engine_kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.maze_seed = maze_seed
        self._step_count = 0

        self._base_cfg = _load_settings(settings if settings else settings_path)
        self._base_cfg.update(engine_kwargs)

        raw_max_steps = self._base_cfg.get("max_episode_steps", max_episode_steps)
        # None (or <= 0) disables time-based truncation so starvation/game-over decide episode end.
        if raw_max_steps is None or float(raw_max_steps) <= 0:
            self.max_episode_steps = None
        else:
            self.max_episode_steps = int(raw_max_steps)

        # Exact centre-lock mode: keep this popped so it never leaks into GameEngine kwargs.
        self._base_cfg.pop("tile_center_tolerance", None)

        # Clip very far distances so distant entities don't add excess noise.
        raw_obs_horizon = self._base_cfg.pop("obs_distance_horizon", 20)
        self.obs_distance_horizon = max(1, int(raw_obs_horizon))

        self.starvation_limit = 30 * 60
        self._ticks_since_food = 0

        self._pygame_initialised = False
        self._screen = None
        self._clock = None
        self.engine = None

        self.action_space = spaces.Discrete(4)

        # Observation channels are mostly [0,1], ghost ray remains signed [-1,1].
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(21,), dtype=np.float32)

        self._last_action = None
        self._last_cardinal_dir = self.UP

        self.pellets_eaten_this_episode = 0
        self.ghosts_eaten_this_episode = 0
        self.power_pellets_eaten_this_episode = 0

        self._visited_tiles: set[tuple[int, int]] = set()
        self._visit_counts: dict[tuple[int, int], int] = {}
        self._bfs_cache: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
        self._total_explorable_tiles: int = 0

        self.current_stage = None
        self.current_epsilon = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_pygame()

        cfg = self._base_cfg.copy()
        if seed is not None:
            cfg["maze_seed"] = seed
            np.random.seed(seed)

        self.engine = GameEngine(**cfg)
        self.engine.game_state = GameState.GAME
        self.engine.paused = False

        self._step_count = 0
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
        self._visit_counts.clear()
        self._bfs_cache = {}
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

    def _valid_cardinal_dirs(self, tile_x: int, tile_y: int) -> list[int]:
        eng = self.engine
        valid = []
        for cdir, (dx, dy) in self._CARDINAL_TO_VEC.items():
            nx, ny = tile_x + dx, tile_y + dy
            if 0 <= nx < eng.maze.width and 0 <= ny < eng.maze.height and eng.maze.maze[ny][nx] == 0:
                valid.append(cdir)
        return valid

    def get_valid_actions(self) -> list[int]:
        """Return only physically valid egocentric actions from current tile/heading."""
        if self.engine is None:
            return [self.FORWARD, self.LEFT, self.RIGHT, self.BACKWARD]

        ts = self.engine.tile_size
        px = self.engine.pacman.x + self.engine.pacman.size // 2
        py = self.engine.pacman.y + self.engine.pacman.size // 2
        tx, ty = int(px // ts), int(py // ts)
        heading = self._get_pacman_heading_cardinal()
        valid_dirs = self._valid_cardinal_dirs(tx, ty)

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

        action_to_dir = {
            self.FORWARD: heading,
            self.LEFT: left_map[heading],
            self.RIGHT: right_map[heading],
            self.BACKWARD: backward_map[heading],
        }

        actions = [a for a, target in action_to_dir.items() if target in valid_dirs]
        # Keep all physically valid actions, including BACKWARD in corridors,
        # so the policy can learn tactical reversals when needed.
        return actions if actions else [self.FORWARD, self.LEFT, self.RIGHT, self.BACKWARD]


    def _bfs_shortest_path_distances(self, start_tx: int, start_ty: int) -> dict[tuple[int, int], int]:
        """Return shortest path length in tiles from Pac-Man tile to reachable maze tiles."""
        eng = self.engine
        maze = eng.maze

        if not (0 <= start_tx < maze.width and 0 <= start_ty < maze.height):
            return {}

        distances: dict[tuple[int, int], int] = {(start_tx, start_ty): 0}
        queue = deque([(start_tx, start_ty)])

        while queue:
            cx, cy = queue.popleft()
            base_dist = distances[(cx, cy)]
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < maze.width and 0 <= ny < maze.height):
                    continue
                if (nx, ny) in distances:
                    continue
                # Pac-Man shortest path should respect passable floor only.
                if maze.maze[ny][nx] != 0:
                    continue
                distances[(nx, ny)] = base_dist + 1
                queue.append((nx, ny))

        return distances

    def _distance_to_signal(self, distance: int) -> float:
        """Convert path/raycast distance to a bounded signal with clipping."""
        d = min(int(distance), self.obs_distance_horizon)
        return 1.0 / (1.0 + float(d))

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
                    w_d = self._distance_to_signal(d)
                    break

                if f_d == 0.0 and (cx, cy) in food_tiles:
                    f_d = self._distance_to_signal(d)

                if p_d == 0.0 and (cx, cy) in power_tiles:
                    p_d = self._distance_to_signal(d)

                if g_d == 0.0:
                    if (cx, cy) in lethal_ghost_tiles:
                        g_d = self._distance_to_signal(d)
                    elif (cx, cy) in edible_ghost_tiles:
                        g_d = -self._distance_to_signal(d)

            global_rays.extend([w_d, f_d, p_d, g_d])

        heading = self._get_pacman_heading_cardinal()
        heading_mapping = {
            self.UP:      [0, 1, 2, 3],
            self.DOWN:    [1, 0, 3, 2],
            self.LEFT_C:  [2, 3, 1, 0],
            self.RIGHT_C: [3, 2, 0, 1],
        }
        order = heading_mapping.get(heading, heading_mapping[self.UP])

        obs = []
        for idx in order:
            base = idx * 4
            obs.extend(global_rays[base: base + 4])

        if not hasattr(self, "_bfs_cache"):
            self._bfs_cache = {}

        key = (tx, ty)
        if key not in self._bfs_cache:
            self._bfs_cache[key] = self._bfs_shortest_path_distances(tx, ty)

        sp_dist = self._bfs_cache[key]

        def inv_nearest(target_tiles: set[tuple[int, int]]) -> float:
            if not target_tiles:
                return 0.0
            best = float("inf")
            for tile in target_tiles:
                d = sp_dist.get(tile)
                if d is not None and d < best:
                    best = d
            if best == float("inf"):
                return 0.0
            return self._distance_to_signal(int(best))

        near_food = inv_nearest(food_tiles)
        near_danger = inv_nearest(lethal_ghost_tiles)
        near_edible = inv_nearest(edible_ghost_tiles)

        obs.append(near_food)
        obs.append(near_danger)
        obs.append(near_edible)

        is_powered = 1.0 if eng.frightened_mode else 0.0
        power_remaining = 0.0
        if eng.frightened_mode and getattr(eng, "frightened_duration", 0) > 0:
            remain = max(0, eng.frightened_duration - eng.frightened_timer)
            power_remaining = max(0.0, min(1.0, float(remain) / float(eng.frightened_duration)))

        obs.append(is_powered)
        obs.append(power_remaining)

        obs = np.array(obs, dtype=np.float32)

        # Keep distances/context in [0,1]; only ghost ray channel is signed.
        return obs

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        eng = self.engine
        ts = eng.tile_size

        # 1. Map egocentric action to cardinal direction ONCE
        heading = self._get_pacman_heading_cardinal()

        valid_actions = self.get_valid_actions()
        blocked_action = action not in valid_actions

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

        # Do not rewrite invalid actions into random valid ones.
        # This keeps learning targets honest: illegal choices are punished directly.
        if blocked_action:
            self._last_action = int(action)
            self._step_count += 1
            self._ticks_since_food += 1

            accumulated_reward = -5.0
            reward_breakdown = {
                "pellet_reward": 0.0,
                "power_reward": 0.0,
                "ghost_reward": 0.0,
                "win_reward": 0.0,
                "death_penalty": 0.0,
                "starvation_penalty": 0.0,
                "food_shaping_reward": 0.0,
                "invalid_action_penalty": -5.0,
                "living_penalty": 0.0,
                "total": 0.0,
            }

            starved = self._ticks_since_food >= self.starvation_limit
            if starved:
                accumulated_reward -= 100.0
                reward_breakdown["starvation_penalty"] -= 100.0

            accumulated_reward -= 0.5
            reward_breakdown["living_penalty"] -= 0.5

            terminated = bool(starved)
            truncated = (self.max_episode_steps is not None) and (self._step_count >= self.max_episode_steps)

            px = eng.pacman.x + eng.pacman.size // 2
            py = eng.pacman.y + eng.pacman.size // 2
            cur_tx = int(px // ts)
            cur_ty = int(py // ts)
            tile_key = (cur_tx, cur_ty)
            visit_count = self._visit_counts.get(tile_key, 0) + 1
            self._visit_counts[tile_key] = visit_count
            if tile_key not in self._visited_tiles:
                self._visited_tiles.add(tile_key)

            reward_breakdown["total"] = float(accumulated_reward)

            if starved:
                death_cause = "STARVATION"
            elif truncated:
                death_cause = "MAX_STEPS"
            else:
                death_cause = "NONE"

            info = self._get_info()
            info.update({
                "death_cause": death_cause,
                "internal_ticks": 0,
                "tile_center": (cur_tx, cur_ty),
                "center_lock_mode": "exact",
                "steps": int(self._step_count),
                "action": int(action),
                "target_dir": int(target_dir),
                "visit_count": int(visit_count),
                "blocked_action": True,
                "reward_breakdown": reward_breakdown,
            })

            if self.render_mode == "human":
                self._render_human()
            return self._get_obs(), accumulated_reward, terminated, truncated, info

        self._last_action = int(action)
        self._last_cardinal_dir = target_dir
        eng.pacman.next_direction = self._CARDINAL_TO_VEC[target_dir]

        accumulated_reward = 0.0

        reward_breakdown = {
            "pellet_reward": 0.0,
            "power_reward": 0.0,
            "ghost_reward": 0.0,
            "win_reward": 0.0,
            "death_penalty": 0.0,
            "starvation_penalty": 0.0,
            "food_shaping_reward": 0.0,
            "living_penalty": 0.0,
            "total": 0.0,
        }

        # 21D observation layout: near_food is index 16.
        initial_obs = self._get_obs()
        dist_food_before = float(initial_obs[16]) if initial_obs.shape[0] >= 21 else 0.0

        # Pre-loop topo-lock tracking
        px_start = eng.pacman.x + eng.pacman.size // 2
        py_start = eng.pacman.y + eng.pacman.size // 2
        start_tx = int(px_start // ts)
        start_ty = int(py_start // ts)

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
            pre_px_center = pre_pac_x + eng.pacman.size // 2
            pre_py_center = pre_pac_y + eng.pacman.size // 2
            eng.update()
            internal_ticks += 1
            self._step_count += 1
            self._ticks_since_food += 1

            reward_tick = 0.0

            # Pellets
            pellets_eaten = max(0, pre_pellets - len(eng.pellets))
            if pellets_eaten > 0:
                pellet_gain = 10.0 * pellets_eaten
                reward_tick += pellet_gain
                reward_breakdown["pellet_reward"] += pellet_gain
                self._ticks_since_food = 0
                self.pellets_eaten_this_episode += pellets_eaten
                self._bfs_cache.clear()

            # Power pellets
            power_eaten = max(0, pre_power - len(eng.power_pellets))
            if power_eaten > 0:
                power_gain = 50.0 * power_eaten
                reward_tick += power_gain
                reward_breakdown["power_reward"] += power_gain
                self._ticks_since_food = 0
                self.power_pellets_eaten_this_episode += power_eaten
                self._bfs_cache.clear()

            # Ghosts
            ghosts_eaten = max(0, sum(1 for g in eng.ghosts if g.state == GhostState.EATEN) - pre_eaten)
            if ghosts_eaten > 0:
                ghost_gain = 200.0 * ghosts_eaten
                reward_tick += ghost_gain
                reward_breakdown["ghost_reward"] += ghost_gain
                self.ghosts_eaten_this_episode += ghosts_eaten

            # Terminal events
            if eng.won and not pre_won:
                reward_tick += 1000.0
                reward_breakdown["win_reward"] += 1000.0

            if eng.lives < pre_lives:
                reward_tick -= 500.0
                reward_breakdown["death_penalty"] -= 500.0

            # Starvation
            starved = self._ticks_since_food >= self.starvation_limit
            if starved:
                reward_tick -= 100.0
                reward_breakdown["starvation_penalty"] -= 100.0

            accumulated_reward += reward_tick

            terminated = eng.game_over or eng.won or starved
            truncated = (self.max_episode_steps is not None) and (self._step_count >= self.max_episode_steps)

            if terminated or truncated:
                break

            # Break Condition (Exact Tile-Centre Lock)
            px = eng.pacman.x + eng.pacman.size // 2
            py = eng.pacman.y + eng.pacman.size // 2
            cur_tx = int(px // ts)
            cur_ty = int(py // ts)

            # Only return control once we've left the starting tile and
            # reached (or crossed) the exact centre of the new tile.
            if (cur_tx != start_tx) or (cur_ty != start_ty):
                center_x = (cur_tx * ts) + (ts // 2)
                center_y = (cur_ty * ts) + (ts // 2)

                on_center = (px == center_x and py == center_y)
                crossed_center = (
                    (pre_px_center - center_x) * (px - center_x) <= 0
                    and (pre_py_center - center_y) * (py - center_y) <= 0
                )

                if on_center or crossed_center:
                    # Snap to exact geometric centre of the NEW tile.
                    eng.pacman.x = center_x - (eng.pacman.size // 2)
                    eng.pacman.y = center_y - (eng.pacman.size // 2)
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

        # --- Exploration (new tile bonus + revisit penalty) ---
        tile_key = (cur_tx, cur_ty)
        visit_count = self._visit_counts.get(tile_key, 0) + 1
        self._visit_counts[tile_key] = visit_count

        if tile_key not in self._visited_tiles:
            self._visited_tiles.add(tile_key)

        new_obs = self._get_obs()
        dist_food_after = float(new_obs[16]) if new_obs.shape[0] >= 21 else 0.0
        shaping_reward = (dist_food_after - dist_food_before) * 5.0
        accumulated_reward += shaping_reward
        reward_breakdown["food_shaping_reward"] += shaping_reward

        accumulated_reward -= 0.5
        reward_breakdown["living_penalty"] -= 0.5

        reward_breakdown["total"] = float(accumulated_reward)

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
            "center_lock_mode": "exact",
            # Total high-level environment steps taken so far in this episode
            "steps": int(self._step_count),
            "action": int(action),
            "target_dir": int(target_dir),
            "visit_count": int(visit_count),
            "blocked_action": False,
            "reward_breakdown": reward_breakdown,
        })

        if self.render_mode == "human": self._render_human()
        return self._get_obs(), accumulated_reward, terminated, truncated, info

    def _get_info(self) -> dict:
        pellets_remaining = len(self.engine.pellets) + len(self.engine.power_pellets)
        map_clear_pct = (len(self._visited_tiles) / self._total_explorable_tiles * 100.0) if self._total_explorable_tiles > 0 else 0.0
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
            "pellets_remaining": pellets_remaining,
            "map_clear_pct": map_clear_pct,
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