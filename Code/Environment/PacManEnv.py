"""
PacManEnv.py
============
Egocentric Raycast Version — 31-dimensional observation.

Observation layout (float32, all values normalised to [-1, 1]):
─────────────────────────────────────────────────────────────────
  Block A — Egocentric raycasts (4 directions × 6 channels = 24)
  ──────────────────────────────────────────────────────────────
  Directions are re-ordered into Pac-Man's egocentric frame
  (forward / left / right / backward) based on current heading.
  Global frame: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.

  Per-direction channels (indices 0-5, 6-11, 12-17, 18-23):
    [0] wall_dist          : inverse wall distance  1/(1+d), mapped to [-1,1]
    [1] food_dist          : inverse distance to nearest pellet on ray
    [2] power_dist         : inverse distance to nearest power pellet on ray
    [3] lethal_ghost_dist  : inverse distance to nearest lethal ghost on ray
    [4] edible_ghost_dist  : inverse distance to nearest edible ghost on ray
    [5] visit_saturation   : fraction of open tiles along this ray already
                             visited (0=fully unexplored, 1=fully explored),
                             mapped to [-1,1].  Gives the agent a PERCEPTIBLE,
                             directly actionable novelty signal per direction.

  NOTE on ghost channels: the original implementation used a single SIGNED
  channel (positive=lethal, negative=edible). This conflated distance
  magnitude with threat semantics, making the signal ambiguous — the same
  value could mean a close edible ghost or a far lethal ghost. Separating
  the channels removes this ambiguity and simplifies the learning problem for
  both DQN and NEAT.

─────────────────────────────────────────────────────────────────
  Block B — BFS global features (3 scalars, indices 24-26)
  ──────────────────────────────────────────────────────────────
  [24] near_food    : BFS shortest-path distance to nearest pellet
  [25] near_danger  : BFS distance to nearest lethal ghost
  [26] near_edible  : BFS distance to nearest frightened ghost
  BFS respects maze topology (no wall hallucination); all mapped to [-1,1].

─────────────────────────────────────────────────────────────────
  Block C — Power state (2 scalars, indices 27-28)
  ──────────────────────────────────────────────────────────────
  [27] is_powered          : {-1, +1} binary frightened-mode indicator
  [28] power_time_remaining: normalised remaining frightened duration

─────────────────────────────────────────────────────────────────
Total: 24 + 3 + 2 = 29 floats.

─────────────────────────────────────────────────────────────────
Normalisation: ALL channels remapped from [0,1] to [-1,1] via 2x-1.
This zero-centred representation is necessary for:
  • DQN   — prevents biased activations, reduces dead-ReLU risk under Adam.
  • NEAT  — ensures evolved networks receive consistent input scales,
            preventing individual features from dominating purely by magnitude.
─────────────────────────────────────────────────────────────────

Action Space: Discrete(4) — egocentric relative actions
    0: FORWARD, 1: LEFT, 2: RIGHT, 3: BACKWARD

Each env.step() fast-forwards physics via an internal while-loop and only
returns when Pac-Man reaches the centre of a new tile (next decision point)
or the episode terminates / truncates. Rewards are accumulated across all
internal ticks.
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
from Code.Engine.GameEngine import GameEngine, GameState
from Code.Engine.Ghost import GhostState


def _load_settings(json_path: str | dict | None = None) -> dict:
    if isinstance(json_path, dict):
        return json_path
    # Resolve settings path defensively so scripts launched from different
    # working directories still load the intended Code/game_settings.json.
    if json_path is not None:
        candidates = [json_path]
        if not os.path.isabs(json_path):
            candidates.extend([
                os.path.join(_HERE, json_path),
                os.path.join(_ROOT, json_path),
                os.path.join(os.getcwd(), json_path),
            ])
    else:
        candidates = [
            os.path.join(_HERE, "game_settings.json"),
            os.path.join(_ROOT, "game_settings.json"),
            os.path.join(os.getcwd(), "game_settings.json"),
        ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return Settings(candidate).get_all()

    # Fall back to the Code-level default path for a clear failure message.
    fallback = os.path.join(_ROOT, "game_settings.json")
    return Settings(fallback).get_all()


class PacManEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # ── Observation layout constants ──────────────────────────────────────────
    RAY_CHANNELS   = 6   # wall, food, power, lethal_ghost, edible_ghost, visit_sat
    N_RAY_DIRS     = 4   # cardinal directions only (UP, DOWN, LEFT, RIGHT)
    _RAY_BLOCK_END = N_RAY_DIRS * RAY_CHANNELS  # 24

    # BFS global features (indices 24-26)
    NEAR_FOOD_IDX    = _RAY_BLOCK_END       # 24  — used for food-shaping reward
    NEAR_DANGER_IDX  = _RAY_BLOCK_END + 1   # 25
    NEAR_EDIBLE_IDX  = _RAY_BLOCK_END + 2   # 26

    # Power state (indices 27-28)
    IS_POWERED_IDX   = _RAY_BLOCK_END + 3   # 27
    POWER_REM_IDX    = _RAY_BLOCK_END + 4   # 28

    OBS_SIZE = 29

    # Alias for food-shaping reward computation
    FOOD_DIST_OBS_IDX = NEAR_FOOD_IDX       # 24

    # Egocentric actions
    FORWARD  = 0
    LEFT     = 1
    RIGHT    = 2
    BACKWARD = 3

    # Cardinal directions
    UP      = 0
    DOWN    = 1
    LEFT_C  = 2
    RIGHT_C = 3

    _CARDINAL_TO_VEC = {UP: (0, -1), DOWN: (0, 1), LEFT_C: (-1, 0), RIGHT_C: (1, 0)}
    _CARDINAL_OPPOSITE = {UP: DOWN, DOWN: UP, LEFT_C: RIGHT_C, RIGHT_C: LEFT_C}

    _GLOBAL_RAY_DIRS = [
        (0, -1),   # UP
        (0, 1),    # DOWN
        (-1, 0),   # LEFT
        (1, 0),    # RIGHT
    ]

    def __init__(
        self,
        render_mode: str | None = None,
        obs_type: str = "vector",
        settings: dict | None = None,
        settings_path: str | None = None,
        max_episode_steps: int | None = 10000,
        maze_seed: int | None = None,
        **engine_kwargs,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.obs_type    = obs_type
        self.maze_seed   = maze_seed
        self._step_count = 0

        self._base_cfg = _load_settings(settings if settings else settings_path)
        self._base_cfg.update(engine_kwargs)

        raw_max_steps = self._base_cfg.get("max_episode_steps", max_episode_steps)
        if raw_max_steps is None or float(raw_max_steps) <= 0:
            self.max_episode_steps = None
        else:
            self.max_episode_steps = int(raw_max_steps)

        self._base_cfg.pop("tile_center_tolerance", None)

        raw_obs_horizon = self._base_cfg.pop("obs_distance_horizon", 20)
        self.obs_distance_horizon = max(1, int(raw_obs_horizon))

        raw_starvation = self._base_cfg.get("starvation_limit_ticks", 30 * 60)
        self._starvation_limit_default = max(1, int(raw_starvation))
        self.starvation_limit = self._starvation_limit_default
        self._ticks_since_food = 0

        self._pygame_initialised = False
        self._screen = None
        self._clock  = None
        self.engine  = None

        self.action_space = spaces.Discrete(4)

        # All 29 channels are in [-1, 1] after normalisation.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float32
        )

        self._last_action       = None
        self._last_cardinal_dir = self.UP

        self.pellets_eaten_this_episode       = 0
        self.ghosts_eaten_this_episode        = 0
        self.power_pellets_eaten_this_episode = 0

        self._visited_tiles:       set[tuple[int, int]]            = set()
        self._visit_counts:        dict[tuple[int, int], int]       = {}
        self._bfs_cache:           dict[tuple[int, int], dict[tuple[int, int], int]] = {}
        self._total_explorable_tiles: int = 0
        self._max_lives:           int = 3

        self.current_stage   = None
        self.current_epsilon = None

    # =========================================================================
    # Reset
    # =========================================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_pygame()

        cfg = self._base_cfg.copy()
        if seed is not None:
            cfg["maze_seed"] = seed
            np.random.seed(seed)

        raw_starvation = cfg.get("starvation_limit_ticks", self._starvation_limit_default)
        self.starvation_limit = max(1, int(raw_starvation))

        self._max_lives = max(1, int(cfg.get("lives", 3)))

        self.engine = GameEngine(**cfg)
        self.engine.game_state = GameState.GAME
        self.engine.paused     = False

        self._step_count  = 0
        self._last_action = None

        dx, dy = self.engine.pacman.direction
        self._last_cardinal_dir = (
            self.RIGHT_C if (dx, dy) == (0, 0)
            else self._get_pacman_heading_cardinal()
        )

        self._ticks_since_food = 0

        self.pellets_eaten_this_episode       = 0
        self.ghosts_eaten_this_episode        = 0
        self.power_pellets_eaten_this_episode = 0

        self._visited_tiles.clear()
        self._visit_counts.clear()
        # BFS cache is valid for the entire episode — maze topology is static.
        self._bfs_cache              = {}
        self._total_explorable_tiles = 0

        maze = self.engine.maze
        for y in range(maze.height):
            for x in range(maze.width):
                if maze.maze[y][x] == 0:
                    self._total_explorable_tiles += 1

        return self._get_obs(), {}

    # =========================================================================
    # Heading helpers
    # =========================================================================
    def _get_pacman_heading_cardinal(self) -> int:
        dx, dy = self.engine.pacman.direction
        if (dx, dy) == (0, 0):
            return self._last_cardinal_dir
        if dx == 0 and dy < 0: return self.UP
        if dx == 0 and dy > 0: return self.DOWN
        if dx < 0 and dy == 0: return self.LEFT_C
        return self.RIGHT_C

    def _get_obs(self) -> np.ndarray:
        return self._get_vector_obs()

    def _valid_cardinal_dirs(self, tile_x: int, tile_y: int) -> list[int]:
        eng   = self.engine
        valid = []
        for cdir, (dx, dy) in self._CARDINAL_TO_VEC.items():
            nx, ny = tile_x + dx, tile_y + dy
            if (
                0 <= nx < eng.maze.width
                and 0 <= ny < eng.maze.height
                and eng.maze.maze[ny][nx] == 0
            ):
                valid.append(cdir)
        return valid

    def get_valid_actions(self) -> list[int]:
        """Return physically valid egocentric actions from the current tile/heading."""
        if self.engine is None:
            return [self.FORWARD, self.LEFT, self.RIGHT, self.BACKWARD]

        ts = self.engine.tile_size
        px = self.engine.pacman.x + self.engine.pacman.size // 2
        py = self.engine.pacman.y + self.engine.pacman.size // 2
        tx, ty = int(px // ts), int(py // ts)

        heading    = self._get_pacman_heading_cardinal()
        valid_dirs = self._valid_cardinal_dirs(tx, ty)

        left_map     = {self.UP: self.LEFT_C,  self.DOWN: self.RIGHT_C, self.LEFT_C: self.DOWN,    self.RIGHT_C: self.UP}
        right_map    = {self.UP: self.RIGHT_C, self.DOWN: self.LEFT_C,  self.LEFT_C: self.UP,      self.RIGHT_C: self.DOWN}
        backward_map = {self.UP: self.DOWN,    self.DOWN: self.UP,      self.LEFT_C: self.RIGHT_C, self.RIGHT_C: self.LEFT_C}

        action_to_dir = {
            self.FORWARD:  heading,
            self.LEFT:     left_map[heading],
            self.RIGHT:    right_map[heading],
            self.BACKWARD: backward_map[heading],
        }

        actions = [a for a, target in action_to_dir.items() if target in valid_dirs]
        return actions if actions else [self.FORWARD, self.LEFT, self.RIGHT, self.BACKWARD]

    # =========================================================================
    # BFS
    # =========================================================================
    def _bfs_shortest_path_distances(
        self, start_tx: int, start_ty: int
    ) -> dict[tuple[int, int], int]:
        """Shortest-path distances from Pac-Man's tile to all reachable tiles.

        Result is cached per (tx, ty) for the entire episode.  Maze topology
        is static within an episode (only pellet *contents* change, not which
        tiles are walkable), so cached distances never need clearing mid-episode.
        """
        eng  = self.engine
        maze = eng.maze

        if not (0 <= start_tx < maze.width and 0 <= start_ty < maze.height):
            return {}

        distances: dict[tuple[int, int], int] = {(start_tx, start_ty): 0}
        queue = deque([(start_tx, start_ty)])

        while queue:
            cx, cy   = queue.popleft()
            base_dist = distances[(cx, cy)]
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < maze.width and 0 <= ny < maze.height):
                    continue
                if (nx, ny) in distances:
                    continue
                if maze.maze[ny][nx] != 0:
                    continue
                distances[(nx, ny)] = base_dist + 1
                queue.append((nx, ny))

        return distances

    # =========================================================================
    # Observation construction
    # =========================================================================
    @staticmethod
    def _to_norm(x: float) -> float:
        """Map [0, 1] → [-1, 1] via linear transform 2x − 1.

        Zero-centred representation required for both DQN (prevents biased
        activations under ReLU/Adam) and NEAT (consistent input scales across
        the evolving population).
        """
        return 2.0 * x - 1.0

    def _distance_to_signal(self, distance: int) -> float:
        """Inverse-distance encoding 1/(1+d), clipped at obs_distance_horizon."""
        d = min(int(distance), self.obs_distance_horizon)
        return 1.0 / (1.0 + float(d))

    def _get_vector_obs(self) -> np.ndarray:
        eng = self.engine
        ts  = eng.tile_size

        px = eng.pacman.x + ts / 2.0
        py = eng.pacman.y + ts / 2.0
        tx = int(px // ts)
        ty = int(py // ts)

        # ── Entity tile sets ──────────────────────────────────────────────────
        food_tiles  = {(int(x // ts), int(y // ts)) for x, y in eng.pellets}
        power_tiles = {(int(x // ts), int(y // ts)) for x, y in eng.power_pellets}

        lethal_ghost_tiles = set()
        edible_ghost_tiles = set()
        for g in eng.ghosts:
            gx = int((g.x + ts / 2.0) // ts)
            gy = int((g.y + ts / 2.0) // ts)
            if g.state == GhostState.FRIGHTENED:
                edible_ghost_tiles.add((gx, gy))
            elif g.state != GhostState.EATEN:
                lethal_ghost_tiles.add((gx, gy))

        # ── Block A: raycasts (6 channels per direction) ──────────────────────
        # Channel order: wall | food | power | lethal_ghost | edible_ghost | visit_sat
        #
        # visit_saturation: fraction of open tiles along this ray that Pac-Man
        # has already visited.  Gives the agent a perceptually grounded,
        # directionally specific novelty signal it can act on.  Without this,
        # any exploration reward would be invisible to the policy.
        global_rays: list[float] = []

        for dx, dy in self._GLOBAL_RAY_DIRS:
            d    = 0
            w_d  = 0.0
            f_d  = 0.0
            p_d  = 0.0
            lg_d = 0.0
            eg_d = 0.0
            ray_open    = 0   # non-wall tiles along this ray
            ray_visited = 0   # of those, how many have been visited

            cx, cy = tx, ty
            while True:
                cx += dx
                cy += dy
                d  += 1

                if (
                    not (0 <= cx < eng.maze.width and 0 <= cy < eng.maze.height)
                    or eng.maze.maze[cy][cx] == 1
                ):
                    w_d = self._distance_to_signal(d)
                    break

                ray_open += 1
                if (cx, cy) in self._visited_tiles:
                    ray_visited += 1

                if f_d  == 0.0 and (cx, cy) in food_tiles:
                    f_d  = self._distance_to_signal(d)
                if p_d  == 0.0 and (cx, cy) in power_tiles:
                    p_d  = self._distance_to_signal(d)
                if lg_d == 0.0 and (cx, cy) in lethal_ghost_tiles:
                    lg_d = self._distance_to_signal(d)
                if eg_d == 0.0 and (cx, cy) in edible_ghost_tiles:
                    eg_d = self._distance_to_signal(d)

            visit_sat = ray_visited / max(1, ray_open) if ray_open > 0 else 0.0

            # All channels mapped to [-1, 1].
            global_rays.extend([
                self._to_norm(w_d),
                self._to_norm(f_d),
                self._to_norm(p_d),
                self._to_norm(lg_d),
                self._to_norm(eg_d),
                self._to_norm(visit_sat),
            ])

        # ── Egocentric reordering ─────────────────────────────────────────────
        heading = self._get_pacman_heading_cardinal()
        heading_mapping = {
            self.UP:      [0, 1, 2, 3],
            self.DOWN:    [1, 0, 3, 2],
            self.LEFT_C:  [2, 3, 1, 0],
            self.RIGHT_C: [3, 2, 0, 1],
        }
        order = heading_mapping.get(heading, heading_mapping[self.UP])

        obs: list[float] = []
        for idx in order:
            base = idx * self.RAY_CHANNELS
            obs.extend(global_rays[base: base + self.RAY_CHANNELS])

        # ── Block B: BFS nearest-entity features ──────────────────────────────
        # Topology-aware distances; cached per start-tile for the episode.
        if not hasattr(self, "_bfs_cache"):
            self._bfs_cache = {}

        key = (tx, ty)
        if key not in self._bfs_cache:
            self._bfs_cache[key] = self._bfs_shortest_path_distances(tx, ty)

        sp_dist = self._bfs_cache[key]

        def inv_nearest(target_tiles: set) -> float:
            if not target_tiles:
                return 0.0
            best = float("inf")
            for tile in target_tiles:
                d = sp_dist.get(tile)
                if d is not None and d < best:
                    best = d
            return 0.0 if best == float("inf") else self._distance_to_signal(int(best))

        obs.append(self._to_norm(inv_nearest(food_tiles)))          # idx 24
        obs.append(self._to_norm(inv_nearest(lethal_ghost_tiles)))  # idx 25
        obs.append(self._to_norm(inv_nearest(edible_ghost_tiles)))  # idx 26

        # ── Block C: Power state ──────────────────────────────────────────────
        is_powered     = 1.0 if eng.frightened_mode else 0.0
        power_remaining = 0.0
        if eng.frightened_mode and getattr(eng, "frightened_duration", 0) > 0:
            remain = max(0, eng.frightened_duration - eng.frightened_timer)
            power_remaining = max(0.0, min(1.0, float(remain) / float(eng.frightened_duration)))

        obs.append(self._to_norm(is_powered))       # idx 27
        obs.append(self._to_norm(power_remaining))  # idx 28

        return np.array(obs, dtype=np.float32)

    # =========================================================================
    # Step
    # =========================================================================
    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        eng = self.engine
        ts  = eng.tile_size

        heading        = self._get_pacman_heading_cardinal()
        valid_actions  = self.get_valid_actions()
        blocked_action = action not in valid_actions

        left_map     = {self.UP: self.LEFT_C,  self.DOWN: self.RIGHT_C, self.LEFT_C: self.DOWN,    self.RIGHT_C: self.UP}
        right_map    = {self.UP: self.RIGHT_C, self.DOWN: self.LEFT_C,  self.LEFT_C: self.UP,      self.RIGHT_C: self.DOWN}
        backward_map = {self.UP: self.DOWN,    self.DOWN: self.UP,      self.LEFT_C: self.RIGHT_C, self.RIGHT_C: self.LEFT_C}

        if action == self.FORWARD:      target_dir = heading
        elif action == self.LEFT:       target_dir = left_map[heading]
        elif action == self.RIGHT:      target_dir = right_map[heading]
        else:                           target_dir = backward_map[heading]

        # ── Blocked action path ───────────────────────────────────────────────
        if blocked_action:
            self._last_action       = int(action)
            self._step_count       += 1
            self._ticks_since_food += 1

            accumulated_reward = -5.0
            reward_breakdown = {
                "pellet_reward":          0.0,
                "power_reward":           0.0,
                "ghost_reward":           0.0,
                "win_reward":             0.0,
                "death_penalty":          0.0,
                "starvation_penalty":     0.0,
                "food_shaping_reward":    0.0,
                "invalid_action_penalty": -5.0,
                "living_penalty":         0.0,
                "total":                  0.0,
            }

            starved = self._ticks_since_food >= self.starvation_limit
            if starved:
                accumulated_reward                 -= 100.0
                reward_breakdown["starvation_penalty"] -= 100.0

            accumulated_reward              -= 0.5
            reward_breakdown["living_penalty"] -= 0.5

            terminated = bool(starved)
            truncated  = (
                self.max_episode_steps is not None
                and self._step_count >= self.max_episode_steps
            )

            px = eng.pacman.x + eng.pacman.size // 2
            py = eng.pacman.y + eng.pacman.size // 2
            cur_tx    = int(px // ts)
            cur_ty    = int(py // ts)
            tile_key  = (cur_tx, cur_ty)
            visit_count = self._visit_counts.get(tile_key, 0) + 1
            self._visit_counts[tile_key] = visit_count
            self._visited_tiles.add(tile_key)

            reward_breakdown["total"] = float(accumulated_reward)
            death_cause = "STARVATION" if starved else ("MAX_STEPS" if truncated else "NONE")

            info = self._get_info()
            info.update({
                "death_cause":      death_cause,
                "internal_ticks":   0,
                "tile_center":      (cur_tx, cur_ty),
                "center_lock_mode": "exact",
                "steps":            int(self._step_count),
                "action":           int(action),
                "target_dir":       int(target_dir),
                "visit_count":      int(visit_count),
                "blocked_action":   True,
                "reward_breakdown": reward_breakdown,
            })

            if self.render_mode == "human":
                self._render_human()
            return self._get_obs(), accumulated_reward, terminated, truncated, info

        # ── Valid action path ─────────────────────────────────────────────────
        self._last_action       = int(action)
        self._last_cardinal_dir = target_dir
        eng.pacman.next_direction = self._CARDINAL_TO_VEC[target_dir]

        accumulated_reward = 0.0
        reward_breakdown = {
            "pellet_reward":       0.0,
            "power_reward":        0.0,
            "ghost_reward":        0.0,
            "win_reward":          0.0,
            "death_penalty":       0.0,
            "starvation_penalty":  0.0,
            "food_shaping_reward": 0.0,
            "living_penalty":      0.0,
            "total":               0.0,
        }

        # Food-shaping: compare BFS nearest-food signal before and after the step.
        # Uses the named index constant — no magic numbers.
        initial_obs      = self._get_obs()
        dist_food_before = float(initial_obs[self.FOOD_DIST_OBS_IDX])

        px_start = eng.pacman.x + eng.pacman.size // 2
        py_start = eng.pacman.y + eng.pacman.size // 2
        start_tx = int(px_start // ts)
        start_ty = int(py_start // ts)

        # ── Tile-lock loop ────────────────────────────────────────────────────
        internal_ticks         = 0
        no_progress_ticks      = 0
        max_no_progress_ticks  = max(2, ts // max(1, eng.pacman.speed))
        pellet_eaten_in_step   = False
        starved    = False
        terminated = False
        truncated  = False

        while True:
            pre_lives   = eng.lives
            pre_won     = eng.won
            pre_pellets = len(eng.pellets)
            pre_power   = len(eng.power_pellets)
            pre_eaten   = sum(1 for g in eng.ghosts if g.state == GhostState.EATEN)

            pre_pac_x     = eng.pacman.x
            pre_pac_y     = eng.pacman.y
            pre_px_center = pre_pac_x + eng.pacman.size // 2
            pre_py_center = pre_pac_y + eng.pacman.size // 2

            eng.update()
            internal_ticks        += 1
            self._step_count      += 1
            self._ticks_since_food += 1

            reward_tick = 0.0

            # Pellets — BFS distances cache NOT cleared: maze walkability is
            # static within an episode; only pellet contents change, not tiles.
            pellets_eaten = max(0, pre_pellets - len(eng.pellets))
            if pellets_eaten > 0:
                pellet_eaten_in_step = True
                pellet_gain = 10.0 * pellets_eaten
                reward_tick += pellet_gain
                reward_breakdown["pellet_reward"] += pellet_gain
                self._ticks_since_food = 0
                self.pellets_eaten_this_episode += pellets_eaten

            power_eaten = max(0, pre_power - len(eng.power_pellets))
            if power_eaten > 0:
                power_gain = 50.0 * power_eaten
                reward_tick += power_gain
                reward_breakdown["power_reward"] += power_gain
                self._ticks_since_food = 0
                self.power_pellets_eaten_this_episode += power_eaten

            ghosts_eaten = max(
                0,
                sum(1 for g in eng.ghosts if g.state == GhostState.EATEN) - pre_eaten,
            )
            if ghosts_eaten > 0:
                ghost_gain = 200.0 * ghosts_eaten
                reward_tick += ghost_gain
                reward_breakdown["ghost_reward"] += ghost_gain
                self.ghosts_eaten_this_episode += ghosts_eaten

            if eng.won and not pre_won:
                reward_tick += 1000.0
                reward_breakdown["win_reward"] += 1000.0

            if eng.lives < pre_lives:
                reward_tick -= 500.0
                reward_breakdown["death_penalty"] -= 500.0

            starved = self._ticks_since_food >= self.starvation_limit
            if starved:
                reward_tick -= 100.0
                reward_breakdown["starvation_penalty"] -= 100.0

            accumulated_reward += reward_tick

            terminated = eng.game_over or eng.won or starved
            truncated  = (
                self.max_episode_steps is not None
                and self._step_count >= self.max_episode_steps
            )

            if terminated or truncated:
                break

            # Break when Pac-Man reaches exact centre of new tile.
            px = eng.pacman.x + eng.pacman.size // 2
            py = eng.pacman.y + eng.pacman.size // 2
            cur_tx = int(px // ts)
            cur_ty = int(py // ts)

            if (cur_tx != start_tx) or (cur_ty != start_ty):
                center_x = (cur_tx * ts) + (ts // 2)
                center_y = (cur_ty * ts) + (ts // 2)

                on_center     = (px == center_x and py == center_y)
                crossed_center = (
                    (pre_px_center - center_x) * (px - center_x) <= 0
                    and (pre_py_center - center_y) * (py - center_y) <= 0
                )

                if on_center or crossed_center:
                    eng.pacman.x = center_x - (eng.pacman.size // 2)
                    eng.pacman.y = center_y - (eng.pacman.size // 2)
                    break

            if eng.pacman.x == pre_pac_x and eng.pacman.y == pre_pac_y:
                no_progress_ticks += 1
            else:
                no_progress_ticks = 0

            if no_progress_ticks >= max_no_progress_ticks:
                eng.pacman.x = (cur_tx * ts) + (ts // 2) - (eng.pacman.size // 2)
                eng.pacman.y = (cur_ty * ts) + (ts // 2) - (eng.pacman.size // 2)
                break

            if eng.pacman.direction == (0, 0):
                eng.pacman.x = (cur_tx * ts) + (ts // 2) - (eng.pacman.size // 2)
                eng.pacman.y = (cur_ty * ts) + (ts // 2) - (eng.pacman.size // 2)
                break

        # ── Post-loop rewards ─────────────────────────────────────────────────
        px = eng.pacman.x + eng.pacman.size // 2
        py = eng.pacman.y + eng.pacman.size // 2
        cur_tx = int(px // ts)
        cur_ty = int(py // ts)

        tile_key    = (cur_tx, cur_ty)
        visit_count = self._visit_counts.get(tile_key, 0) + 1
        self._visit_counts[tile_key] = visit_count
        self._visited_tiles.add(tile_key)

        # Food-approach shaping: rewards movement toward nearest pellet.
        # Exploration context is provided locally via visit_saturation per ray.
        if not pellet_eaten_in_step:
            new_obs          = self._get_obs()
            dist_food_after  = float(new_obs[self.FOOD_DIST_OBS_IDX])
            shaping_reward   = (dist_food_after - dist_food_before) * 5.0
            accumulated_reward += shaping_reward
            reward_breakdown["food_shaping_reward"] += shaping_reward

        accumulated_reward              -= 0.5
        reward_breakdown["living_penalty"] -= 0.5
        reward_breakdown["total"] = float(accumulated_reward)

        if starved:           death_cause = "STARVATION"
        elif eng.won:         death_cause = "WIN"
        elif eng.game_over:   death_cause = "GHOST"
        elif truncated:       death_cause = "MAX_STEPS"
        else:                 death_cause = "NONE"

        info = self._get_info()
        info.update({
            "death_cause":      death_cause,
            "internal_ticks":   internal_ticks,
            "tile_center":      (cur_tx, cur_ty),
            "center_lock_mode": "exact",
            "steps":            int(self._step_count),
            "action":           int(action),
            "target_dir":       int(target_dir),
            "visit_count":      int(visit_count),
            "blocked_action":   False,
            "reward_breakdown": reward_breakdown,
        })

        if self.render_mode == "human":
            self._render_human()
        return self._get_obs(), accumulated_reward, terminated, truncated, info

    # =========================================================================
    # Info
    # =========================================================================
    def _get_info(self) -> dict:
        pellets_remaining = len(self.engine.pellets) + len(self.engine.power_pellets)
        explore_rate = (
            len(self._visited_tiles) / self._total_explorable_tiles
            if self._total_explorable_tiles > 0 else 0.0
        )
        map_clear_pct = explore_rate * 100.0
        return {
            "score":                  self.engine.pacman.score,
            "frightened":             self.engine.frightened_mode,
            "stage":                  self.current_stage,
            "epsilon":                self.current_epsilon,
            "pellets":                self.pellets_eaten_this_episode,
            "ghosts":                 self.ghosts_eaten_this_episode,
            "power_pellets":          self.power_pellets_eaten_this_episode,
            "maze_seed":              getattr(self.engine, "maze_seed", None),
            "explored_tiles":         len(self._visited_tiles),
            "total_explorable_tiles": self._total_explorable_tiles,
            "explore_rate":           explore_rate,
            "pellets_remaining":      pellets_remaining,
            "map_clear_pct":          map_clear_pct,
        }

    # =========================================================================
    # Rendering
    # =========================================================================
    def _ensure_pygame(self):
        if self._pygame_initialised:
            return
        if self.render_mode is None:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        if self.render_mode == "human":
            res = self._base_cfg.get("window_resolution", "800x800")
            w, h = map(int, res.split("x"))
            self._screen = pygame.display.set_mode((w, h))
            self._clock  = pygame.time.Clock()
        self._pygame_initialised = True

    def _draw_debug_sensors(self):
        return

    def render(self):
        if self.render_mode == "rgb_array":
            self._ensure_pygame()
            width  = self.engine.maze.width  * self.engine.tile_size
            height = self.engine.maze.height * self.engine.tile_size
            surf   = pygame.Surface((width, height))
            surf.fill((0, 0, 0))
            self.engine.draw(surf)
            return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
        elif self.render_mode == "human":
            self._render_human()
            return None

    def _render_human(self):
        if not self._screen:
            return
        pygame.event.pump()
        self._screen.fill((0, 0, 0))
        self.engine.draw(self._screen)
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if not self._pygame_initialised:
            return

        # Important for multi-env visual runs: headless/rgb_array envs should not
        # shut down pygame globally, otherwise the training display surface is lost.
        if self.render_mode == "human":
            pygame.quit()

        self._screen = None
        self._clock = None
        self._pygame_initialised = False

    @staticmethod
    def _parse_res(res_str):
        w, h = res_str.split("x")
        return int(w), int(h)
