"""
PacManEnv.py
============
Consolidated Egocentric Version.
Observation: 27 floats (8x3 Egocentric Raycasts, 2x Relative Compass, 1x Fright State).
Action Space: Discrete(3) -> 0: FORWARD, 1: LEFT, 2: RIGHT.
"""

import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

# Sibling module imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.Settings import Settings
from Code.GameEngine import GameEngine, GameState
from Code.Ghost import GhostState

# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_settings(json_path: str | dict | None = None) -> dict:
    if isinstance(json_path, dict):
        return json_path
    if json_path is None:
        json_path = os.path.join(_HERE, "game_settings.json")
    s = Settings(json_path)
    return s.get_all()

# ── Environment ───────────────────────────────────────────────────────────────

class PacManEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    # Egocentric actions
    FORWARD = 0
    LEFT    = 1
    RIGHT   = 2

    # Global cardinal directions for engine communication
    UP      = 0
    DOWN    = 1
    LEFT_C  = 2
    RIGHT_C = 3

    _CARDINAL_TO_VEC = {
        UP:      (0, -1),
        DOWN:    (0,  1),
        LEFT_C:  (-1, 0),
        RIGHT_C: (1,  0),
    }

    _CARDINAL_OPPOSITE = {
        UP: DOWN,
        DOWN: UP,
        LEFT_C: RIGHT_C,
        RIGHT_C: LEFT_C,
    }

    _GLOBAL_RAY_DIRS = [
        (0, -1), (0, 1), (-1, 0), (1, 0),      # N, S, W, E
        (-1, -1), (1, -1), (-1, 1), (1, 1)     # NW, NE, SW, SE
    ]

    def __init__(
            self,
            render_mode: str | None = None,
            obs_type: str = "vector",
            settings: dict | None = None,
            settings_path: str | None = None,
            max_episode_steps: int = 10000,
            maze_seed: int | None = None,
            **engine_kwargs,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.maze_seed = maze_seed
        self._step_count = 0

        self._base_cfg = _load_settings(settings if settings else settings_path)
        self._base_cfg.update(engine_kwargs)

        self.step_penalty     = 0.0 # Intentionally zeroed for biological starvation logic
        self.reversal_penalty = self._base_cfg.get("reversal_penalty", -0.2)
        self.max_episode_steps = self._base_cfg.get("max_episode_steps", max_episode_steps)

        self.starvation_limit = 20 * 60
        self._ticks_since_food = 0

        self._pygame_initialised = False
        self._screen = None
        self._clock = None
        self.engine = None
        self._engine = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(27,), dtype=np.float32)

        self._prev_score = 0
        self._prev_lives = 3
        self._won_already = False
        self._levels_completed = 0
        self._last_action = None
        self._last_cardinal_dir = self.UP

        self._recent_tiles = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_pygame()

        cfg = self._base_cfg.copy()
        if seed is not None:
            cfg["maze_seed"] = seed

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
        valid_cardinals = self._get_valid_cardinal_actions()
        heading = self._get_pacman_heading_cardinal()

        if action == self.FORWARD:
            target_dir = heading
        elif action == self.LEFT:
            target_dir = {self.UP: self.LEFT_C, self.DOWN: self.RIGHT_C, self.LEFT_C: self.DOWN, self.RIGHT_C: self.UP}[heading]
        else:  # RIGHT
            target_dir = {self.UP: self.RIGHT_C, self.DOWN: self.LEFT_C, self.LEFT_C: self.UP, self.RIGHT_C: self.DOWN}[heading]

        wall_penalty = 0.0
        if target_dir not in valid_cardinals:
            wall_penalty = -0.5
            target_dir = heading if heading in valid_cardinals else (valid_cardinals[0] if valid_cardinals else heading)

        reversal_reward = 0.0
        if target_dir == self._CARDINAL_OPPOSITE.get(self._last_cardinal_dir):
            reversal_reward = self.reversal_penalty

        self._last_cardinal_dir = target_dir
        self.engine.pacman.next_direction = self._CARDINAL_TO_VEC[target_dir]
        self._last_action = action

        accumulated_reward = 0.0
        ts = self.engine.tile_size

        while True:
            reward_this_tick = self.step_penalty + reversal_reward + wall_penalty
            reversal_reward = 0.0
            wall_penalty = 0.0

            self.engine.update()
            self._step_count += 1
            self._ticks_since_food += 1

            score_delta = self.engine.pacman.score - self._prev_score
            if score_delta > 0:
                reward_this_tick += float(score_delta)
                self._ticks_since_food = 0

            self._prev_score = self.engine.pacman.score

            if self.engine.lives < self._prev_lives:
                reward_this_tick -= 50.0
                self._prev_lives = self.engine.lives

            if self.engine.won and not self._won_already:
                reward_this_tick += 100.0
                self._won_already = True
                self._levels_completed += 1

            accumulated_reward += reward_this_tick

            starved = (self._ticks_since_food >= self.starvation_limit)
            terminated = self.engine.game_over or self.engine.won or starved
            truncated = False

            if terminated:
                if starved:
                    print(
                        f"\n[DEATH] Agent STARVED after {self._ticks_since_food} ticks without food (Step: {self._step_count}).")
                    accumulated_reward -= 50.0
                elif self.engine.game_over:
                    accumulated_reward -= 50.0
                break

            if (self.engine.pacman.x % ts == 0) and (self.engine.pacman.y % ts == 0):
                if not hasattr(self, '_recent_tiles'): self._recent_tiles = []
                tx, ty = int(self.engine.pacman.x / ts), int(self.engine.pacman.y / ts)
                self._recent_tiles.append((tx, ty))
                if len(self._recent_tiles) > 15: self._recent_tiles.pop(0)
                if self._recent_tiles.count((tx, ty)) >= 4: accumulated_reward -= 1.0
                break

        if self.render_mode == "human": self._render_human()
        return self._get_obs(), accumulated_reward, terminated, truncated, self._get_info()

    # ── Observation Logic ─────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return self._get_vector_obs()

    def _get_pacman_heading_cardinal(self) -> int:
        dx, dy = self.engine.pacman.direction
        if (dx, dy) == (0, 0): return self._last_cardinal_dir
        if dx == 0 and dy < 0: return self.UP
        if dx == 0 and dy > 0: return self.DOWN
        if dx < 0 and dy == 0: return self.LEFT_C
        return self.RIGHT_C

    def _get_vector_obs(self) -> np.ndarray:
        eng = self.engine
        ts = eng.tile_size
        px, py = eng.pacman.x + ts/2, eng.pacman.y + ts/2
        tx, ty = int(px/ts), int(py/ts)

        food = set((int(x/ts), int(y/ts)) for x, y in eng.pellets + eng.power_pellets)
        lethal = set((int((g.x+ts/2)/ts), int((g.y+ts/2)/ts)) for g in eng.ghosts if g.state in (GhostState.CHASE, GhostState.SCATTER))
        edible = set((int((g.x+ts/2)/ts), int((g.y+ts/2)/ts)) for g in eng.ghosts if g.state == GhostState.FRIGHTENED)

        # 1. Global Raycasts
        global_rays = []
        for dx, dy in self._GLOBAL_RAY_DIRS:
            d, w_d, f_d, g_d = 0, 0.0, 0.0, 0.0
            cx, cy = tx, ty
            while True:
                cx += dx; cy += dy; d += 1
                if not (0 <= cx < eng.maze.width and 0 <= cy < eng.maze.height) or eng.maze.maze[cy][cx] == 1:
                    w_d = 1.0/d; break
                if f_d == 0.0 and (cx, cy) in food: f_d = 1.0/d
                if g_d == 0.0:
                    if (cx, cy) in lethal: g_d = 1.0/d
                    elif (cx, cy) in edible: g_d = -1.0/d
            global_rays.extend([w_d, f_d, g_d])

        # 2. Ego-Centric Rotation
        heading = self._get_pacman_heading_cardinal()
        mapping = {
            self.UP:      [0, 1, 2, 3, 4, 5, 6, 7],
            self.DOWN:    [1, 0, 3, 2, 7, 6, 5, 4],
            self.LEFT_C:  [2, 3, 1, 0, 6, 4, 7, 5],
            self.RIGHT_C: [3, 2, 0, 1, 5, 7, 4, 6]
        }
        order = mapping.get(heading, mapping[self.UP])
        obs = []
        for idx in order:
            obs.extend(global_rays[idx*3 : idx*3 + 3])

        # 3. Relative Compass
        if eng.pellets or eng.power_pellets:
            all_f = eng.pellets + eng.power_pellets
            best = sorted([( (f[0]-px)**2 + (f[1]-py)**2, f[0]-px, f[1]-py ) for f in all_f])[0]
            dx, dy = best[1], best[2]
            if heading == self.UP:      rx, ry = dx, dy
            elif heading == self.DOWN:  rx, ry = -dx, -dy
            elif heading == self.LEFT_C: rx, ry = dy, -dx
            else:                       rx, ry = -dy, dx
            obs.extend([rx/(eng.maze.width*ts), ry/(eng.maze.height*ts)])
        else:
            obs.extend([0.0, 0.0])

        obs.append(1.0 if eng.frightened_mode else 0.0)
        return np.array(obs, dtype=np.float32)

    def _get_valid_cardinal_actions(self) -> list[int]:
        eng, ts = self.engine, self.engine.tile_size
        tx, ty = int((eng.pacman.x + ts/2)/ts), int((eng.pacman.y + ts/2)/ts)
        valid = []
        for card, (dx, dy) in self._CARDINAL_TO_VEC.items():
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < eng.maze.width and 0 <= ny < eng.maze.height:
                if eng.maze.maze[ny][nx] == 0: valid.append(card)
        return valid

    def get_valid_relative_actions(self) -> list[int]:
        valid_cardinals = self._get_valid_cardinal_actions()
        heading = self._get_pacman_heading_cardinal()
        valid_rel = []

        if heading in valid_cardinals:
            valid_rel.append(self.FORWARD)

        left_c = {self.UP: self.LEFT_C, self.DOWN: self.RIGHT_C, self.LEFT_C: self.DOWN, self.RIGHT_C: self.UP}[heading]
        if left_c in valid_cardinals:
            valid_rel.append(self.LEFT)

        right_c = {self.UP: self.RIGHT_C, self.DOWN: self.LEFT_C, self.LEFT_C: self.UP, self.RIGHT_C: self.DOWN}[heading]
        if right_c in valid_cardinals:
            valid_rel.append(self.RIGHT)

        if not valid_rel:
            valid_rel.append(self.FORWARD)

        return valid_rel

    def _get_info(self) -> dict:
        return {"score": self.engine.pacman.score, "frightened": self.engine.frightened_mode}

    # ── Boilerplate ───────────────────────────────────────────────────────────

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

    def _draw_debug_sensors(self):
        return
        if not self._screen: return
        ts = self.engine.tile_size
        pc = (int(self.engine.pacman.x + ts/2), int(self.engine.pacman.y + ts/2))
        for dx, dy in self._GLOBAL_RAY_DIRS:
            cx, cy = int(pc[0]/ts), int(pc[1]/ts)
            while True:
                cx += dx; cy += dy
                if not (0 <= cx < self.engine.maze.width and 0 <= cy < self.engine.maze.height) or self.engine.maze.maze[cy][cx] == 1: break
            pygame.draw.line(self._screen, (0,255,255), pc, (cx*ts+ts/2, cy*ts+ts/2), 1)

    def _render_human(self):
        if not self._screen: return
        pygame.event.pump()
        self._screen.fill((0,0,0))
        self.engine.draw(self._screen)
        self._draw_debug_sensors()
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._pygame_initialised: pygame.quit(); self._pygame_initialised = False

    @staticmethod
    def _parse_res(res_str):
        w, h = res_str.split('x')
        return int(w), int(h)