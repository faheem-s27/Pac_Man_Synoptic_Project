"""
neat_train_visual.py
====================
Trains NEAT with every genome running as its own independent GameEngine,
all rendered scaled-down into a single tiled window so you can watch the
entire population at once.

Layout  (default pop=150):
  Dynamically scales columns/rows to fit MAX_WINDOW.
  The full map is rendered into an off-screen surface then scaled to fit.

Controls
--------
  ESC / close window  →  stop training, save best genome found so far

Usage
-----
    python -m Code.neat_train_visual
    python -m Code.neat_train_visual --checkpoint checkpoints/neat-checkpoint-9
"""

import os
import sys
import pickle
import argparse
import math
import numpy as np

import neat
import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.GameEngine import GameEngine, GameState
from Code.Ghost      import GhostState
from Code.Settings   import Settings
from Code.CurriculumManager import CurriculumManager

# ─── Tunables ────────────────────────────────────────────────────────────────
MAZE_ALGORITHM  = "recursive_backtracking"
CONFIG_PATH     = os.path.join(_HERE, "neat_config.cfg")
CHECKPOINT_DIR  = os.path.join(_HERE, "checkpoints")
NUM_GENERATIONS = 200
MAX_STEPS       = 3_500  # Aggressively restricted to punish infinite loops

MAX_WINDOW_W = 1400
MAX_WINDOW_H = 900
INFO_BAR_H   = 44
TARGET_FPS   = 60
# ─────────────────────────────────────────────────────────────────────────────

# ACTION: 4-Action Space strictly enforced (NOOP amputated)
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
DIR_MAP      = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}


def compute_grid_layout(n: int, game_w: int, game_h: int):
    """Best (cols, rows, cell_w, cell_h) so n cells fit in MAX_WINDOW."""
    best = None
    best_area = 0
    for cols in range(1, n + 1):
        rows  = math.ceil(n / cols)
        scale = min((MAX_WINDOW_W // cols) / game_w,
                    ((MAX_WINDOW_H - INFO_BAR_H) // rows) / game_h)
        cw = int(game_w * scale)
        ch = int(game_h * scale)
        if cw < 40 or ch < 40:
            continue
        if cw * ch > best_area:
            best_area = cw * ch
            best = (cols, rows, cw, ch)
    if best is None:
        scale = min(MAX_WINDOW_W / (n * game_w), (MAX_WINDOW_H - INFO_BAR_H) / game_h)
        cw = max(40, int(game_w * scale))
        ch = max(40, int(game_h * scale))
        best = (n, 1, cw, ch)
    return best


def _make_engine(settings: dict) -> GameEngine:
    cfg = dict(settings)
    cfg["maze_algorithm"]   = MAZE_ALGORITHM
    cfg["enable_sound"]     = False   # never play audio during training
    cfg.setdefault("maze_seed", None) # random procedural generation
    engine = GameEngine(**cfg)
    engine.game_state = GameState.GAME
    engine.paused     = False
    return engine


# ─── Per-genome runner ────────────────────────────────────────────────────────

class GenomeRunner:
    """One complete independent game + NEAT network for a single genome."""

    _game_w: int = 0
    _game_h: int = 0

    def __init__(self, genome, config, env_settings: dict):
        self.genome       = genome
        self.net          = neat.nn.FeedForwardNetwork.create(genome, config)
        self.engine       = _make_engine(env_settings)
        self.done         = False
        self.total_reward = 0.0
        self.steps        = 0
        self.prev_score   = 0
        self.prev_lives   = self.engine.lives

        GenomeRunner._game_w = self.engine.screen_width
        GenomeRunner._game_h = self.engine.screen_height
        self._surf = pygame.Surface((GenomeRunner._game_w, GenomeRunner._game_h))

    # ── build observation vector (mirrors PacManEnv._get_vector_obs exactly) ──
    def _obs(self) -> list[float]:
        eng = self.engine
        ts  = eng.tile_size
        mw  = eng.maze.width  * ts
        mh  = eng.maze.height * ts

        # ACTION: Strict Center Mass Anchor (Top-left origin is banned)
        pac_cx = eng.pacman.x + (ts / 2.0)
        pac_cy = eng.pacman.y + (ts / 2.0)

        # [0-3] Pac-Man
        obs = [pac_cx / mw, pac_cy / mh, float(eng.pacman.direction[0]), float(eng.pacman.direction[1])]

        # [4-27] Ghosts
        for i in range(4):
            if i < len(eng.ghosts):
                g  = eng.ghosts[i]
                g_cx = g.x + (ts / 2.0)
                g_cy = g.y + (ts / 2.0)

                rx = (g_cx - pac_cx) / mw
                ry = (g_cy - pac_cy) / mh
                dist = (rx**2 + ry**2)**0.5

                threat = (1.0 if g.state in (GhostState.CHASE, GhostState.SCATTER)
                          else -1.0 if g.state == GhostState.FRIGHTENED else 0.0)
                obs.extend([rx, ry, dist, float(g.current_dir[0]), float(g.current_dir[1]), threat])
            else:
                obs.extend([0.0, 0.0, 1.5, 0.0, 0.0, 0.0])

        # [28-29] Nearest pellet radar
        all_p = eng.pellets + eng.power_pellets
        if all_p:
            # PELLETS are already PRE-CALCULATED as center-mass pixel coordinates
            dists = [(p[0]-pac_cx)**2 + (p[1]-pac_cy)**2 for p in all_p]
            cp    = all_p[dists.index(min(dists))]
            obs.extend([(cp[0]-pac_cx)/mw, (cp[1]-pac_cy)/mh])
        else:
            obs.extend([0.0, 0.0])

        # [30-33] Wall sensors (Center Mass Boundary Checking)
        cur_tx = int(pac_cx / ts)
        cur_ty = int(pac_cy / ts)
        for ddx, ddy in [(0,-1),(0,1),(-1,0),(1,0)]:
            tx, ty = cur_tx+ddx, cur_ty+ddy
            if 0 <= tx < eng.maze.width and 0 <= ty < eng.maze.height:
                obs.append(1.0 if eng.maze.maze[ty][tx] == 1 else 0.0)
            else:
                obs.append(1.0)

        # [34-39] Global state
        total_p      = max(eng.pellets_eaten_this_level + len(eng.pellets), 1)
        pellet_ratio = eng.pellets_eaten_this_level / total_p
        frit_active  = 1.0 if eng.frightened_mode else 0.0
        frit_time    = eng.frightened_timer / max(eng.frightened_duration, 1)

        pp_dist = 1.5
        if eng.power_pellets:
            pp_dist = min((p[0]-pac_cx)**2 + (p[1]-pac_cy)**2 for p in eng.power_pellets)**0.5 / mw

        obs.extend([pellet_ratio, frit_active, frit_time,
                    eng.lives / 3.0, 1.0 if eng.global_scatter_mode else 0.0, pp_dist])

        return obs

    # ── one simulation step ───────────────────────────────────────────────────
    def step(self):
        if self.done:
            return

        obs    = self._obs()
        output = self.net.activate(obs)
        action = int(np.argmax(output))  # ACTION: Robust argmax for 4-outputs

        self.engine.pacman.next_direction = DIR_MAP[action]
        self.engine.update()
        self.steps += 1

        # Reward = score delta / 2.0 − time penalty (mirrors PacManEnv)
        reward = -0.05
        score_delta = self.engine.pacman.score - self.prev_score
        if score_delta > 0:
            reward += float(score_delta) / 2.0
        self.prev_score = self.engine.pacman.score

        if self.engine.lives < self.prev_lives:
            reward -= 50.0
            self.prev_lives = self.engine.lives

        if self.engine.won:
            reward += 100.0
            self.engine.next_level()

        self.total_reward += reward

        if self.engine.game_over or self.steps >= MAX_STEPS:
            if self.engine.game_over:
                self.total_reward -= 50.0
            self.genome.fitness = self.total_reward
            self.done = True

    # ── render this genome's game into its grid cell ──────────────────────────
    def draw_cell(self, window: pygame.Surface, col: int, row: int,
                  cell_w: int, cell_h: int,
                  is_best: bool, label_font: pygame.font.Font):
        self._surf.fill((0, 0, 0))
        self.engine.draw(self._surf)

        if self.done:
            dim = pygame.Surface((GenomeRunner._game_w, GenomeRunner._game_h), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 180))
            self._surf.blit(dim, (0, 0))

        scaled = pygame.transform.scale(self._surf, (cell_w, cell_h))
        dest_x = col * cell_w
        dest_y = row * cell_h
        window.blit(scaled, (dest_x, dest_y))

        if is_best:
            pygame.draw.rect(window, (255, 215, 0), (dest_x, dest_y, cell_w, cell_h), 3)

        color = (255, 215, 0) if is_best else ((80, 80, 80) if self.done else (200, 200, 200))
        lbl   = label_font.render(f"{self.total_reward:+.0f}", True, color)
        window.blit(lbl, (dest_x + 3, dest_y + 3))


# ─── Best-genome saver ────────────────────────────────────────────────────────

class BestGenomeSaver(neat.reporting.BaseReporter):
    def __init__(self, save_dir):
        self.save_dir     = save_dir
        self.best_fitness = float("-inf")
        os.makedirs(save_dir, exist_ok=True)

    def post_evaluate(self, config, population, species_set, best_genome):
        if best_genome is not None and best_genome.fitness is not None and best_genome.fitness > self.best_fitness:
            self.best_fitness = best_genome.fitness
            path = os.path.join(self.save_dir, "best_genome.pkl")
            with open(path, "wb") as f:
                pickle.dump(best_genome, f)
            print(f"  ✓ New best  fitness={self.best_fitness:.1f}  → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(checkpoint=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    if checkpoint and os.path.exists(checkpoint):
        print(f"Resuming from: {checkpoint}")
        population = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(
        generation_interval=10,
        filename_prefix=os.path.join(CHECKPOINT_DIR, "neat-checkpoint-"),
    ))
    population.add_reporter(BestGenomeSaver(CHECKPOINT_DIR))

    pygame.init()
    curriculum = CurriculumManager(os.path.join(_ROOT, "game_settings.json"))

    label_font = pygame.font.Font(None, 18)
    info_font  = pygame.font.Font(None, 26)
    fps_clock  = pygame.time.Clock()

    generation_counter = [population.generation]
    window             = [None]

    def _save_best_on_exit(genome_list):
        scored = [g for g in genome_list if getattr(g, 'fitness', None) is not None and g.fitness > -9999]
        if scored:
            best = max(scored, key=lambda g: g.fitness)
            path = os.path.join(CHECKPOINT_DIR, "best_genome.pkl")
            with open(path, "wb") as f:
                pickle.dump(best, f)
            print(f"Saved on exit  fitness={best.fitness:.1f}  → {path}")

    def eval_genomes(genomes, cfg):
        gen = generation_counter[0]
        generation_counter[0] += 1

        # ACTION: Fetch specific Curriculum settings for the current generation
        current_settings = curriculum.get_settings_for_generation(gen)

        genome_list = [g for _, g in genomes]
        n           = len(genome_list)
        for g in genome_list:
            g.fitness = -9999.0

        # Inject dynamic settings into the runners
        runners = [GenomeRunner(g, cfg, current_settings) for g in genome_list]

        game_w  = GenomeRunner._game_w
        game_h  = GenomeRunner._game_h
        cols, rows, cell_w, cell_h = compute_grid_layout(n, game_w, game_h)
        win_w   = cols * cell_w
        win_h   = rows * cell_h + INFO_BAR_H

        if window[0] is None or window[0].get_size() != (win_w, win_h):
            window[0] = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption("NEAT Pac-Man — Visual Curriculum Training")

        best_runner_idx = None

        while not all(r.done for r in runners):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    _save_best_on_exit(genome_list)
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    _save_best_on_exit(genome_list)
                    pygame.quit(); sys.exit()

            for r in runners:
                r.step()

            alive = [i for i, r in enumerate(runners) if not r.done]
            if alive:
                best_runner_idx = max(alive, key=lambda i: runners[i].total_reward)

            game_area_h = rows * cell_h
            window[0].fill((15, 15, 15))
            for idx, r in enumerate(runners):
                r.draw_cell(window[0], idx % cols, idx // cols,
                            cell_w, cell_h,
                            is_best=(idx == best_runner_idx),
                            label_font=label_font)

            for c in range(1, cols):
                pygame.draw.line(window[0], (40, 40, 40), (c*cell_w, 0), (c*cell_w, game_area_h))
            for r in range(1, rows):
                pygame.draw.line(window[0], (40, 40, 40), (0, r*cell_h), (win_w, r*cell_h))

            alive_count = sum(1 for r in runners if not r.done)
            best_r  = max(r.total_reward for r in runners)
            avg_r   = sum(r.total_reward for r in runners) / n
            info    = (f"Gen {gen}  |  Alive {alive_count}/{n}  |  "
                       f"Best: {best_r:+.0f}  |  Avg: {avg_r:+.0f}  |  "
                       f"FPS {fps_clock.get_fps():.0f}")

            window[0].fill((0, 0, 0), (0, game_area_h, win_w, INFO_BAR_H))
            window[0].blit(info_font.render(info, True, (200, 200, 200)), (8, game_area_h + 10))
            pygame.display.flip()
            fps_clock.tick(TARGET_FPS)

        for i, g in enumerate(genome_list):
            g.fitness = runners[i].total_reward

    try:
        # Run 1 generation at a time to sync with the counter properly, or run full loop
        # The while loop is handled natively by NEAT, we just pass the evaluator
        winner = population.run(eval_genomes, NUM_GENERATIONS)
        path = os.path.join(CHECKPOINT_DIR, "winner_genome.pkl")
        with open(path, "wb") as f:
            pickle.dump(winner, f)
        print(f"\nWinner → {path}  fitness={winner.fitness:.1f}")
    except SystemExit:
        pass
    finally:
        if pygame.get_init():
            pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    run(checkpoint=args.checkpoint)