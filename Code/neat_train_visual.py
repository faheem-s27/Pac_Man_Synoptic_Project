"""
neat_train_visual.py
====================
Trains NEAT with every genome running as its own independent GameEngine,
all rendered scaled-down into a single tiled window so you can watch the
entire population at once.

Layout  (default pop=50):
  7 columns × 8 rows  = 56 cells  (last 6 stay blank)
  Each cell is CELL_W × CELL_H pixels.
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

import neat
import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.GameEngine import GameEngine, GameState
from Code.Ghost      import GhostState
from Code.Settings   import Settings

# ─── Tunables ────────────────────────────────────────────────────────────────
MAZE_ALGORITHM  = "recursive_backtracking"
CONFIG_PATH     = os.path.join(_HERE, "neat_config.ini")
CHECKPOINT_DIR  = os.path.join(_HERE, "checkpoints")
NUM_GENERATIONS = 200
MAX_STEPS       = 3_000

GRID_COLS       = 5          # columns of games
CELL_W          = 160        # pixel width  of each cell in the window
CELL_H          = 160        # pixel height of each cell in the window
INFO_BAR_H      = 44
TARGET_FPS      = 60
# ─────────────────────────────────────────────────────────────────────────────

ACTION_NAMES = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT"]
DIR_MAP      = {0:(0,0), 1:(0,-1), 2:(0,1), 3:(-1,0), 4:(1,0)}


def _load_base_cfg() -> dict:
    s = Settings(os.path.join(_HERE, "game_settings.json"))
    return s.get_all()


def _make_engine(base_cfg: dict) -> GameEngine:
    cfg = dict(base_cfg)
    cfg["maze_algorithm"]   = MAZE_ALGORITHM
    cfg["use_classic_maze"] = base_cfg.get("use_classic_maze", False)
    cfg["enable_sound"]     = False   # never play audio during training
    # maze_seed comes from settings JSON (key "maze_seed"); None = random each run
    cfg.setdefault("maze_seed", None)
    engine = GameEngine(**cfg)
    engine.game_state = GameState.GAME
    engine.paused     = False
    return engine


# ─── Per-genome runner ────────────────────────────────────────────────────────

class GenomeRunner:
    """One complete independent game + NEAT network for a single genome."""

    # Off-screen surface shared size = real game resolution
    _game_w: int = 0
    _game_h: int = 0

    def __init__(self, genome, config, base_cfg: dict):
        self.genome       = genome
        self.net          = neat.nn.FeedForwardNetwork.create(genome, config)
        self.engine       = _make_engine(base_cfg)
        self.done         = False
        self.total_reward = 0.0
        self.steps        = 0
        self.prev_score   = 0
        self.prev_lives   = self.engine.lives

        # Off-screen surface at full game resolution
        GenomeRunner._game_w = self.engine.screen_width
        GenomeRunner._game_h = self.engine.screen_height
        self._surf = pygame.Surface((GenomeRunner._game_w, GenomeRunner._game_h))

    # ── build observation vector (mirrors PacManEnv._get_vector_obs) ─────────
    def _obs(self) -> list[float]:
        eng = self.engine
        ts  = eng.tile_size
        mw  = eng.maze.width  * ts
        mh  = eng.maze.height * ts
        pac = eng.pacman

        obs = [pac.x / mw, pac.y / mh, float(pac.direction[0]), float(pac.direction[1])]

        for i in range(4):
            if i < len(eng.ghosts):
                g  = eng.ghosts[i]
                rx = (g.x - pac.x) / mw
                ry = (g.y - pac.y) / mh
                threat = 0.0
                if g.state in (GhostState.CHASE, GhostState.SCATTER):
                    threat = 1.0
                elif g.state == GhostState.FRIGHTENED:
                    threat = -1.0
                obs.extend([rx, ry, (rx**2 + ry**2)**0.5, threat])
            else:
                obs.extend([0.0, 0.0, 1.0, 0.0])

        total_p = len(eng.pellets) + eng.pellets_eaten_this_level
        pellet_ratio = eng.pellets_eaten_this_level / max(total_p, 1)
        frit_active  = 1.0 if eng.frightened_mode else 0.0
        frit_time    = eng.frightened_timer / max(eng.frightened_duration, 1)

        all_p = eng.pellets + eng.power_pellets
        if all_p:
            dists = [(p[0]-pac.x)**2 + (p[1]-pac.y)**2 for p in all_p]
            cp    = all_p[dists.index(min(dists))]
            obs.extend([(cp[0]-pac.x)/mw, (cp[1]-pac.y)/mh])
        else:
            obs.extend([0.0, 0.0])

        cur_tx = int(pac.x / ts)
        cur_ty = int(pac.y / ts)
        for ddx, ddy in [(0,-1),(0,1),(-1,0),(1,0)]:
            tx, ty = cur_tx+ddx, cur_ty+ddy
            if 0 <= tx < eng.maze.width and 0 <= ty < eng.maze.height:
                obs.append(1.0 if eng.maze.maze[ty][tx] == 1 else 0.0)
            else:
                obs.append(1.0)

        obs.extend([pellet_ratio, frit_active, frit_time,
                    float(eng.lives / 3), float(eng.global_scatter_mode)])

        # Nearest power pellet direction + count
        if eng.power_pellets:
            pp_dists = [(p[0]-pac.x)**2 + (p[1]-pac.y)**2 for p in eng.power_pellets]
            cp = eng.power_pellets[pp_dists.index(min(pp_dists))]
            obs.extend([(cp[0]-pac.x)/mw, (cp[1]-pac.y)/mh,
                        len(eng.power_pellets) / max(len(eng.power_pellets), 1)])
        else:
            obs.extend([0.0, 0.0, 0.0])

        return obs

    # ── one simulation step ───────────────────────────────────────────────────
    def step(self):
        if self.done:
            return

        obs    = self._obs()
        output = self.net.activate(obs)
        action = output.index(max(output))

        if action != 0:
            self.engine.pacman.next_direction = DIR_MAP[action]

        self.engine.update()
        self.steps += 1

        # Reward = score delta − time penalty
        score_delta = self.engine.pacman.score - self.prev_score
        reward      = float(score_delta) - 0.1
        self.prev_score = self.engine.pacman.score

        if self.engine.lives < self.prev_lives:
            reward -= 500.0
            self.prev_lives = self.engine.lives

        if self.engine.won:
            reward += 500.0
            self.engine.next_level()

        self.total_reward += reward

        if self.engine.game_over or self.steps >= MAX_STEPS:
            if self.engine.game_over:
                self.total_reward -= 500.0
            self.genome.fitness = self.total_reward
            self.done = True

    # ── render this genome's game into its grid cell ──────────────────────────
    def draw_cell(self, window: pygame.Surface, col: int, row: int,
                  is_best: bool, label_font: pygame.font.Font):
        self._surf.fill((0, 0, 0))
        self.engine.draw(self._surf)

        # Dim finished cells
        if self.done:
            dim = pygame.Surface((GenomeRunner._game_w, GenomeRunner._game_h),
                                 pygame.SRCALPHA)
            dim.fill((0, 0, 0, 180))
            self._surf.blit(dim, (0, 0))

        # Scale down to cell size
        scaled = pygame.transform.scale(self._surf, (CELL_W, CELL_H))

        # Gold border for the current best genome
        dest_x = col * CELL_W
        dest_y = row * CELL_H
        window.blit(scaled, (dest_x, dest_y))

        if is_best:
            pygame.draw.rect(window, (255, 215, 0),
                             (dest_x, dest_y, CELL_W, CELL_H), 3)

        # Tiny fitness label
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
        if best_genome.fitness > self.best_fitness:
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
    base_cfg = _load_base_cfg()

    label_font = pygame.font.Font(None, 18)
    info_font  = pygame.font.Font(None, 26)
    fps_clock  = pygame.time.Clock()

    generation_counter = [0]
    window             = [None]   # created lazily once we know pop size

    def _save_best_on_exit(genome_list):
        scored = [g for g in genome_list if g.fitness is not None and g.fitness > -9999]
        if scored:
            best = max(scored, key=lambda g: g.fitness)
            path = os.path.join(CHECKPOINT_DIR, "best_genome.pkl")
            with open(path, "wb") as f:
                pickle.dump(best, f)
            print(f"Saved on exit  fitness={best.fitness:.1f}  → {path}")

    def eval_genomes(genomes, cfg):
        generation_counter[0] += 1
        gen = generation_counter[0]

        genome_list = [g for _, g in genomes]
        n           = len(genome_list)
        for g in genome_list:
            g.fitness = -9999.0

        # Work out grid dimensions from population size
        cols      = GRID_COLS
        rows      = math.ceil(n / cols)
        win_w     = cols * CELL_W
        win_h     = rows * CELL_H + INFO_BAR_H

        # Create / resize window
        if window[0] is None:
            window[0] = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption("NEAT Pac-Man — Visual Training")

        runners = [GenomeRunner(g, cfg, base_cfg) for g in genome_list]
        best_runner_idx = None

        while not all(r.done for r in runners):
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    _save_best_on_exit(genome_list)
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    _save_best_on_exit(genome_list)
                    pygame.quit(); sys.exit()

            # Step all live runners
            for r in runners:
                r.step()

            # Find best alive runner for highlight
            alive = [i for i, r in enumerate(runners) if not r.done]
            if alive:
                best_runner_idx = max(alive, key=lambda i: runners[i].total_reward)

            # Draw grid
            window[0].fill((15, 15, 15))
            for idx, r in enumerate(runners):
                col = idx % cols
                row = idx // cols
                r.draw_cell(window[0], col, row,
                            is_best=(idx == best_runner_idx),
                            label_font=label_font)

            # Grid lines
            game_area_h = rows * CELL_H
            for c in range(1, cols):
                pygame.draw.line(window[0], (40, 40, 40),
                                 (c*CELL_W, 0), (c*CELL_W, game_area_h))
            for r in range(1, rows):
                pygame.draw.line(window[0], (40, 40, 40),
                                 (0, r*CELL_H), (win_w, r*CELL_H))

            # Info bar
            alive_count = sum(1 for r in runners if not r.done)
            best_r  = max(r.total_reward for r in runners)
            avg_r   = sum(r.total_reward for r in runners) / n
            info    = (f"Gen {gen}  |  Alive {alive_count}/{n}  |  "
                       f"Best: {best_r:+.0f}  |  Avg: {avg_r:+.0f}  |  "
                       f"FPS {fps_clock.get_fps():.0f}")
            window[0].fill((0, 0, 0), (0, game_area_h, win_w, INFO_BAR_H))
            window[0].blit(info_font.render(info, True, (200, 200, 200)),
                           (8, game_area_h + 10))
            pygame.display.flip()
            fps_clock.tick(TARGET_FPS)

        # Assign final fitnesses
        for i, g in enumerate(genome_list):
            g.fitness = runners[i].total_reward

    try:
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

