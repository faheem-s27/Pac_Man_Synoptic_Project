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

_HERE      = os.path.dirname(os.path.abspath(__file__))
_NEAT_ROOT = os.path.dirname(_HERE)                          # Code/Models/NEAT/
_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))  # project root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.Environment.PacManEnv import PacManEnv
from Code.Environment.CurriculumManager import CurriculumManager

# ─── Tunables ────────────────────────────────────────────────────────────────
MAZE_ALGORITHM  = "recursive_backtracking"
CONFIG_PATH     = os.path.join(_NEAT_ROOT, "neat_config.cfg")
CHECKPOINT_DIR  = os.path.join(_NEAT_ROOT, "Checkpoints")
NUM_GENERATIONS = 200
NEAT_MAX_EPISODE_STEPS = None

MAX_WINDOW_W = 1400
MAX_WINDOW_H = 900
INFO_BAR_H   = 44
TARGET_FPS   = 60
# ─────────────────────────────────────────────────────────────────────────────

# ACTION: Egocentric 4-action space (FORWARD/LEFT/RIGHT/BACKWARD)
ACTION_NAMES = ["FORWARD", "LEFT", "RIGHT", "BACKWARD"]


def _settings_for_generation(curriculum: CurriculumManager, generation: int) -> dict:
    stage_count = max(1, len(curriculum.stage_profiles))
    if NUM_GENERATIONS <= 1:
        stage_idx = 0
    else:
        progress = max(0.0, min(1.0, float(generation) / float(NUM_GENERATIONS - 1)))
        stage_idx = min(stage_count - 1, int(progress * stage_count))
    curriculum.current_stage = stage_idx
    settings = curriculum.get_settings()
    settings["max_episode_steps"] = NEAT_MAX_EPISODE_STEPS
    return settings


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


def _make_env(settings: dict, maze_seed: int | None = None) -> PacManEnv:
    cfg = dict(settings)
    cfg["enable_sound"] = False
    if maze_seed is not None:
        cfg["maze_seed"] = maze_seed
    env = PacManEnv(render_mode="rgb_array", obs_type="vector", settings=cfg)
    env.reset(seed=maze_seed)
    return env


# ─── Per-genome runner ────────────────────────────────────────────────────────

class GenomeRunner:
    """One complete independent PacManEnv + NEAT network for a single genome."""

    _game_w: int = 0
    _game_h: int = 0

    def __init__(self, genome, config, env_settings: dict):
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        maze_seed = int(np.random.randint(0, 10_000_000))
        self.env = _make_env(env_settings, maze_seed=maze_seed)
        self.engine = self.env.engine
        self.obs, _ = self.env.reset(seed=maze_seed)
        self.done = False
        self.total_reward = 0.0
        self.steps = 0
        self.max_steps = env_settings.get('max_episode_steps', NEAT_MAX_EPISODE_STEPS)

        GenomeRunner._game_w = self.engine.screen_width
        GenomeRunner._game_h = self.engine.screen_height
        self._surf = pygame.Surface((GenomeRunner._game_w, GenomeRunner._game_h))

    def step(self):
        if self.done: return
        output = self.net.activate(self.obs.tolist())
        valid_actions = self.env.get_valid_actions()
        action = max(valid_actions, key=lambda a: output[a]) if valid_actions else int(np.argmax(output))

        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        self.engine = self.env.engine
        self.steps += 1

        self.total_reward += reward
        reached_step_limit = (self.max_steps is not None) and (self.steps >= self.max_steps)
        if terminated or truncated or reached_step_limit:
            self.genome.fitness = self.total_reward
            self.done = True

    # ── render this genome's game into its grid cell ──────────────────────────
    def draw_cell(self, window: pygame.Surface, col: int, row: int,
                  cell_w: int, cell_h: int,
                  is_best: bool, label_font: pygame.font.Font):
        self._surf.fill((0, 0, 0))
        frame = self.env.render()
        if frame is not None:
            self._surf.blit(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (0, 0))

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

    def close(self):
        self.env.close()


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

    probe_settings = _settings_for_generation(curriculum=CurriculumManager(), generation=0)
    probe_env = PacManEnv(render_mode=None, obs_type="vector", settings=probe_settings)
    probe_obs, _ = probe_env.reset(seed=123)
    obs_dim = len(probe_obs)
    action_dim = int(getattr(probe_env.action_space, "n", 4))
    probe_env.close()
    if config.genome_config.num_inputs != obs_dim:
        raise ValueError(
            f"NEAT config num_inputs={config.genome_config.num_inputs} does not match PacManEnv obs_dim={obs_dim}."
        )
    if config.genome_config.num_outputs != action_dim:
        raise ValueError(
            f"NEAT config num_outputs={config.genome_config.num_outputs} does not match action_dim={action_dim}."
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
    curriculum = CurriculumManager()

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

        # Fetch curriculum settings for this generation.
        current_settings = _settings_for_generation(curriculum, gen)

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
        win_surface = window[0]
        assert win_surface is not None

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
            win_surface.fill((15, 15, 15))
            for idx, r in enumerate(runners):
                r.draw_cell(win_surface, idx % cols, idx // cols,
                            cell_w, cell_h,
                            is_best=(idx == best_runner_idx),
                            label_font=label_font)

            for c in range(1, cols):
                pygame.draw.line(win_surface, (40, 40, 40), (c*cell_w, 0), (c*cell_w, game_area_h))
            for r in range(1, rows):
                pygame.draw.line(win_surface, (40, 40, 40), (0, r*cell_h), (win_w, r*cell_h))

            alive_count = sum(1 for r in runners if not r.done)
            best_r  = max(r.total_reward for r in runners)
            avg_r   = sum(r.total_reward for r in runners) / n
            info    = (f"Gen {gen}  |  Alive {alive_count}/{n}  |  "
                       f"Best: {best_r:+.0f}  |  Avg: {avg_r:+.0f}  |  "
                       f"FPS {fps_clock.get_fps():.0f}")

            win_surface.fill((0, 0, 0), (0, game_area_h, win_w, INFO_BAR_H))
            win_surface.blit(info_font.render(info, True, (200, 200, 200)), (8, game_area_h + 10))
            pygame.display.flip()
            fps_clock.tick(TARGET_FPS)

        for i, g in enumerate(genome_list):
            g.fitness = runners[i].total_reward
        for r in runners:
            r.close()

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