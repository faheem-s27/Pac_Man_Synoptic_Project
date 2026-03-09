"""
neat_visualize.py
=================
Run a saved genome with real-time sensory debug lines to audit the AI's 'vision'.
"""

import os
import sys
import pickle
import pygame
import numpy as np
import neat

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Code.PacManEnv import PacManEnv
from Code.Ghost import GhostState


def draw_debug_radar(env):
    """Draws the 40-element sensor vector directly onto the PyGame window."""
    if env._screen is None or env._engine is None:
        return

    eng = env._engine
    pac_pos = (eng.pacman.x, eng.pacman.y)
    mw_px = eng.maze.width * eng.tile_size
    mh_px = eng.maze.height * eng.tile_size

    # 1. Pellet Radar (Green)
    if eng.pellets or eng.power_pellets:
        all_p = eng.pellets + eng.power_pellets
        dists = [(p[0] - pac_pos[0]) ** 2 + (p[1] - pac_pos[1]) ** 2 for p in all_p]
        target_p = all_p[np.argmin(dists)]
        pygame.draw.line(env._screen, (0, 255, 0), pac_pos, (target_p[0], target_p[1]), 2)

    # 2. Ghost Tracking (Red for Threat, Blue for Frightened)
    for g in eng.ghosts:
        color = (255, 0, 0) if g.state in [GhostState.CHASE, GhostState.SCATTER] else (0, 255, 255)
        pygame.draw.line(env._screen, color, pac_pos, (g.x, g.y), 1)

    # 3. Wall Sensors (Yellow Circles)
    ts = eng.tile_size
    for dx, dy in [(0, -ts), (0, ts), (-ts, 0), (ts, 0)]:
        tx, ty = int((pac_pos[0] + dx) / ts), int((pac_pos[1] + dy) / ts)
        if not (0 <= tx < eng.maze.width and 0 <= ty < eng.maze.height) or eng.maze.maze[ty][tx] == 1:
            pygame.draw.circle(env._screen, (255, 255, 0), (pac_pos[0] + dx, pac_pos[1] + dy), 5)


def run_visual_replay(genome_path):
    # Load Genome
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Load NEAT Config
    config_path = os.path.join(_HERE, "neat_config.cfg")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Init Env
    env = PacManEnv(render_mode="human", obs_type="vector", maze_seed=None)
    obs, _ = env.reset()

    print(f"\n[DEBUG] Running {genome_path} with Sensory Overlay...")

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # AI Action
        outputs = net.activate(obs.tolist())
        action = int(np.argmax(outputs))

        obs, reward, terminated, truncated, _ = env.step(action)

        # INJECT VISUALIZATION: Overlay sensors on the frame
        draw_debug_radar(env)
        pygame.display.flip()

        if terminated or truncated:
            obs, _ = env.reset()

        clock.tick(60)

    env.close()


if __name__ == "__main__":
    # Point this to your latest best genome
    GENOME_FILE = os.path.join(_HERE, "checkpoints", "best_genome.pkl")
    run_visual_replay(GENOME_FILE)