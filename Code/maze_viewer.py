import sys
import os
import argparse
import pygame
import random
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from Code.Maze import Maze
from Code.Pathfinding import Pathfinding, validate_maze_connectivity

# Ghost colors
BLINKY_RED = (255, 0, 0)
PINKY_PINK = (255, 184, 255)
INKY_CYAN = (0, 255, 255)
CLYDE_ORANGE = (255, 184, 82)
SETTINGS_PATH = os.path.join(_HERE, "game_settings.json")


def _parse_resolution(resolution_str):
    """Match GameEngine.parse_resolution behavior."""
    try:
        w, h = resolution_str.split('x')
        return int(w), int(h)
    except Exception:
        return 800, 800


def _resolve_engine_maze_dimensions(window_resolution, tile_size):
    """Replicate GameEngine maze tile sizing exactly."""
    screen_w, screen_h = _parse_resolution(window_resolution)
    maze_w = screen_w // tile_size
    maze_h = screen_h // tile_size

    if maze_w % 2 == 0:
        maze_w -= 1
    if maze_h % 2 == 0:
        maze_h -= 1
    return maze_w, maze_h


def _load_viewer_defaults():
    """Load defaults from game settings so viewer matches runtime config."""
    defaults = {
        "window_resolution": "800x800",
        "tile_size": 40,
        "maze_algorithm": "recursive_backtracking",
        "maze_seed": None,
    }
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            defaults["window_resolution"] = data.get("window_resolution", defaults["window_resolution"])
            defaults["tile_size"] = int(data.get("tile_size", defaults["tile_size"]))
            defaults["maze_algorithm"] = data.get("maze_algorithm", defaults["maze_algorithm"])

            seed_val = data.get("maze_seed", None)
            if seed_val in ("", None):
                defaults["maze_seed"] = None
            else:
                defaults["maze_seed"] = int(seed_val)
    except Exception:
        pass
    return defaults


def _build_maze_grid_from_seed(tile_size, width, height, algorithm, seed):
    m = Maze(tile_size=tile_size, width=width, height=height,
             algorithm=algorithm, seed=seed)
    return m, [row[:] for row in m.maze]


def _compute_spawn_for_flood(maze):
    spawn_gx = getattr(maze, 'door_x', maze.width // 2)
    door_y = getattr(maze, 'door_y', None)
    if door_y is None:
        spawn_gy = getattr(maze, 'cage_top', maze.height // 2) - 1
    else:
        spawn_gy = door_y - 1

    spawn_gx = max(0, min(maze.width - 1, spawn_gx))
    spawn_gy = max(0, min(maze.height - 1, spawn_gy))

    if getattr(maze, 'maze', None) is None: return None

    if maze.maze[spawn_gy][spawn_gx] not in (0, 2):
        found = False
        for dy in range(-3, 4):
            ny = spawn_gy + dy
            if 0 <= ny < maze.height and maze.maze[ny][spawn_gx] in (0, 2):
                spawn_gy = ny
                found = True
                break
        if not found: return None

    return spawn_gx, spawn_gy


def _compute_flood_reachable(maze):
    start = _compute_spawn_for_flood(maze)
    if start is None: return set()

    sx, sy = start
    visited = set()
    queue = [(sx, sy)]

    while queue:
        cx, cy = queue.pop(0)
        if (cx, cy) in visited: continue
        visited.add((cx, cy))

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < maze.width and 0 <= ny < maze.height:
                if maze.maze[ny][nx] in (0, 2) and (nx, ny) not in visited:
                    queue.append((nx, ny))

    return visited


def _get_spawn_and_corners(maze):
    spawn_gx = getattr(maze, 'door_x', maze.width // 2)
    door_y = getattr(maze, 'door_y', None)
    if door_y is None:
        spawn_gy = getattr(maze, 'cage_top', maze.height // 2) - 1
    else:
        spawn_gy = door_y - 1

    spawn_gx = max(0, min(maze.width - 1, spawn_gx))
    spawn_gy = max(0, min(maze.height - 1, spawn_gy))

    if maze.maze[spawn_gy][spawn_gx] not in (0, 2):
        for dy in range(-3, 4):
            ny = spawn_gy + dy
            if 0 <= ny < maze.height and maze.maze[ny][spawn_gx] in (0, 2):
                spawn_gy = ny
                break

    corners = [
        (1, 1),
        (maze.width - 2, 1),
        (1, maze.height - 2),
        (maze.width - 2, maze.height - 2),
    ]
    return (spawn_gx, spawn_gy), corners


def _compute_nodes(maze):
    nodes = []
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    opposites = {(0, -1): (0, 1), (0, 1): (0, -1), (-1, 0): (1, 0), (1, 0): (-1, 0)}

    for y in range(maze.height):
        for x in range(maze.width):
            if maze.maze[y][x] in (0, 2):
                valid_dirs = []
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < maze.width and 0 <= ny < maze.height:
                        if maze.maze[ny][nx] in (0, 2):
                            valid_dirs.append((dx, dy))

                is_node = False
                if len(valid_dirs) >= 3: is_node = True
                elif len(valid_dirs) == 2:
                    if opposites[valid_dirs[0]] != valid_dirs[1]: is_node = True
                elif len(valid_dirs) == 1: is_node = True

                if is_node: nodes.append((x, y))
    return nodes


def run_viewer(seed=None, algorithm="recursive_backtracking", tile_size=40,
               show_pellets=True, show_power_pellets=True,
               show_flood_fill=True, show_nodes=True,
               show_ghost_paths=True,
               window_resolution="800x800"):
    pygame.init()

    maze_w_tiles, maze_h_tiles = _resolve_engine_maze_dimensions(window_resolution, tile_size)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    def reload_maze(target_seed):
        """Encapsulates all computation so it can be re-triggered on click."""
        maze, base_grid = _build_maze_grid_from_seed(
            tile_size, maze_w_tiles, maze_h_tiles, algorithm, target_seed
        )

        # 1. Validation
        if target_seed is not None:
            try:
                _, test_grid = _build_maze_grid_from_seed(tile_size, maze_w_tiles, maze_h_tiles, algorithm, target_seed)
                if base_grid == test_grid:
                    seed_status = "VALID" if validate_maze_connectivity(maze) else "INVALID (flood-fill failed)"
                else:
                    seed_status = "INPUT SEED BANNED → resampled layout"
            except Exception:
                seed_status = "ERROR RECHECKING"
        else:
            seed_status = "VALID (random)" if validate_maze_connectivity(maze) else "INVALID (flood-fill failed)"

        pygame.display.set_caption(f"Maze Viewer - seed={target_seed} [{seed_status}]")

        # 2. Extract Data
        cage_left = getattr(maze, 'cage_left', -1)
        cage_right = getattr(maze, 'cage_right', -1)
        cage_top = getattr(maze, 'cage_top', -1)
        cage_bottom = getattr(maze, 'cage_bottom', -1)

        pellets = []
        half_ts = tile_size // 2

        def is_in_cage(tx, ty):
            return (cage_left <= tx <= cage_right) and (cage_top <= ty <= cage_bottom)

        # Match GameEngine._initialize_pellets so viewer quantity and layout stay in sync.
        for y in range(maze.height):
            for x in range(maze.width):
                if maze.maze[y][x] == 0 and not is_in_cage(x, y):
                    cx = x * tile_size + half_ts
                    cy = y * tile_size + half_ts
                    pellets.append((cx, cy))

                    if x + 1 < maze.width and maze.maze[y][x + 1] == 0 and not is_in_cage(x + 1, y):
                        pellets.append((cx + half_ts, cy))

                    if y + 1 < maze.height and maze.maze[y + 1][x] == 0 and not is_in_cage(x, y + 1):
                        pellets.append((cx, cy + half_ts))

        spawn_g, corners = _get_spawn_and_corners(maze)
        pf = Pathfinding(maze)
        paths = []
        for (tx, ty), color in zip(corners, [BLINKY_RED, PINKY_PINK, INKY_CYAN, CLYDE_ORANGE]):
            path = pf.find_shortest_path(spawn_g[0], spawn_g[1], tx, ty, current_dir=(0, 0)) if (0 <= tx < maze.width and 0 <= ty < maze.height) else []
            paths.append((path, color))

        flood_reachable = _compute_flood_reachable(maze)
        topological_nodes = _compute_nodes(maze)

        return maze, target_seed, seed_status, pellets, spawn_g, paths, flood_reachable, topological_nodes

    # Initial Load
    active_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    maze, current_seed, seed_status, pellets, spawn_g, paths, flood_reachable, topological_nodes = reload_maze(active_seed)
    width_px = maze.width * tile_size
    height_px = maze.height * tile_size
    screen = pygame.display.set_mode((width_px, height_px))

    # Runtime overlay toggles (CLI flags define initial state).
    draw_pellets = show_pellets
    draw_power_pellets = show_power_pellets
    draw_flood_fill = show_flood_fill
    draw_nodes = show_nodes
    draw_ghost_paths = show_ghost_paths

    def _on_off(value):
        return "ON" if value else "OFF"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_1, pygame.K_KP1):
                    draw_pellets = not draw_pellets
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    draw_power_pellets = not draw_power_pellets
                elif event.key in (pygame.K_3, pygame.K_KP3):
                    draw_flood_fill = not draw_flood_fill
                elif event.key in (pygame.K_4, pygame.K_KP4):
                    draw_nodes = not draw_nodes
                elif event.key in (pygame.K_5, pygame.K_KP5):
                    draw_ghost_paths = not draw_ghost_paths
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # RECALCULATE ENTIRE MAZE ON CLICK
                active_seed = random.randint(0, 2**31 - 1)
                maze, current_seed, seed_status, pellets, spawn_g, paths, flood_reachable, topological_nodes = reload_maze(active_seed)

        screen.fill((0, 0, 0))
        maze.draw(screen)

        if draw_flood_fill and flood_reachable:
            overlay = pygame.Surface((width_px, height_px), pygame.SRCALPHA)
            for gx, gy in flood_reachable:
                pygame.draw.rect(overlay, (0, 0, 255, 60), pygame.Rect(gx * tile_size, gy * tile_size, tile_size, tile_size))
            screen.blit(overlay, (0, 0))

        if draw_pellets:
            for px, py in pellets:
                pygame.draw.circle(screen, (255, 255, 0), (px, py), max(2, tile_size // 8))

        if draw_power_pellets:
            corner_tiles = [(1, 1), (maze.width - 2, 1), (1, maze.height - 2), (maze.width - 2, maze.height - 2)]
            for gx, gy in corner_tiles:
                if 0 <= gx < maze.width and 0 <= gy < maze.height and maze.maze[gy][gx] == 0:
                    pygame.draw.circle(screen, (0, 255, 255), (gx * tile_size + tile_size // 2, gy * tile_size + tile_size // 2), max(4, tile_size // 6))

        if draw_ghost_paths:
            for path, color in paths:
                if not path:
                    continue
                pts = [(gx * tile_size + tile_size // 2, gy * tile_size + tile_size // 2) for gx, gy in path]
                if len(pts) >= 2:
                    pygame.draw.lines(screen, color, False, pts, max(2, tile_size // 20))
                pygame.draw.circle(screen, color, pts[-1], max(3, tile_size // 10))

        # --- TOPOLOGICAL NODES ---
        if draw_nodes:
            for nx, ny in topological_nodes:
                px, py = nx * tile_size + tile_size // 2, ny * tile_size + tile_size // 2
                pygame.draw.circle(screen, (0, 255, 0), (px, py), max(4, tile_size // 4))
                pygame.draw.circle(screen, (0, 0, 0), (px, py), max(2, tile_size // 8))

        # Spawn Marker
        spawn_for_flood = _compute_spawn_for_flood(maze)
        sx, sy = spawn_for_flood if spawn_for_flood else spawn_g
        pygame.draw.circle(screen, (255, 255, 255), (sx * tile_size + tile_size // 2, sy * tile_size + tile_size // 2), max(4, tile_size // 9), 1)

        # Text Overlay
        text = f"Seed: {current_seed} | Status: {seed_status} | CLICK TO RANDOMIZE"
        screen.blit(font.render(text, True, (255, 255, 255)), (5, 5))

        helper_bg_h = 24
        pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, height_px - helper_bg_h, width_px, helper_bg_h))
        helper = (
            f"1 Pellets[{_on_off(draw_pellets)}]  "
            f"2 Power[{_on_off(draw_power_pellets)}]  "
            f"3 Flood[{_on_off(draw_flood_fill)}]  "
            f"4 Nodes[{_on_off(draw_nodes)}]  "
            f"5 GhostPath[{_on_off(draw_ghost_paths)}]"
        )
        screen.blit(font.render(helper, True, (180, 180, 180)), (5, height_px - helper_bg_h + 4))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    defaults = _load_viewer_defaults()
    parser = argparse.ArgumentParser(description="Standalone maze viewer for inspecting generated layouts/seeds.")
    parser.add_argument("--seed", type=int, default=defaults["maze_seed"], help="Maze seed to visualize.")
    parser.add_argument("--algorithm", type=str, default=defaults["maze_algorithm"],
                        choices=["recursive_backtracking", "prims", "random_walk"],
                        help="Maze generation algorithm.")
    parser.add_argument("--tile-size", type=int, default=defaults["tile_size"], help="Tile size in pixels.")
    parser.add_argument("--window-resolution", type=str, default=defaults["window_resolution"],
                        help="Resolution used to derive maze dimensions (same as GameEngine).")
    parser.add_argument("--no-pellets", action="store_false", help="Hide regular pellets.")
    parser.add_argument("--no-power-pellets", action="store_false", help="Hide power pellets overlay.")
    parser.add_argument("--no-flood", action="store_false", help="Disable flood-fill debug overlay.")
    parser.add_argument("--no-nodes", action="store_false", help="Hide topological node overlay.")
    parser.add_argument("--no-ghost-paths", action="store_false", help="Hide ghost pathfinding overlay.")

    args = parser.parse_args()

    run_viewer(
        seed=args.seed,
        algorithm=args.algorithm,
        tile_size=args.tile_size,
        show_pellets=not args.no_pellets,
        show_power_pellets=not args.no_power_pellets,
        show_flood_fill=not args.no_flood,
        show_nodes=not args.no_nodes,
        show_ghost_paths=not args.no_ghost_paths,
        window_resolution=args.window_resolution
    )

