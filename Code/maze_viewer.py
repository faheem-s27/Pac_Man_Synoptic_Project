import sys
import os
import argparse
import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from Code.Maze import Maze
from Code.Pathfinding import Pathfinding, validate_maze_connectivity

# Ghost colors (Blinky, Pinky, Inky, Clyde)
BLINKY_RED = (255, 0, 0)
PINKY_PINK = (255, 184, 255)
INKY_CYAN = (0, 255, 255)
CLYDE_ORANGE = (255, 184, 82)


def _build_maze_grid_from_seed(tile_size, width, height, algorithm, seed):
    """Helper to construct a Maze and return its underlying grid for comparison."""
    m = Maze(tile_size=tile_size, width=width, height=height,
             algorithm=algorithm, seed=seed)
    return m, [row[:] for row in m.maze]


def _compute_spawn_for_flood(maze):
    """Match validate_maze_connectivity's spawn selection.

    Spawn is door_x, cage_bottom+1, with a small vertical search for an
    open/door tile if that exact cell is blocked.
    """
    spawn_gx = getattr(maze, 'door_x', maze.width // 2)
    spawn_gy = getattr(maze, 'cage_bottom', maze.height // 2) + 1

    spawn_gx = max(0, min(maze.width - 1, spawn_gx))
    spawn_gy = max(0, min(maze.height - 1, spawn_gy))

    if getattr(maze, 'maze', None) is None:
        return None

    if maze.maze[spawn_gy][spawn_gx] not in (0, 2):
        found = False
        for dy in range(-3, 4):
            ny = spawn_gy + dy
            if 0 <= ny < maze.height and maze.maze[ny][spawn_gx] in (0, 2):
                spawn_gy = ny
                found = True
                break
        if not found:
            return None

    return spawn_gx, spawn_gy


def _compute_flood_reachable(maze):
    """Run the same strict flood fill as validate_maze_connectivity and
    return the set of reachable tiles from spawn.
    """
    start = _compute_spawn_for_flood(maze)
    if start is None:
        return set()

    sx, sy = start

    visited = set()
    queue = [(sx, sy)]

    while queue:
        cx, cy = queue.pop(0)
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < maze.width and 0 <= ny < maze.height:
                if maze.maze[ny][nx] in (0, 2) and (nx, ny) not in visited:
                    queue.append((nx, ny))

    return visited


def _get_spawn_and_corners(maze):
    """Return Pac-Man spawn grid position and the four corner targets for
    visual path overlays (separate from flood-fill)."""
    # Spawn: one row below cage bottom, same column as door, with fallback search
    spawn_gx = getattr(maze, 'door_x', maze.width // 2)
    spawn_gy = getattr(maze, 'cage_bottom', maze.height // 2) + 1

    spawn_gx = max(0, min(maze.width - 1, spawn_gx))
    spawn_gy = max(0, min(maze.height - 1, spawn_gy))

    if maze.maze[spawn_gy][spawn_gx] not in (0, 2):
        found = False
        for dy in range(-3, 4):
            ny = spawn_gy + dy
            if 0 <= ny < maze.height and maze.maze[ny][spawn_gx] in (0, 2):
                spawn_gy = ny
                found = True
                break
        if not found:
            # No valid spawn found; still return something, but paths will be empty
            pass

    # Corner targets (inset from borders)
    corners = [
        (1, 1),
        (maze.width - 2, 1),
        (1, maze.height - 2),
        (maze.width - 2, maze.height - 2),
    ]

    return (spawn_gx, spawn_gy), corners


def run_viewer(seed=None, algorithm="recursive_backtracking", tile_size=40,
               show_pellets=True, show_power_pellets=True,
               show_flood_fill=True):
    pygame.init()

    # Base maze used for rendering
    maze, base_grid = _build_maze_grid_from_seed(
        tile_size=tile_size,
        width=20,
        height=21,
        algorithm=algorithm,
        seed=seed,
    )

    # Determine seed validity according to strict flood-fill validator
    seed_status = "N/A"
    if seed is not None:
        try:
            _, test_grid = _build_maze_grid_from_seed(
                tile_size=tile_size,
                width=20,
                height=21,
                algorithm=algorithm,
                seed=seed,
            )
            if base_grid == test_grid:
                seed_status = "VALID" if validate_maze_connectivity(maze) else "INVALID (flood-fill unreachable tiles)"
            else:
                seed_status = "INPUT SEED BANNED → viewer showing resampled layout"
        except Exception:
            seed_status = "ERROR WHEN RECHECKING SEED"

    if seed is None:
        # test if its valid
        try:
            if validate_maze_connectivity(maze):
                seed_status = "VALID (randomly generated)"
            else:
                seed_status = "INVALID (flood-fill unreachable tiles)"
        except Exception:
            seed_status = "ERROR WHEN VALIDATING RANDOM SEED"

    width_px = maze.width * tile_size
    height_px = maze.height * tile_size

    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption(f"Maze Viewer - seed={seed} [{seed_status}]")
    clock = pygame.time.Clock()

    pellets = []

    # Safely extract cage boundaries
    cage_left = getattr(maze, 'cage_left', -1)
    cage_right = getattr(maze, 'cage_right', -1)
    cage_top = getattr(maze, 'cage_top', -1)
    cage_bottom = getattr(maze, 'cage_bottom', -1)

    if show_pellets or show_power_pellets:
        # Simplified pellet placement with strict cage exclusion
        for y in range(maze.height):
            for x in range(maze.width):
                if maze.maze[y][x] == 0:
                    in_cage = (cage_left <= x <= cage_right) and (cage_top <= y <= cage_bottom)
                    if not in_cage:
                        pellets.append((x * tile_size + tile_size // 2,
                                        y * tile_size + tile_size // 2))

    # Precompute pathfinding paths from spawn to each corner
    spawn_g, corners = _get_spawn_and_corners(maze)
    pf = Pathfinding(maze)
    paths = []
    corner_colors = [BLINKY_RED, PINKY_PINK, INKY_CYAN, CLYDE_ORANGE]

    for (tx, ty), color in zip(corners, corner_colors):
        if 0 <= tx < maze.width and 0 <= ty < maze.height:
            path = pf.find_shortest_path(spawn_g[0], spawn_g[1], tx, ty,
                                         current_dir=(0, 0))
        else:
            path = []
        paths.append((path, color))

    # Precompute flood-fill reachable tiles (strict validator view)
    flood_reachable = _compute_flood_reachable(maze) if show_flood_fill else set()

    running = True
    font = pygame.font.SysFont(None, 20)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.fill((0, 0, 0))
        maze.draw(screen)

        # Optional flood-fill overlay (semi-transparent blue on reachable tiles)
        if show_flood_fill and flood_reachable:
            overlay = pygame.Surface((width_px, height_px), pygame.SRCALPHA)
            for gx, gy in flood_reachable:
                px = gx * tile_size
                py = gy * tile_size
                pygame.draw.rect(overlay, (0, 0, 255, 60), pygame.Rect(px, py, tile_size, tile_size))
            screen.blit(overlay, (0, 0))

        # Draw pellets
        if show_pellets:
            for px, py in pellets:
                pygame.draw.circle(screen, (255, 255, 0), (px, py), max(2, tile_size // 8))

        # Power pellets optional: for now, use same as pellets but larger and different color
        if show_power_pellets:
            pp_color = (0, 255, 255)
            corner_tiles = [
                (1, 1),
                (maze.width - 2, 1),
                (1, maze.height - 2),
                (maze.width - 2, maze.height - 2),
            ]
            for gx, gy in corner_tiles:
                if 0 <= gx < maze.width and 0 <= gy < maze.height and maze.maze[gy][gx] == 0:
                    px = gx * tile_size + tile_size // 2
                    py = gy * tile_size + tile_size // 2
                    pygame.draw.circle(screen, pp_color, (px, py), max(4, tile_size // 6))

        # Draw pathfinding paths from spawn to each corner in ghost colors
        for path, color in paths:
            if not path:
                continue
            pts = []
            for gx, gy in path:
                px = gx * tile_size + tile_size // 2
                py = gy * tile_size + tile_size // 2
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(screen, color, False, pts, max(2, tile_size // 20))
            tx, ty = path[-1]
            tx_px = tx * tile_size + tile_size // 2
            ty_px = ty * tile_size + tile_size // 2
            pygame.draw.circle(screen, color, (tx_px, ty_px), max(3, tile_size // 10))

        # Highlight the spawn point (same as flood-fill start)
        spawn_for_flood = _compute_spawn_for_flood(maze)
        if spawn_for_flood is not None:
            sx, sy = spawn_for_flood
        else:
            sx, sy = spawn_g
        sx_px = sx * tile_size + tile_size // 2
        sy_px = sy * tile_size + tile_size // 2
        pygame.draw.circle(screen, (255, 255, 255), (sx_px, sy_px), max(4, tile_size // 9), 1)

        # Overlay seed status text at top-left
        if seed is not None:
            text = f"Seed: {seed} | Status: {seed_status}"
        else:
            text = f"Seed: random | Status: {seed_status}"
        text_surf = font.render(text, True, (255, 255, 255))
        screen.blit(text_surf, (5, 5))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    SEED = 2041178492
    parser = argparse.ArgumentParser(description="Standalone maze viewer for inspecting generated layouts/seeds.")
    parser.add_argument("--seed", type=int, default=SEED, help="Maze seed to visualize.")
    parser.add_argument("--algorithm", type=str, default="recursive_backtracking",
                        choices=["recursive_backtracking", "prims", "random_walk"],
                        help="Maze generation algorithm.")
    parser.add_argument("--tile-size", type=int, default=40, help="Tile size in pixels.")
    parser.add_argument("--no-pellets", action="store_true", help="Hide regular pellets.")
    parser.add_argument("--no-power-pellets", action="store_true", help="Hide power pellets overlay.")
    parser.add_argument("--no-flood", action="store_true", help="Disable flood-fill debug overlay.")

    args = parser.parse_args()

    run_viewer(
        seed=args.seed,
        algorithm=args.algorithm,
        tile_size=args.tile_size,
        show_pellets=not args.no_pellets,
        show_power_pellets=not args.no_power_pellets,
        show_flood_fill=not args.no_flood,
    )
