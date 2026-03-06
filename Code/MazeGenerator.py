import random

# Maze layout: 0 = path, 1 = wall
CLASSIC_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


def generate_maze(use_classic=True, width=20, height=21, algorithm="recursive_backtracking", seed=None):
    if use_classic:
        return [row[:] for row in CLASSIC_MAZE]

    if seed is not None:
        random.seed(seed)

    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    if algorithm == "recursive_backtracking":
        return generate_recursive_backtracking(width, height)
    elif algorithm == "prims":
        return generate_prims(width, height)
    elif algorithm == "random_walk":
        return generate_random_walk(width, height)
    else:
        return generate_recursive_backtracking(width, height)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_cage_zone(width, height):
    """Return the set of (x,y) grid positions that ARE the cage box (no margin).
    Dead-end removal won't touch these tiles at all."""
    center_x = width // 2
    center_y = height // 2
    cage_w, cage_h = 6, 3
    cage_left   = center_x - cage_w // 2
    cage_top    = center_y - cage_h // 2
    cage_right  = cage_left + cage_w - 1
    cage_bottom = cage_top  + cage_h - 1
    zone = set()
    for cy in range(cage_top, cage_bottom + 1):
        for cx in range(cage_left, cage_right + 1):
            zone.add((cx, cy))
    return zone


def _carve_left_half(maze, width, height, algorithm):
    """Run the chosen carving algorithm on the left half only."""
    # +2 so the DFS (which steps by 2) can reach a cell adjacent to centre
    half_width = width // 2 + 2

    if algorithm == "prims":
        start_x = random.randrange(1, half_width, 2)
        start_y = random.randrange(1, height - 1, 2)
        maze[start_y][start_x] = 0
        walls = []
        for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
            nx, ny = start_x + dx, start_y + dy
            if 0 < nx < half_width and 0 < ny < height - 1:
                walls.append((start_x + dx // 2, start_y + dy // 2, nx, ny))
        while walls:
            wall_x, wall_y, next_x, next_y = random.choice(walls)
            walls.remove((wall_x, wall_y, next_x, next_y))
            if maze[next_y][next_x] == 1:
                maze[wall_y][wall_x] = 0
                maze[next_y][next_x] = 0
                for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
                    nx, ny = next_x + dx, next_y + dy
                    if 0 < nx < half_width and 0 < ny < height - 1:
                        if maze[ny][nx] == 1:
                            walls.append((next_x + dx // 2, next_y + dy // 2, nx, ny))
    else:
        # recursive_backtracking / random_walk share the same DFS carver
        start_x, start_y = 1, 1
        maze[start_y][start_x] = 0
        stack   = [(start_x, start_y)]
        visited = {(start_x, start_y)}
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < half_width and 0 < ny < height - 1 and (nx, ny) not in visited:
                    neighbors.append((nx, ny, dx, dy))
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                maze[cy + dy // 2][cx + dx // 2] = 0
                maze[ny][nx] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        if algorithm == "random_walk":
            loop_count = max(5, (width * height) // 30)
            for _ in range(loop_count):
                x = random.randrange(1, half_width - 1)
                y = random.randrange(1, height - 1)
                if maze[y][x] == 1:
                    n = sum(1 for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]
                            if 0 <= x+dx < half_width and 0 <= y+dy < height
                            and maze[y+dy][x+dx] == 0)
                    if n >= 2:
                        maze[y][x] = 0


def _build_maze(width, height, algorithm):
    """Shared pipeline: carve → dead-end removal → cage → mirror → connect halves."""
    maze = [[1]*width for _ in range(height)]

    # Carve left half (includes centre column so carver can reach it)
    _carve_left_half(maze, width, height, algorithm)

    # Remove dead ends BEFORE placing the cage
    cage_zone = _get_cage_zone(width, height)
    eliminate_dead_ends(maze, width, height, cage_zone=cage_zone)

    # Place cage AFTER dead-end removal so its walls are never torn open
    create_ghost_cage(maze, width, height)

    # Mirror left → right
    mirror_maze(maze, width, height)

    # -----------------------------------------------------------------------
    # CRITICAL: punch left-right connector corridors so the two halves are
    # always reachable.  We carve a horizontal open row above the cage and
    # one below it, then make sure each of those rows connects to the rest of
    # the maze on both sides.
    # -----------------------------------------------------------------------
    cl, ct, cr, cb, door_x, door_y = create_ghost_cage.last_bounds

    # Row above cage
    above_y = ct - 1
    if above_y > 0:
        for x in range(1, width - 1):
            maze[above_y][x] = 0
        # Re-stamp cage top wall (mirror may have shifted it)
        for x in range(cl, cr + 1):
            maze[ct][x] = 1
        maze[door_y][door_x] = 2   # restore door

    # Row below cage
    below_y = cb + 1
    if below_y < height - 1:
        for x in range(1, width - 1):
            maze[below_y][x] = 0
        # Re-stamp cage bottom wall
        for x in range(cl, cr + 1):
            maze[cb][x] = 1

    # Ensure the tile directly above the door is open
    if ct - 1 > 0:
        maze[ct - 1][door_x] = 0

    return maze


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def generate_recursive_backtracking(width, height):
    return _build_maze(width, height, "recursive_backtracking")


def generate_prims(width, height):
    return _build_maze(width, height, "prims")


def generate_random_walk(width, height):
    return _build_maze(width, height, "random_walk")


# ---------------------------------------------------------------------------
# Maze operations
# ---------------------------------------------------------------------------

def eliminate_dead_ends(maze, width, height, max_iterations=100, cage_zone=None):
    """Remove dead ends. Skips any tile inside cage_zone."""
    if cage_zone is None:
        cage_zone = set()

    changed = True
    iterations = 0
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if (x, y) in cage_zone:
                    continue
                if maze[y][x] == 0:
                    path_count = sum(
                        1 for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]
                        if 0 <= x+dx < width and 0 <= y+dy < height
                        and maze[y+dy][x+dx] == 0
                    )
                    if path_count == 1:
                        candidates = []
                        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                            nx, ny = x+dx, y+dy
                            if (0 <= nx < width and 0 <= ny < height
                                    and maze[ny][nx] == 1
                                    and (nx, ny) not in cage_zone):
                                wn = sum(
                                    1 for ddx, ddy in [(0,1),(0,-1),(1,0),(-1,0)]
                                    if 0 <= nx+ddx < width and 0 <= ny+ddy < height
                                    and maze[ny+ddy][nx+ddx] == 0
                                )
                                if wn > 0:
                                    candidates.append(((nx, ny), wn))
                        if candidates:
                            candidates.sort(key=lambda c: c[1], reverse=True)
                            wx, wy = candidates[0][0]
                            maze[wy][wx] = 0
                            changed = True


def create_ghost_cage(maze, width, height):
    """
    Stamp a solid-walled cage box in the centre of the maze.
    Always called AFTER eliminate_dead_ends so nothing can re-open the walls.

    Layout  (cage_w=6, cage_h=3):
        top row    : all walls
        middle row : wall | open | open | open | open | wall
        bottom row : all walls
    Door   : centre tile of top wall replaced with tile value 2
    """
    center_x = width  // 2
    center_y = height // 2

    cage_w = 6
    cage_h = 3
    cage_left   = center_x - cage_w // 2
    cage_top    = center_y - cage_h // 2
    cage_right  = cage_left + cage_w - 1
    cage_bottom = cage_top  + cage_h - 1

    # Stamp walls / interior — ignore border constraint (cage is always interior)
    for y in range(cage_top, cage_bottom + 1):
        for x in range(cage_left, cage_right + 1):
            if y == cage_top or y == cage_bottom or x == cage_left or x == cage_right:
                maze[y][x] = 1   # wall
            else:
                maze[y][x] = 0   # open interior

    # Door tile (value=2) at top-centre
    door_x = center_x
    door_y = cage_top
    maze[door_y][door_x] = 2

    # Guarantee the tile above the door is open so ghosts can exit
    if cage_top - 1 > 0:
        maze[cage_top - 1][door_x] = 0

    # Guarantee the tiles left and right of the door on the top row connect
    # to nothing unexpected — ensure top-row neighbours are proper walls
    for x in range(cage_left, cage_right + 1):
        if maze[cage_top][x] == 0 and x != door_x:
            maze[cage_top][x] = 1

    create_ghost_cage.last_bounds = (cage_left, cage_top, cage_right, cage_bottom, door_x, door_y)


create_ghost_cage.last_bounds = None


def mirror_maze(maze, width, height):
    """Mirror left half → right half."""
    center = width // 2
    for y in range(height):
        for x in range(center):
            maze[y][width - 1 - x] = maze[y][x]
