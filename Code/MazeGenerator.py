import random

# Maze layout: 0 = path, 1 = wall
# Note: Rows 10 have 0s on the edges (positions 0 and 19) for teleport tunnels
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
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Teleport row
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


def generate_maze(use_classic=True, width=20, height=21, algorithm="recursive_backtracking"):
    """
    Generate a maze layout using various algorithms.

    Args:
        use_classic: If True, return the classic maze
        width: Width of maze in tiles (must be odd)
        height: Height of maze in tiles (must be odd)
        algorithm: "recursive_backtracking", "prims", or "random_walk"

    Returns:
        2D list representing the maze (0 = path, 1 = wall)
    """
    if use_classic:
        return CLASSIC_MAZE

    # Ensure odd dimensions for proper maze generation
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


def generate_recursive_backtracking(width, height):
    """
    Generate maze using recursive backtracking algorithm.
    Creates a perfect maze with no loops.
    Symmetrical layout with ghost cage in center.
    """
    # Initialize maze with all walls
    maze = [[1 for _ in range(width)] for _ in range(height)]

    # Mark borders as walls
    for x in range(width):
        maze[0][x] = 1
        maze[height - 1][x] = 1
    for y in range(height):
        maze[y][0] = 1
        maze[y][width - 1] = 1

    # Only generate left half, then mirror it for symmetry
    half_width = width // 2 + 1

    # Carve starting cell (top-left area)
    start_x, start_y = 1, 1
    maze[start_y][start_x] = 0

    # Stack for backtracking
    stack = [(start_x, start_y)]
    visited = {(start_x, start_y)}

    while stack:
        cx, cy = stack[-1]

        # Get unvisited neighbors (2 cells away to maintain wall structure)
        neighbors = []
        for dx, dy, _ in [(0, -2, 'U'), (2, 0, 'R'), (0, 2, 'D'), (-2, 0, 'L')]:
            nx, ny = cx + dx, cy + dy
            if (0 < nx < half_width and 0 < ny < height - 1 and
                (nx, ny) not in visited):
                neighbors.append((nx, ny, dx, dy))

        if neighbors:
            # Choose random neighbor
            nx, ny, dx, dy = random.choice(neighbors)

            # Carve path between current and neighbor
            maze[cy + dy // 2][cx + dx // 2] = 0
            maze[ny][nx] = 0

            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            # Backtrack
            stack.pop()

    # Create ghost cage in center
    create_ghost_cage(maze, width, height)

    # Mirror the left half to right half for symmetry
    mirror_maze(maze, width, height)

    # Add teleport tunnels for wraparound movement
    create_teleport_tunnels(maze, width, height)

    # Add strategic loops to avoid dead ends
    eliminate_dead_ends(maze, width, height)

    return maze


def generate_prims(width, height):
    """
    Generate maze using Prim's algorithm.
    Creates a maze with natural-looking randomness.
    Symmetrical layout with ghost cage in center.
    """
    # Initialize maze with all walls
    maze = [[1 for _ in range(width)] for _ in range(height)]

    # Mark borders as walls
    for x in range(width):
        maze[0][x] = 1
        maze[height - 1][x] = 1
    for y in range(height):
        maze[y][0] = 1
        maze[y][width - 1] = 1

    # Only generate left half, then mirror it for symmetry
    half_width = width // 2 + 1

    # Start from random odd cell in left half
    start_x = random.randrange(1, half_width, 2)
    start_y = random.randrange(1, height - 1, 2)
    maze[start_y][start_x] = 0

    # Walls list contains walls that border the maze
    walls = []

    # Add initial walls around starting cell
    for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
        nx, ny = start_x + dx, start_y + dy
        if 0 < nx < half_width and 0 < ny < height - 1:
            walls.append((start_x + dx // 2, start_y + dy // 2, nx, ny))

    while walls:
        # Pick random wall
        wall_x, wall_y, next_x, next_y = random.choice(walls)
        walls.remove((wall_x, wall_y, next_x, next_y))

        if maze[next_y][next_x] == 1:
            # Carve passage
            maze[wall_y][wall_x] = 0
            maze[next_y][next_x] = 0

            # Add neighboring walls
            for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
                nx, ny = next_x + dx, next_y + dy
                if 0 < nx < half_width and 0 < ny < height - 1:
                    if maze[ny][nx] == 1:
                        walls.append((next_x + dx // 2, next_y + dy // 2, nx, ny))

    # Create ghost cage in center
    create_ghost_cage(maze, width, height)

    # Mirror the left half to right half for symmetry
    mirror_maze(maze, width, height)

    # Add teleport tunnels for wraparound movement
    create_teleport_tunnels(maze, width, height)

    # Add strategic loops to avoid dead ends
    eliminate_dead_ends(maze, width, height)

    return maze


def generate_random_walk(width, height):
    """
    Generate maze using random walk algorithm.
    Creates interesting mazes with multiple loops.
    Symmetrical layout with ghost cage in center.
    """
    # Initialize maze with all walls
    maze = [[1 for _ in range(width)] for _ in range(height)]

    # Mark borders as walls
    for x in range(width):
        maze[0][x] = 1
        maze[height - 1][x] = 1
    for y in range(height):
        maze[y][0] = 1
        maze[y][width - 1] = 1

    # Only generate left half, then mirror it for symmetry
    half_width = width // 2 + 1

    # Start carving from top-left
    start_x, start_y = 1, 1
    maze[start_y][start_x] = 0

    # Random walk generation on left half
    stack = [(start_x, start_y)]
    visited = {(start_x, start_y)}

    while stack:
        cx, cy = stack[-1]

        # Get unvisited neighbors (2 cells away to maintain wall structure)
        neighbors = []
        for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
            nx, ny = cx + dx, cy + dy
            if (0 < nx < half_width and 0 < ny < height - 1 and
                (nx, ny) not in visited):
                neighbors.append((nx, ny, dx, dy))

        if neighbors:
            # Choose random neighbor
            nx, ny, dx, dy = random.choice(neighbors)

            # Carve path between current and neighbor
            maze[cy + dy // 2][cx + dx // 2] = 0
            maze[ny][nx] = 0

            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            # Backtrack
            stack.pop()

    # Add more loops than recursive backtracking for variety
    loop_count = max(5, (width * height) // 30)
    for _ in range(loop_count):
        x = random.randrange(1, half_width - 1)
        y = random.randrange(1, height - 1)

        if maze[y][x] == 1:
            # Count path neighbors
            neighbors = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < half_width and 0 <= ny < height and maze[ny][nx] == 0:
                    neighbors += 1

            # Remove wall if it connects 2+ paths
            if neighbors >= 2:
                maze[y][x] = 0

    # Create ghost cage in center
    create_ghost_cage(maze, width, height)

    # Mirror the left half to right half for symmetry
    mirror_maze(maze, width, height)

    # Add teleport tunnels for wraparound movement
    create_teleport_tunnels(maze, width, height)

    # Add strategic loops to avoid dead ends
    eliminate_dead_ends(maze, width, height)

    return maze


def eliminate_dead_ends(maze, width, height, max_iterations=100):
    """
    Remove dead ends by opening walls to create alternative paths.
    This ensures ghosts cannot trap Pac-Man.
    """
    changed = True
    iterations = 0

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if maze[y][x] == 0:
                    # Count path neighbors
                    path_count = 0
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 0:
                            path_count += 1

                    # If dead end (only 1 neighbor), find best wall to open
                    if path_count == 1:
                        candidates = []
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < width and 0 <= ny < height and
                                maze[ny][nx] == 1):
                                # Count paths this wall connects to
                                wall_neighbors = 0
                                for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                    nnx, nny = nx + ddx, ny + ddy
                                    if 0 <= nnx < width and 0 <= nny < height and maze[nny][nnx] == 0:
                                        wall_neighbors += 1

                                if wall_neighbors > 0:
                                    candidates.append(((nx, ny), wall_neighbors))

                        if candidates:
                            # Open wall connecting most paths
                            candidates.sort(key=lambda x: x[1], reverse=True)
                            (wall_x, wall_y), _ = candidates[0]
                            maze[wall_y][wall_x] = 0
                            changed = True


def create_ghost_cage(maze, width, height):
    """
    Create a ghost cage in the center of the maze.
    A small area where ghosts can spawn without blocking off maze sections.
    Uses a minimal design to prevent unreachable areas.
    """
    # Calculate center position
    center_x = width // 2
    center_y = height // 2

    # Create a smaller 2x2 cage in the center
    cage_width = 4
    cage_height = 2
    cage_left = center_x - cage_width // 2
    cage_top = center_y - cage_height // 2

    # Clear the cage area (make it paths)
    for y in range(cage_top, cage_top + cage_height):
        for x in range(cage_left, cage_left + cage_width):
            if 0 < x < width - 1 and 0 < y < height - 1:
                maze[y][x] = 0


def create_teleport_tunnels(maze, width, height):
    """
    Create teleport tunnels on the left and right edges of the maze.
    Opens a horizontal corridor at the middle height for wraparound movement.
    """
    # Find middle row
    teleport_row = height // 2

    # Ensure the entire row is a path (clear any walls)
    for x in range(width):
        maze[teleport_row][x] = 0

    # Make sure there's a wall in the center to separate the two tunnels
    center_x = width // 2
    maze[teleport_row][center_x] = 1

    # Open the edges explicitly (positions 0 and width-1)
    maze[teleport_row][0] = 0
    maze[teleport_row][width - 1] = 0

    # Connect the teleport row to surrounding paths
    # Clear a few cells before and after to ensure connectivity
    for x in range(1, 7):  # Left side connection
        maze[teleport_row][x] = 0
    for x in range(width - 7, width - 1):  # Right side connection
        maze[teleport_row][x] = 0




def mirror_maze(maze, width, height):
    """
    Mirror the left half of the maze to the right half for symmetry.
    Preserves the center column.
    """
    center = width // 2

    # Mirror from left to right
    for y in range(height):
        for x in range(center):
            # Mirror x position on the right side
            mirror_x = width - 1 - x
            maze[y][mirror_x] = maze[y][x]


