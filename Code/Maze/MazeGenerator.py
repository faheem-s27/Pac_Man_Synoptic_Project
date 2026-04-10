import random
from Code.Engine.Pathfinding import validate_maze_connectivity


def generate_maze(width=20, height=21,
                  algorithm="recursive_backtracking", seed=None,
                  validate=True, max_attempts=100):
    """Generate a maze, optionally validating connectivity via flood-fill."""
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    def _build_with_seed(s):
        # Use isolated RNG state so generation does not mutate global random state.
        rng = random.Random(s)

        if algorithm == "recursive_backtracking":
            return generate_recursive_backtracking(width, height, rng=rng)
        elif algorithm == "prims":
            return generate_prims(width, height, rng=rng)
        elif algorithm == "random_walk":
            return generate_random_walk(width, height, rng=rng)
        else:
            return generate_recursive_backtracking(width, height, rng=rng)

    if not validate:
        return _build_with_seed(seed)

    attempts = 0
    seed_rng = random.Random(seed) if seed is not None else random.Random()
    current_seed = seed if seed is not None else seed_rng.randint(0, 2**31 - 1)
    best_maze = None

    while attempts < max_attempts:
        attempts += 1
        raw_maze = _build_with_seed(current_seed)

        class _TempMaze:
            def __init__(self, grid):
                self.maze = grid
                self.height = len(grid)
                self.width = len(grid[0]) if self.height > 0 else 0

                center_x = self.width // 2
                center_y = self.height // 2
                cage_w = 5
                cage_h = 3

                self.cage_left = center_x - cage_w // 2
                self.cage_top = center_y - cage_h // 2
                self.cage_right = self.cage_left + cage_w - 1
                self.cage_bottom = self.cage_top + cage_h - 1
                self.door_x = center_x
                self.door_y = self.cage_top

        temp_maze = _TempMaze(raw_maze)

        if validate_maze_connectivity(temp_maze):
            return raw_maze

        best_maze = raw_maze
        current_seed = (current_seed + 1) % (2**31 - 1)

    print("[MazeValidator] Max attempts reached; returning last generated maze (may not be fully connected).")
    return best_maze if best_maze is not None else _build_with_seed(seed)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _is_open_tile(v):
    return v in (0, 2)


def _open_neighbor_count(maze, x, y, width, height):
    return sum(
        1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if 0 <= x + dx < width and 0 <= y + dy < height
        and _is_open_tile(maze[y + dy][x + dx])
    )


def _get_cage_zone(width, height):
    center_x = width // 2
    center_y = height // 2
    cage_w, cage_h = 5, 3
    cage_left = center_x - cage_w // 2
    cage_top = center_y - cage_h // 2
    cage_right = cage_left + cage_w - 1
    cage_bottom = cage_top + cage_h - 1

    zone = set()
    for cy in range(cage_top, cage_bottom + 1):
        for cx in range(cage_left, cage_right + 1):
            zone.add((cx, cy))
    return zone


def _get_cage_bounds(width, height):
    center_x = width // 2
    center_y = height // 2
    cage_w, cage_h = 5, 3
    cage_left = center_x - cage_w // 2
    cage_top = center_y - cage_h // 2
    cage_right = cage_left + cage_w - 1
    cage_bottom = cage_top + cage_h - 1
    door_x = center_x
    door_y = cage_top
    return cage_left, cage_top, cage_right, cage_bottom, door_x, door_y


def _carve_left_half(maze, width, height, algorithm, rng):
    """Run the chosen carving algorithm on the left half only."""
    half_width = width // 2 + 2

    if algorithm == "prims":
        start_x = rng.randrange(1, half_width, 2)
        start_y = rng.randrange(1, height - 1, 2)
        maze[start_y][start_x] = 0
        walls = []

        for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
            nx, ny = start_x + dx, start_y + dy
            if 0 < nx < half_width and 0 < ny < height - 1:
                walls.append((start_x + dx // 2, start_y + dy // 2, nx, ny))

        while walls:
            wall_x, wall_y, next_x, next_y = rng.choice(walls)
            walls.remove((wall_x, wall_y, next_x, next_y))

            if maze[next_y][next_x] == 1:
                maze[wall_y][wall_x] = 0
                maze[next_y][next_x] = 0

                for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
                    nx, ny = next_x + dx, next_y + dy
                    if 0 < nx < half_width and 0 < ny < height - 1 and maze[ny][nx] == 1:
                        walls.append((next_x + dx // 2, next_y + dy // 2, nx, ny))
    else:
        start_x, start_y = 1, 1
        maze[start_y][start_x] = 0
        stack = [(start_x, start_y)]
        visited = {(start_x, start_y)}

        while stack:
            cx, cy = stack[-1]
            neighbors = []

            for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < half_width and 0 < ny < height - 1 and (nx, ny) not in visited:
                    neighbors.append((nx, ny, dx, dy))

            if neighbors:
                nx, ny, dx, dy = rng.choice(neighbors)
                maze[cy + dy // 2][cx + dx // 2] = 0
                maze[ny][nx] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        if algorithm == "random_walk":
            loop_count = max(5, (width * height) // 40)
            for _ in range(loop_count):
                x = rng.randrange(1, half_width - 1)
                y = rng.randrange(1, height - 1)
                if maze[y][x] == 1:
                    n = _open_neighbor_count(maze, x, y, half_width, height)
                    if n >= 2:
                        maze[y][x] = 0


def mirror_maze(maze, width, height):
    """Mirror left half -> right half."""
    center = width // 2
    for y in range(height):
        for x in range(center):
            maze[y][width - 1 - x] = maze[y][x]


def _choose_bridge_rows(height, cage_top, cage_bottom, count, rng):
    candidates = []

    for y in [cage_top - 1, cage_bottom + 1, cage_top - 3, cage_bottom + 3]:
        if 1 <= y < height - 1:
            candidates.append(y)

    seen = set()
    ordered = []
    for y in candidates:
        if y not in seen:
            seen.add(y)
            ordered.append(y)

    if len(ordered) <= count:
        return ordered

    primary = []
    if cage_top - 1 in ordered:
        primary.append(cage_top - 1)
    if cage_bottom + 1 in ordered:
        primary.append(cage_bottom + 1)

    remaining = [y for y in ordered if y not in primary]

    if count <= len(primary):
        return primary[:count]

    need = count - len(primary)
    chosen_extra = rng.sample(remaining, min(need, len(remaining)))
    return sorted(primary + chosen_extra)


def _carve_straight_tunnel(maze, x1, y1, x2, y2):
    """Carve a straight 1-tile-wide horizontal or vertical tunnel."""
    if x1 == x2:
        step = 1 if y2 >= y1 else -1
        for y in range(y1, y2 + step, step):
            maze[y][x1] = 0
    elif y1 == y2:
        step = 1 if x2 >= x1 else -1
        for x in range(x1, x2 + step, step):
            maze[y1][x] = 0


def _add_center_bridges(maze, width, height, cage_bounds, bridge_count=3, rng=None):
    """
    Add 2 or 3 narrow centre bridge corridors.
    These are strictly 1-tile-wide corridors, not open bands.
    """
    if rng is None:
        rng = random

    cl, ct, cr, cb, door_x, door_y = cage_bounds
    bridge_rows = _choose_bridge_rows(height, ct, cb, bridge_count, rng)

    left_anchor = max(1, cl - 3)
    right_anchor = min(width - 2, cr + 3)

    for y in bridge_rows:
        if y <= 0 or y >= height - 1:
            continue

        # Single 1-tile-wide horizontal connector across the centre
        _carve_straight_tunnel(maze, left_anchor, y, right_anchor, y)

        # Vertical stubs to tie it into the maze without creating open areas
        if y < ct:
            _carve_straight_tunnel(maze, left_anchor, y, left_anchor, max(1, y - 2))
            _carve_straight_tunnel(maze, right_anchor, y, right_anchor, max(1, y - 2))
        else:
            _carve_straight_tunnel(maze, left_anchor, y, left_anchor, min(height - 2, y + 2))
            _carve_straight_tunnel(maze, right_anchor, y, right_anchor, min(height - 2, y + 2))

    # Keep cage intact
    create_ghost_cage(maze, width, height)

    # Ensure the tile above the door is open
    if ct - 1 > 0:
        maze[ct - 1][door_x] = 0


def create_ghost_cage(maze, width, height):
    """
    Stamp a solid-walled cage box in the centre of the maze.

    Layout (cage_w=5, cage_h=3):
        top row    : all walls except door
        middle row : wall | open | open | open | wall
        bottom row : all walls
    """
    center_x = width // 2
    center_y = height // 2

    cage_w = 5
    cage_h = 3
    cage_left = center_x - cage_w // 2
    cage_top = center_y - cage_h // 2
    cage_right = cage_left + cage_w - 1
    cage_bottom = cage_top + cage_h - 1

    for y in range(cage_top, cage_bottom + 1):
        for x in range(cage_left, cage_right + 1):
            if y == cage_top or y == cage_bottom or x == cage_left or x == cage_right:
                maze[y][x] = 1
            else:
                maze[y][x] = 0

    door_x = center_x
    door_y = cage_top
    maze[door_y][door_x] = 2

    if cage_top - 1 > 0:
        maze[cage_top - 1][door_x] = 0

    for x in range(cage_left, cage_right + 1):
        if maze[cage_top][x] == 0 and x != door_x:
            maze[cage_top][x] = 1

    bounds = (cage_left, cage_top, cage_right, cage_bottom, door_x, door_y)
    # Backward compatibility: legacy code reads this mutable function attribute.
    create_ghost_cage.last_bounds = bounds
    return bounds


# Backward compatibility default for callers that inspect before first cage stamp.
create_ghost_cage.last_bounds = None


# ---------------------------------------------------------------------------
# Corridor-shape enforcement
# ---------------------------------------------------------------------------

def eliminate_dead_ends(maze, width, height, max_iterations=100, cage_zone=None):
    """
    Remove dead ends by opening side walls, but never carve inside the cage zone.
    """
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
                if not _is_open_tile(maze[y][x]):
                    continue

                path_count = _open_neighbor_count(maze, x, y, width, height)
                if path_count != 1:
                    continue

                candidates = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if (nx, ny) in cage_zone:
                        continue
                    if maze[ny][nx] != 1:
                        continue

                    wn = _open_neighbor_count(maze, nx, ny, width, height)
                    if wn > 0:
                        candidates.append((wn, nx, ny))

                if candidates:
                    candidates.sort(reverse=True)
                    _, wx, wy = candidates[0]
                    maze[wy][wx] = 0
                    changed = True


def enforce_no_2x2_open_areas(maze, width, height, cage_zone=None, protected=None, max_passes=12):
    """
    Ensure there are no 2x2 fully-open blocks anywhere outside protected tiles.
    This is the key fix for 'corridors only' layout.
    """
    if cage_zone is None:
        cage_zone = set()
    if protected is None:
        protected = set()

    for _ in range(max_passes):
        changed = False

        for y in range(height - 1):
            for x in range(width - 1):
                block = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

                if all(_is_open_tile(maze[by][bx]) for bx, by in block):
                    candidates = []
                    for bx, by in block:
                        if (bx, by) in cage_zone or (bx, by) in protected:
                            continue
                        if maze[by][bx] == 2:
                            continue

                        degree = _open_neighbor_count(maze, bx, by, width, height)
                        # Prefer sealing tiles that are not critical junctions
                        candidates.append((degree, bx, by))

                    if not candidates:
                        continue

                    candidates.sort()
                    _, sx, sy = candidates[0]
                    maze[sy][sx] = 1
                    changed = True

        if not changed:
            break


def _trace_dead_end_corridor_length(maze, x, y, width, height):
    if not _is_open_tile(maze[y][x]):
        return None
    if _open_neighbor_count(maze, x, y, width, height) != 1:
        return None

    prev = None
    cur = (x, y)
    distance = 0

    while True:
        cx, cy = cur
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height and _is_open_tile(maze[ny][nx]):
                if prev is None or (nx, ny) != prev:
                    neighbors.append((nx, ny))

        total_degree = _open_neighbor_count(maze, cx, cy, width, height)

        if total_degree >= 3:
            return distance
        if len(neighbors) == 0:
            return distance
        if len(neighbors) > 1:
            return distance

        prev = cur
        cur = neighbors[0]
        distance += 1

        if distance > width * height:
            return distance


def refine_recursive_backtracking_layout(
    maze,
    width,
    height,
    cage_zone=None,
    passes=12,
    min_loop_distance=2,
    max_probe=8,
    min_dead_end_corridor_len=3
):
    """
    Recursive-backtracking-only refinement.

    Goals:
    - reduce dead ends
    - create loops deeper in corridors
    - keep corridors narrow
    """
    if cage_zone is None:
        cage_zone = set()

    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def in_bounds(x, y):
        return 0 < x < width - 1 and 0 < y < height - 1

    def is_open(x, y):
        return in_bounds(x, y) and _is_open_tile(maze[y][x])

    def open_neighbors(x, y):
        out = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if is_open(nx, ny):
                out.append((nx, ny))
        return out

    def find_dead_ends():
        result = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if (x, y) in cage_zone:
                    continue
                if not is_open(x, y):
                    continue
                if len(open_neighbors(x, y)) != 1:
                    continue

                length = _trace_dead_end_corridor_length(maze, x, y, width, height)
                if length is not None:
                    result.append((length, x, y))
        result.sort()
        return result

    def trace_corridor(start_x, start_y):
        corridor = [(start_x, start_y)]
        prev = None
        cur = (start_x, start_y)

        while True:
            cx, cy = cur
            nbrs = open_neighbors(cx, cy)
            if prev is not None:
                nbrs = [p for p in nbrs if p != prev]

            if len(nbrs) != 1:
                break

            nxt = nbrs[0]
            corridor.append(nxt)
            prev = cur
            cur = nxt

            nx, ny = cur
            if len(open_neighbors(nx, ny)) != 2:
                break

        return corridor

    def best_side_connection(corridor):
        corridor_set = set(corridor)
        best = None

        for depth in range(len(corridor) - 1, -1, -1):
            x, y = corridor[depth]

            corridor_dirs = set()
            for nx, ny in open_neighbors(x, y):
                if (nx, ny) in corridor_set:
                    corridor_dirs.add((nx - x, ny - y))

            for dx, dy in dirs:
                if (dx, dy) in corridor_dirs or (-dx, -dy) in corridor_dirs:
                    continue

                walls_to_carve = []

                for step in range(1, max_probe + 1):
                    px, py = x + dx * step, y + dy * step

                    if not in_bounds(px, py):
                        break
                    if (px, py) in cage_zone:
                        break

                    if maze[py][px] == 1:
                        walls_to_carve.append((px, py))
                        continue

                    if _is_open_tile(maze[py][px]):
                        if (px, py) in corridor_set:
                            break
                        if len(walls_to_carve) >= min_loop_distance:
                            score = (depth, len(walls_to_carve), -step)
                            candidate = (score, walls_to_carve)
                            if best is None or candidate[0] > best[0]:
                                best = candidate
                        break

                    break

        return best[1] if best is not None else None

    for _ in range(passes):
        dead_ends = find_dead_ends()
        if not dead_ends:
            break

        changed = False

        for length, dx, dy in dead_ends:
            if length >= min_dead_end_corridor_len:
                continue

            corridor = trace_corridor(dx, dy)
            carve_path = best_side_connection(corridor)

            if carve_path:
                for cx, cy in carve_path:
                    maze[cy][cx] = 0
                changed = True

        if not changed:
            break


def _collect_wall_component(maze, sx, sy, width, height, half_limit, visited):
    stack = [(sx, sy)]
    component = set()
    visited.add((sx, sy))

    while stack:
        x, y = stack.pop()
        component.add((x, y))
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (1 <= nx <= half_limit and 1 <= ny < height - 1):
                continue
            if (nx, ny) in visited:
                continue
            if maze[ny][nx] != 1:
                continue
            visited.add((nx, ny))
            stack.append((nx, ny))

    return component


def _find_bridgeable_wall_segment_on_row(maze, component, y, min_len=4):
    row_x = sorted(x for (x, ry) in component if ry == y)
    if not row_x:
        return None

    segments = []
    start = row_x[0]
    prev = row_x[0]
    for x in row_x[1:]:
        if x == prev + 1:
            prev = x
            continue
        segments.append((start, prev))
        start = x
        prev = x
    segments.append((start, prev))

    candidates = []
    for sx, ex in segments:
        if (ex - sx + 1) < min_len:
            continue
        if not _is_open_tile(maze[y][sx - 1]):
            continue
        if not _is_open_tile(maze[y][ex + 1]):
            continue
        candidates.append((sx, ex))

    if not candidates:
        return None
    return max(candidates, key=lambda p: p[1] - p[0])


def _find_bridgeable_wall_segment_on_col(maze, component, x, min_len=4):
    col_y = sorted(y for (cx, y) in component if cx == x)
    if not col_y:
        return None

    segments = []
    start = col_y[0]
    prev = col_y[0]
    for y in col_y[1:]:
        if y == prev + 1:
            prev = y
            continue
        segments.append((start, prev))
        start = y
        prev = y
    segments.append((start, prev))

    candidates = []
    for sy, ey in segments:
        if (ey - sy + 1) < min_len:
            continue
        if not _is_open_tile(maze[sy - 1][x]):
            continue
        if not _is_open_tile(maze[ey + 1][x]):
            continue
        candidates.append((sy, ey))

    if not candidates:
        return None
    return max(candidates, key=lambda p: p[1] - p[0])


def slice_large_wall_blocks(
    maze,
    width,
    height,
    cage_zone=None,
    protected=None,
    min_span=5,
    min_area=24,
    min_cut_len=4,
):
    """
    Slice oversized interior wall masses by carving bridge corridors through them.
    Runs on left half before mirroring so symmetry is preserved automatically.
    """
    if cage_zone is None:
        cage_zone = set()
    if protected is None:
        protected = set()

    half_limit = width // 2
    visited = set()

    for y in range(1, height - 1):
        for x in range(1, half_limit + 1):
            if (x, y) in visited:
                continue
            if maze[y][x] != 1:
                continue

            component = _collect_wall_component(maze, x, y, width, height, half_limit, visited)
            if not component:
                continue

            if any((cx, cy) in cage_zone or (cx, cy) in protected for (cx, cy) in component):
                continue

            xs = [cx for cx, _ in component]
            ys = [cy for _, cy in component]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = max_x - min_x + 1
            span_y = max_y - min_y + 1

            # Ignore border-touching and already-narrow wall structures.
            if min_x <= 1 or max_x >= half_limit or min_y <= 1 or max_y >= height - 2:
                continue
            if span_x < min_span and span_y < min_span:
                continue
            if len(component) < min_area:
                continue

            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            did_cut = False

            row_candidates = sorted(
                range(min_y + 1, max_y),
                key=lambda ry: abs(ry - center_y)
            )
            for ry in row_candidates:
                seg = _find_bridgeable_wall_segment_on_row(maze, component, ry, min_len=min_cut_len)
                if seg is None:
                    continue
                sx, ex = seg
                for cx in range(sx, ex + 1):
                    maze[ry][cx] = 0
                did_cut = True
                break

            col_candidates = sorted(
                range(min_x + 1, max_x),
                key=lambda cx: abs(cx - center_x)
            )
            for cx in col_candidates:
                seg = _find_bridgeable_wall_segment_on_col(maze, component, cx, min_len=min_cut_len)
                if seg is None:
                    continue
                sy, ey = seg
                for cy in range(sy, ey + 1):
                    maze[cy][cx] = 0
                did_cut = True
                break

            if did_cut:
                # Refresh nearby topology so later passes see a clean corridor graph.
                eliminate_dead_ends(maze, width, height, max_iterations=2, cage_zone=cage_zone)


def _protected_corridor_tiles(width, height):
    """
    Tiles that should not be sealed by the 2x2 cleanup.
    Mainly the ghost-door path immediately above the cage.
    """
    cl, ct, cr, cb, door_x, door_y = _get_cage_bounds(width, height)
    protected = {(door_x, door_y)}
    if ct - 1 > 0:
        protected.add((door_x, ct - 1))
    return protected


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

def _build_maze(width, height, algorithm, rng=None):
    if rng is None:
        rng = random

    maze = [[1] * width for _ in range(height)]

    # 1. Carve initial paths on the left side
    _carve_left_half(maze, width, height, algorithm, rng)

    cage_zone = _get_cage_zone(width, height)

    # 2. Execute ALL random refinements on the left half BEFORE mirroring
    eliminate_dead_ends(maze, width, height, cage_zone=cage_zone)
    enforce_no_2x2_open_areas(
        maze, width, height,
        cage_zone=cage_zone,
        protected=_protected_corridor_tiles(width, height)
    )

    if algorithm == "recursive_backtracking":
        refine_recursive_backtracking_layout(
            maze,
            width,
            height,
            cage_zone=cage_zone,
            passes=12,
            min_loop_distance=2,
            max_probe=10,
            min_dead_end_corridor_len=3
        )

    # Slice oversized wall masses into additional corridors before symmetry copy.
    slice_large_wall_blocks(
        maze,
        width,
        height,
        cage_zone=cage_zone,
        protected=_protected_corridor_tiles(width, height),
        min_span=5,
        min_area=24,
        min_cut_len=4,
    )

    # Final shape enforcement on the left half
    enforce_no_2x2_open_areas(
        maze, width, height,
        cage_zone=cage_zone,
        protected=_protected_corridor_tiles(width, height),
        max_passes=20
    )
    eliminate_dead_ends(maze, width, height, max_iterations=20, cage_zone=cage_zone)
    enforce_no_2x2_open_areas(
        maze, width, height,
        cage_zone=cage_zone,
        protected=_protected_corridor_tiles(width, height),
        max_passes=20
    )

    # 3. MIRROR THE MAZE
    # The left half is now perfectly refined. We physically copy it to the right.
    mirror_maze(maze, width, height)

    # 4. Stamp the Cage and Add Center Bridges
    # The cage and bridges are mathematically centered, so they preserve symmetry.
    cage_bounds = create_ghost_cage(maze, width, height)
    bridge_count = rng.choice([2, 3])
    _add_center_bridges(maze, width, height, cage_bounds=cage_bounds, bridge_count=bridge_count, rng=rng)

    # 5. Final global cleanup after bridge carving.
    # Bridges can reintroduce 2x2 open pockets near centre; clean them on full maze.
    protected = _protected_corridor_tiles(width, height)
    enforce_no_2x2_open_areas(
        maze, width, height,
        cage_zone=cage_zone,
        protected=protected,
        max_passes=24
    )
    eliminate_dead_ends(maze, width, height, max_iterations=8, cage_zone=cage_zone)
    enforce_no_2x2_open_areas(
        maze, width, height,
        cage_zone=cage_zone,
        protected=protected,
        max_passes=24
    )

    # 6. Re-enforce symmetry after full-maze cleanup.
    # Cleanup tie-breakers can introduce tiny left/right drift; mirror fixes that.
    mirror_maze(maze, width, height)

    # 7. Re-stamp cage last so cleanup/mirroring cannot damage it, and clear door path.
    _, ct, _, _, door_x, _ = create_ghost_cage(maze, width, height)
    if ct - 1 > 0:
        maze[ct - 1][door_x] = 0

    return maze


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def generate_recursive_backtracking(width, height, rng=None):
    return _build_maze(width, height, "recursive_backtracking", rng=rng)


def generate_prims(width, height, rng=None):
    return _build_maze(width, height, "prims", rng=rng)


def generate_random_walk(width, height, rng=None):
    return _build_maze(width, height, "random_walk", rng=rng)
