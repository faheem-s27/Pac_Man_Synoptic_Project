import heapq
import itertools
import pygame

def _heuristic(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def _heuristic_with_wraparound(x1, y1, x2, y2, maze_width, teleport_row):
    """Heuristic that considers wraparound on the teleport row."""
    # Normal Manhattan distance
    normal_dist = abs(x1 - x2) + abs(y1 - y2)

    # If both positions are on the teleport row, consider wraparound distance
    if y1 == teleport_row and y2 == teleport_row:
        # Calculate wraparound distance (going through the edge)
        wrap_dist = min(
            x1 + (maze_width - x2),  # Go left through edge to reach right
            x2 + (maze_width - x1)   # Go right through edge to reach left
        )
        return min(normal_dist, wrap_dist)

    return normal_dist

class Pathfinding:
    def __init__(self, maze):
        self.maze = maze
        self.path = []

    def find_shortest_path(self, start_gx, start_gy, target_gx, target_gy, current_dir=(0, 0), allow_wraparound=True):
        """Finds path while penalizing 180-degree reversals and optionally handling wraparound teleportation."""
        start = (start_gx, start_gy)
        target = (target_gx, target_gy)

        # Get teleport row from maze
        teleport_row = self.maze.teleport_row if hasattr(self.maze, 'teleport_row') else None

        # Disable wraparound if not allowed
        if not allow_wraparound:
            teleport_row = None

        # Identify the tile directly behind the ghost's heading
        behind_tile = (start_gx - current_dir[0], start_gy - current_dir[1])

        # If already at the target, return empty to trigger momentum logic
        if start == target:
            return []

        counter = itertools.count()
        open_set = []
        heapq.heappush(open_set, (0, next(counter), start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: _heuristic_with_wraparound(start[0], start[1], target[0], target[1],
                                                      self.maze.width, teleport_row if teleport_row else -1)}
        open_set_hash = {start}

        while open_set:
            current = heapq.heappop(open_set)[2]
            open_set_hash.remove(current)

            if current == target:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            cx, cy = current

            # Regular neighbors (up, down, left, right)
            neighbors = []
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                neighbor = (cx + dx, cy + dy)

                if 0 <= neighbor[0] < self.maze.width and 0 <= neighbor[1] < self.maze.height:
                    if hasattr(self.maze, 'is_ghost_wall'):
                        passable = not self.maze.is_ghost_wall(neighbor[0], neighbor[1])
                    else:
                        passable = self.maze.maze[neighbor[1]][neighbor[0]] == 0
                    if passable:
                        neighbors.append(neighbor)

            # No wraparound — teleport tunnels are disabled

            # Process all neighbors
            for neighbor in neighbors:
                # Apply massive penalty to the 'behind' tile at the start node
                move_cost = 1
                if current == start and neighbor == behind_tile:
                    move_cost = 1000

                tent_g = g_score[current] + move_cost
                if tent_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tent_g
                    f_score[neighbor] = tent_g + _heuristic_with_wraparound(
                        neighbor[0], neighbor[1], target[0], target[1],
                        self.maze.width, teleport_row if teleport_row else -1
                    )

                    if neighbor not in open_set_hash:
                        open_set_hash.add(neighbor)
                        heapq.heappush(open_set, (f_score[neighbor], next(counter), neighbor))

        return []
    def draw_path(self, surface, path, color=(255, 100, 100), line_width=2):
        if not path:
            return

        teleport_row = self.maze.teleport_row if hasattr(self.maze, 'teleport_row') else None

        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]

            px1 = x1 * self.maze.tile_size + self.maze.tile_size // 2
            py1 = y1 * self.maze.tile_size + self.maze.tile_size // 2
            px2 = x2 * self.maze.tile_size + self.maze.tile_size // 2
            py2 = y2 * self.maze.tile_size + self.maze.tile_size // 2

            # Don't draw line if it's a wraparound (going from edge to edge on teleport row)
            if teleport_row is not None and y1 == teleport_row and y2 == teleport_row:
                if abs(x1 - x2) > 1:  # Wraparound detected (jump from 0 to width-1 or vice versa)
                    continue  # Skip drawing this line segment

            pygame.draw.line(surface, color, (px1, py1), (px2, py2), line_width)

        node_radius = 3
        for x, y in path:
            px = x * self.maze.tile_size + self.maze.tile_size // 2
            py = y * self.maze.tile_size + self.maze.tile_size // 2
            pygame.draw.circle(surface, color, (px, py), node_radius)

    def get_path_length(self):
        return len(self.path)

    def get_path(self):
        return self.path