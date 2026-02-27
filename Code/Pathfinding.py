import heapq
import itertools
import pygame

def _heuristic(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

class Pathfinding:
    def __init__(self, maze):
        self.maze = maze
        self.path = []

    def find_shortest_path(self, start_gx, start_gy, target_gx, target_gy, current_dir=(0, 0)):
        """Finds path while penalizing 180-degree reversals to maintain forward flow."""
        start = (start_gx, start_gy)
        target = (target_gx, target_gy)

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
        f_score = {start: _heuristic(start[0], start[1], target[0], target[1])}
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
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                neighbor = (cx + dx, cy + dy)

                if 0 <= neighbor[0] < self.maze.width and 0 <= neighbor[1] < self.maze.height:
                    if self.maze.maze[neighbor[1]][neighbor[0]] == 0:

                        # Apply massive penalty to the 'behind' tile at the start node
                        move_cost = 1
                        if current == start and neighbor == behind_tile:
                            move_cost = 1000

                        tent_g = g_score[current] + move_cost
                        if tent_g < g_score.get(neighbor, float('inf')):
                            came_from[neighbor] = current
                            g_score[neighbor] = tent_g
                            f_score[neighbor] = tent_g + _heuristic(neighbor[0], neighbor[1], target[0], target[1])

                            if neighbor not in open_set_hash:
                                open_set_hash.add(neighbor)
                                heapq.heappush(open_set, (f_score[neighbor], next(counter), neighbor))
        return []
    def draw_path(self, surface, path, color=(255, 100, 100), line_width=2):
        if not path:
            return

        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]

            px1 = x1 * self.maze.tile_size + self.maze.tile_size // 2
            py1 = y1 * self.maze.tile_size + self.maze.tile_size // 2
            px2 = x2 * self.maze.tile_size + self.maze.tile_size // 2
            py2 = y2 * self.maze.tile_size + self.maze.tile_size // 2

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