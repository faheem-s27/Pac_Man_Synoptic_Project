from collections import deque
import pygame

class Pathfinding:
    """Pathfinding system using BFS to find shortest path in the maze"""

    def __init__(self, maze):
        """
        Initialize pathfinding with maze reference.

        Args:
            maze: Maze object containing the maze layout
        """
        self.maze = maze
        self.path = []
        self.start_x = 1
        self.start_y = 1

    def find_shortest_path(self, target_x, target_y):
        """
        Find shortest path from top-left corner to target using BFS.

        Args:
            target_x: Target X grid coordinate
            target_y: Target Y grid coordinate

        Returns:
            List of (x, y) coordinates representing the shortest path
        """
        # Convert pixel coordinates to grid coordinates
        target_grid_x = target_x // self.maze.tile_size
        target_grid_y = target_y // self.maze.tile_size

        # BFS to find shortest path
        queue = deque([(self.start_x, self.start_y, [(self.start_x, self.start_y)])])
        visited = set([(self.start_x, self.start_y)])

        while queue:
            x, y, path = queue.popleft()

            # Check if we reached the target
            if x == target_grid_x and y == target_grid_y:
                self.path = path
                return path

            # Explore neighbors (up, down, left, right)
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy

                # Check if neighbor is valid and not visited
                if (0 <= nx < self.maze.width and
                    0 <= ny < self.maze.height and
                    (nx, ny) not in visited and
                    self.maze.maze[ny][nx] == 0):  # 0 = path, 1 = wall

                    visited.add((nx, ny))
                    new_path = path + [(nx, ny)]
                    queue.append((nx, ny, new_path))

        # No path found
        self.path = []
        return []

    def draw_path(self, surface, target_x, target_y, color=(255, 100, 100), line_width=2):
        """
        Draw the shortest path on the screen.

        Args:
            surface: Pygame surface to draw on
            target_x: Target X pixel coordinate (Pac-Man X)
            target_y: Target Y pixel coordinate (Pac-Man Y)
            color: Color of the path line (default red)
            line_width: Width of the path line
        """
        # Find the shortest path
        path = self.find_shortest_path(target_x, target_y)

        if not path or len(path) < 2:
            return

        # Draw line segments connecting path nodes
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]

            # Convert grid coordinates to pixel coordinates (center of tile)
            px1 = x1 * self.maze.tile_size + self.maze.tile_size // 2
            py1 = y1 * self.maze.tile_size + self.maze.tile_size // 2
            px2 = x2 * self.maze.tile_size + self.maze.tile_size // 2
            py2 = y2 * self.maze.tile_size + self.maze.tile_size // 2

            pygame.draw.line(surface, color, (px1, py1), (px2, py2), line_width)

        # Draw circles at path nodes
        node_radius = 3
        for x, y in path:
            px = x * self.maze.tile_size + self.maze.tile_size // 2
            py = y * self.maze.tile_size + self.maze.tile_size // 2
            pygame.draw.circle(surface, color, (px, py), node_radius)

    def get_path_length(self):
        """Get the length of the current path"""
        return len(self.path)

    def get_path(self):
        """Get the current cached path"""
        return self.path

