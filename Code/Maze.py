import pygame
import math
import MazeGenerator

class Maze:
    """Represents the Pac-Man maze layout and collision detection"""

    def __init__(self, tile_size=40, width=20, height=21, use_classic=True, algorithm="recursive_backtracking"):
        """
        Initialize the maze.

        Args:
            tile_size: Size of each tile in pixels (default 40)
            width: Width of maze in tiles (default 20)
            height: Height of maze in tiles (default 21)
            use_classic: If True, uses the classic maze (default True)
            algorithm: "recursive_backtracking", "prims", or "random_walk"
        """
        self.tile_size = tile_size
        self.wall_color = (33, 33, 222)  # Arcade blue
        self.path_color = (0, 0, 0)  # Black
        self.maze = MazeGenerator.generate_maze(use_classic, width, height, algorithm)
        self.width = len(self.maze[0])
        self.height = len(self.maze)

    def is_wall(self, x, y):
        """Check if a grid position is a wall"""
        x = int(x)
        y = int(y)
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return self.maze[y][x] == 1

    def can_move(self, x, y, size):
        """Check if a character can move to this position (collision detection)"""
        # Check all corners of the character's bounding box
        left = x // self.tile_size
        right = (x + size - 1) // self.tile_size
        top = y // self.tile_size
        bottom = (y + size - 1) // self.tile_size

        # Check all four corners
        return not (self.is_wall(left, top) or self.is_wall(right, top) or
                    self.is_wall(left, bottom) or self.is_wall(right, bottom))

    def draw(self, surface):
        """Draws hollow Pac-Man borders with calculated border radii."""
        pad = self.tile_size // 20
        ts = self.tile_size
        arc_rect_size = pad * 2  # The bounding box for the corner arc

        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y][x] == 1:
                    px = x * ts
                    py = y * ts

                    # 1. Adjacency Checks
                    top_is_path = y > 0 and self.maze[y - 1][x] == 0
                    bottom_is_path = y < self.height - 1 and self.maze[y + 1][x] == 0
                    left_is_path = x > 0 and self.maze[y][x - 1] == 0
                    right_is_path = x < self.width - 1 and self.maze[y][x + 1] == 0

                    # 2. Corner Radius Logic (Arcs)
                    # Top-Left Corner
                    if top_is_path and left_is_path:
                        rect = pygame.Rect(px + pad, py + pad, arc_rect_size, arc_rect_size)
                        pygame.draw.arc(surface, self.wall_color, rect, math.pi / 2, math.pi, 2)

                    # Top-Right Corner
                    if top_is_path and right_is_path:
                        rect = pygame.Rect(px + ts - pad - arc_rect_size, py + pad, arc_rect_size, arc_rect_size)
                        pygame.draw.arc(surface, self.wall_color, rect, 0, math.pi / 2, 2)

                    # Bottom-Left Corner
                    if bottom_is_path and left_is_path:
                        rect = pygame.Rect(px + pad, py + ts - pad - arc_rect_size, arc_rect_size, arc_rect_size)
                        pygame.draw.arc(surface, self.wall_color, rect, math.pi, 3 * math.pi / 2, 2)

                    # Bottom-Right Corner
                    if bottom_is_path and right_is_path:
                        rect = pygame.Rect(px + ts - pad - arc_rect_size, py + ts - pad - arc_rect_size, arc_rect_size,
                                           arc_rect_size)
                        pygame.draw.arc(surface, self.wall_color, rect, 3 * math.pi / 2, 2 * math.pi, 2)

                    # 3. Shortened Straight Lines
                    # Furthermore, we only draw straight lines where corners DO NOT exist to prevent overlap.
                    if top_is_path:
                        start_x = px + pad + (arc_rect_size / 2 if left_is_path else 0)
                        end_x = px + ts - pad - (arc_rect_size / 2 if right_is_path else 0)
                        if start_x < end_x:
                            pygame.draw.line(surface, self.wall_color, (start_x, py + pad), (end_x, py + pad), 2)

                    if bottom_is_path:
                        start_x = px + pad + (arc_rect_size / 2 if left_is_path else 0)
                        end_x = px + ts - pad - (arc_rect_size / 2 if right_is_path else 0)
                        if start_x < end_x:
                            pygame.draw.line(surface, self.wall_color, (start_x, py + ts - pad), (end_x, py + ts - pad),
                                             2)

                    if left_is_path:
                        start_y = py + pad + (arc_rect_size / 2 if top_is_path else 0)
                        end_y = py + ts - pad - (arc_rect_size / 2 if bottom_is_path else 0)
                        if start_y < end_y:
                            pygame.draw.line(surface, self.wall_color, (px + pad, start_y), (px + pad, end_y), 2)

                    if right_is_path:
                        start_y = py + pad + (arc_rect_size / 2 if top_is_path else 0)
                        end_y = py + ts - pad - (arc_rect_size / 2 if bottom_is_path else 0)
                        if start_y < end_y:
                            pygame.draw.line(surface, self.wall_color, (px + ts - pad, start_y), (px + ts - pad, end_y),
                                             2)
