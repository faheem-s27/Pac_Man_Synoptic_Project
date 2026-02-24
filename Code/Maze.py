import pygame
import random

class Maze:
    """Represents the Pac-Man maze layout and collision detection"""

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

    def __init__(self, tile_size=40):
        """
        Initialize the maze.

        Args:
            tile_size: Size of each tile in pixels (default 40)
            width: Width of maze in tiles (default None, uses classic maze width)
            height: Height of maze in tiles (default None, uses classic maze height)
            use_classic: If True and width/height are None, uses the classic maze
        """
        self.tile_size = tile_size
        self.wall_color = (33, 33, 222)  # Arcade blue
        self.path_color = (0, 0, 0)  # Black
        self.maze = self.CLASSIC_MAZE
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
        """Draw the maze on the surface, combining adjacent walls into longer rectangles"""
        drawn = set()

        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y][x] == 1 and (x, y) not in drawn:
                    # Find the length of connected walls horizontally
                    width = 1
                    while x + width < self.width and self.maze[y][x + width] == 1 and (x + width, y) not in drawn:
                        width += 1

                    # Find the height of connected walls vertically (checking if all cells in rectangle are walls)
                    height = 1
                    can_extend = True
                    while y + height < self.height and can_extend:
                        for check_x in range(x, x + width):
                            if self.maze[y + height][check_x] != 1:
                                can_extend = False
                                break
                        if can_extend:
                            height += 1

                    # Draw the combined rectangle
                    rect = pygame.Rect(x * self.tile_size, y * self.tile_size,
                                      width * self.tile_size, height * self.tile_size)
                    pygame.draw.rect(surface, self.wall_color, rect, 3)

                    # Mark all cells in this rectangle as drawn
                    for dy in range(height):
                        for dx in range(width):
                            drawn.add((x + dx, y + dy))

