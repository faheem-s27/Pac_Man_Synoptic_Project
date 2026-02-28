import pygame
import math
import Code.MazeGenerator as MazeGenerator

class Maze:
    def __init__(self, tile_size=40, width=20, height=21, use_classic=True, algorithm="recursive_backtracking"):
        self.tile_size = tile_size
        self.wall_color = (33, 33, 222)
        self.path_color = (0, 0, 0)
        self.maze = MazeGenerator.generate_maze(use_classic, width, height, algorithm)
        self.width = len(self.maze[0])
        self.height = len(self.maze)

        # Find teleport tunnel row (row where both edges are open)
        self.teleport_row = self._find_teleport_row()

    def _find_teleport_row(self):
        """Find the row that has open edges on both left and right for teleportation."""
        for y in range(self.height):
            # Check if both left (x=0) and right (x=width-1) are paths
            if self.maze[y][0] == 0 and self.maze[y][self.width - 1] == 0:
                return y
        return None  # No teleport row found

    def handle_teleportation(self, x, y):
        """Handle wraparound teleportation at maze edges. Returns new (x, y) position."""
        if self.teleport_row is None:
            return x, y  # No teleportation if no teleport row found

        # Get the character's tile position
        tile_y = int(y) // self.tile_size

        # Only teleport on the teleport row
        if tile_y != self.teleport_row:
            return x, y

        # Check if character has gone off the edges (pixel-level)
        max_x = self.width * self.tile_size

        # If Pacman goes off the left edge (x is negative), wrap to right
        if x < 0:
            new_x = max_x + x  # Preserve the offset
            return new_x, y

        # If Pacman goes off the right edge, wrap to left
        if x >= max_x:
            new_x = x - max_x  # Preserve the offset
            return new_x, y

        return x, y

    def is_wall(self, x, y):
        x = int(x)
        y = int(y)

        # Allow wraparound on teleport row
        if self.teleport_row is not None and y == self.teleport_row:
            # If going off left edge, it's not a wall (will wraparound)
            if x < 0:
                return False
            # If going off right edge, it's not a wall (will wraparound)
            if x >= self.width:
                return False

        # Normal bounds checking for other rows
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

        # Handle wraparound: wrap tile coordinates if on teleport row
        if self.teleport_row is not None and top == self.teleport_row:
            # Wrap left coordinate if negative
            if left < 0:
                left = self.width - 1
            # Wrap right coordinate if beyond width
            if right >= self.width:
                right = 0

        return not (self.is_wall(left, top) or self.is_wall(right, top) or
                    self.is_wall(left, bottom) or self.is_wall(right, bottom))

    def draw(self, surface):
        pad = self.tile_size // 20
        ts = self.tile_size
        arc_rect_size = pad * 2

        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y][x] == 1:
                    px = x * ts
                    py = y * ts

                    top_is_path = y > 0 and self.maze[y - 1][x] == 0
                    bottom_is_path = y < self.height - 1 and self.maze[y + 1][x] == 0
                    left_is_path = x > 0 and self.maze[y][x - 1] == 0
                    right_is_path = x < self.width - 1 and self.maze[y][x + 1] == 0

                    if top_is_path and left_is_path:
                        rect = pygame.Rect(px + pad, py + pad, arc_rect_size, arc_rect_size)
                        pygame.draw.arc(surface, self.wall_color, rect, math.pi / 2, math.pi, 2)

                    if top_is_path and right_is_path:
                        rect = pygame.Rect(px + ts - pad - arc_rect_size, py + pad, arc_rect_size, arc_rect_size)
                        pygame.draw.arc(surface, self.wall_color, rect, 0, math.pi / 2, 2)

                    if bottom_is_path and left_is_path:
                        rect = pygame.Rect(px + pad, py + ts - pad - arc_rect_size, arc_rect_size, arc_rect_size)
                        pygame.draw.arc(surface, self.wall_color, rect, math.pi, 3 * math.pi / 2, 2)

                    if bottom_is_path and right_is_path:
                        rect = pygame.Rect(px + ts - pad - arc_rect_size, py + ts - pad - arc_rect_size, arc_rect_size, arc_rect_size)
                        pygame.draw.arc(surface, self.wall_color, rect, 3 * math.pi / 2, 2 * math.pi, 2)

                    if top_is_path:
                        start_x = px + pad + (arc_rect_size / 2 if left_is_path else 0)
                        end_x = px + ts - pad - (arc_rect_size / 2 if right_is_path else 0)
                        if start_x < end_x:
                            pygame.draw.line(surface, self.wall_color, (start_x, py + pad), (end_x, py + pad), 2)

                    if bottom_is_path:
                        start_x = px + pad + (arc_rect_size / 2 if left_is_path else 0)
                        end_x = px + ts - pad - (arc_rect_size / 2 if right_is_path else 0)
                        if start_x < end_x:
                            pygame.draw.line(surface, self.wall_color, (start_x, py + ts - pad), (end_x, py + ts - pad), 2)

                    if left_is_path:
                        start_y = py + pad + (arc_rect_size / 2 if top_is_path else 0)
                        end_y = py + ts - pad - (arc_rect_size / 2 if bottom_is_path else 0)
                        if start_y < end_y:
                            pygame.draw.line(surface, self.wall_color, (px + pad, start_y), (px + pad, end_y), 2)

                    if right_is_path:
                        start_y = py + pad + (arc_rect_size / 2 if top_is_path else 0)
                        end_y = py + ts - pad - (arc_rect_size / 2 if bottom_is_path else 0)
                        if start_y < end_y:
                            pygame.draw.line(surface, self.wall_color, (px + ts - pad, start_y), (px + ts - pad, end_y), 2)