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

    def is_wall(self, x, y):
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