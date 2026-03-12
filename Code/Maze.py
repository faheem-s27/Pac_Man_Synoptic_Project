import pygame
import math
import Code.MazeGenerator as MazeGenerator

class Maze:
    DOOR_TILE = 2  # Ghost door — passable by ghosts, wall for Pac-Man

    def __init__(self, tile_size=40, width=20, height=21, algorithm="recursive_backtracking", seed=None):
        self.tile_size = tile_size
        self.wall_color = (33, 33, 222)
        self.door_color = (255, 182, 255)
        self.path_color = (0, 0, 0)
        self.maze = MazeGenerator.generate_maze(width, height, algorithm, seed=seed)
        self.width = len(self.maze[0])
        self.height = len(self.maze)

        # Read cage bounds written by create_ghost_cage
        bounds = MazeGenerator.create_ghost_cage.last_bounds
        if bounds:
            cl, ct, cr, cb, dx, dy = bounds
            self.cage_left   = cl
            self.cage_top    = ct
            self.cage_right  = cr
            self.cage_bottom = cb
            self.door_x      = dx
            self.door_y      = dy
        else:
            # Classic maze fallback
            self.cage_left   = self.width  // 2 - 3
            self.cage_top    = self.height // 2 - 1
            self.cage_right  = self.width  // 2 + 3
            self.cage_bottom = self.height // 2 + 1
            self.door_x      = self.width  // 2
            self.door_y      = self.cage_top

        # No teleport tunnels
        self.teleport_row = None

    def handle_teleportation(self, x, y):
        """Teleportation disabled — returns position unchanged."""
        return x, y

    def is_wall(self, x, y):
        """Returns True for solid walls AND door tile (Pac-Man cannot pass door)."""
        x, y = int(x), int(y)
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        tile = self.maze[y][x]
        return tile == 1 or tile == self.DOOR_TILE

    def is_ghost_wall(self, x, y):
        """Ghosts can pass through the door tile but not solid walls."""
        x, y = int(x), int(y)
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return self.maze[y][x] == 1

    def can_move(self, x, y, size):
        """Pac-Man collision — door tile counts as a wall."""
        left   = int(x) // self.tile_size
        right  = int(x + size - 1) // self.tile_size
        top    = int(y) // self.tile_size
        bottom = int(y + size - 1) // self.tile_size
        return not (self.is_wall(left, top)   or self.is_wall(right, top) or
                    self.is_wall(left, bottom) or self.is_wall(right, bottom))

    def can_ghost_move(self, x, y, size):
        """Ghost collision — door tile is passable."""
        left   = int(x) // self.tile_size
        right  = int(x + size - 1) // self.tile_size
        top    = int(y) // self.tile_size
        bottom = int(y + size - 1) // self.tile_size
        return not (self.is_ghost_wall(left, top)   or self.is_ghost_wall(right, top) or
                    self.is_ghost_wall(left, bottom) or self.is_ghost_wall(right, bottom))

    def draw(self, surface):
        pad = max(1, self.tile_size // 20)
        ts  = self.tile_size
        arc_sz = pad * 2

        for y in range(self.height):
            for x in range(self.width):
                tile = self.maze[y][x]

                if tile == self.DOOR_TILE:
                    px, py = x * ts, y * ts
                    bar_h  = max(3, ts // 8)
                    bar_y  = py + ts // 2 - bar_h // 2
                    pygame.draw.rect(surface, self.door_color,
                                     pygame.Rect(px, bar_y, ts, bar_h))
                    continue

                if tile != 1:
                    continue

                px, py = x * ts, y * ts

                def open_nb(cy, cx):
                    if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                        return False
                    return self.maze[cy][cx] != 1   # 0 or 2 both count as open

                top_o    = open_nb(y - 1, x)
                bottom_o = open_nb(y + 1, x)
                left_o   = open_nb(y, x - 1)
                right_o  = open_nb(y, x + 1)

                if top_o and left_o:
                    pygame.draw.arc(surface, self.wall_color,
                        pygame.Rect(px + pad, py + pad, arc_sz, arc_sz),
                        math.pi / 2, math.pi, 2)
                if top_o and right_o:
                    pygame.draw.arc(surface, self.wall_color,
                        pygame.Rect(px + ts - pad - arc_sz, py + pad, arc_sz, arc_sz),
                        0, math.pi / 2, 2)
                if bottom_o and left_o:
                    pygame.draw.arc(surface, self.wall_color,
                        pygame.Rect(px + pad, py + ts - pad - arc_sz, arc_sz, arc_sz),
                        math.pi, 3 * math.pi / 2, 2)
                if bottom_o and right_o:
                    pygame.draw.arc(surface, self.wall_color,
                        pygame.Rect(px + ts - pad - arc_sz, py + ts - pad - arc_sz, arc_sz, arc_sz),
                        3 * math.pi / 2, 2 * math.pi, 2)

                if top_o:
                    sx = px + pad + (arc_sz / 2 if left_o  else 0)
                    ex = px + ts - pad - (arc_sz / 2 if right_o else 0)
                    if sx < ex:
                        pygame.draw.line(surface, self.wall_color, (sx, py + pad), (ex, py + pad), 2)
                if bottom_o:
                    sx = px + pad + (arc_sz / 2 if left_o  else 0)
                    ex = px + ts - pad - (arc_sz / 2 if right_o else 0)
                    if sx < ex:
                        pygame.draw.line(surface, self.wall_color, (sx, py + ts - pad), (ex, py + ts - pad), 2)
                if left_o:
                    sy = py + pad + (arc_sz / 2 if top_o    else 0)
                    ey = py + ts - pad - (arc_sz / 2 if bottom_o else 0)
                    if sy < ey:
                        pygame.draw.line(surface, self.wall_color, (px + pad, sy), (px + pad, ey), 2)
                if right_o:
                    sy = py + pad + (arc_sz / 2 if top_o    else 0)
                    ey = py + ts - pad - (arc_sz / 2 if bottom_o else 0)
                    if sy < ey:
                        pygame.draw.line(surface, self.wall_color, (px + ts - pad, sy), (px + ts - pad, ey), 2)
