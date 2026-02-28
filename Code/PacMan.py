import pygame

class PacMan:
    def __init__(self, x, y, tile_size=40, speed=2):
        self.x = x
        self.y = y
        self.tile_size = tile_size
        self.size = tile_size
        self.speed = speed
        self.direction = (0, 0)
        self.next_direction = (0, 0)
        self.color = (255, 255, 0)
        self.score = 0
        self.pellets_eaten = 0

    def is_aligned_to_tile(self):
        center_x = self.x + self.size // 2
        center_y = self.y + self.size // 2
        tile_offset_x = center_x % self.tile_size
        tile_offset_y = center_y % self.tile_size
        # Dynamic tolerance based on speed to handle speeds 2, 3, 4, etc.
        tolerance = self.speed + 1
        return (abs(tile_offset_x - self.tile_size // 2) < tolerance and
                abs(tile_offset_y - self.tile_size // 2) < tolerance)

    def set_direction(self, direction):
        if self.is_aligned_to_tile():
            self.next_direction = direction
        else:
            self.next_direction = direction

    def update(self, maze):
        if self.is_aligned_to_tile() and self.next_direction != (0, 0):
            next_x = self.x + self.next_direction[0] * self.speed
            next_y = self.y + self.next_direction[1] * self.speed

            if maze.can_move(next_x, next_y, self.size):
                self.direction = self.next_direction
                self.next_direction = (0, 0)

        next_x = self.x + self.direction[0] * self.speed
        next_y = self.y + self.direction[1] * self.speed

        if maze.can_move(next_x, next_y, self.size):
            self.x = next_x
            self.y = next_y

    def draw(self, surface):
        center_x = self.x + self.size // 2
        center_y = self.y + self.size // 2
        pygame.draw.circle(surface, self.color, (int(center_x), int(center_y)), self.size // 3)

    def get_grid_position(self):
        return self.x // self.tile_size, self.y // self.tile_size

    def eat_pellet(self, points=10):
        self.score += points
        self.pellets_eaten += 1