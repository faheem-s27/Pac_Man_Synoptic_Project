import pygame
import math

class PacMan:
    """Represents the Pac-Man character"""

    def __init__(self, x, y, tile_size=40, speed=50):
        self.x = x
        self.y = y
        self.tile_size = tile_size
        self.size = tile_size  # Slightly smaller than tile for movement
        self.speed = speed
        self.direction = (0, 0)  # Current direction
        self.next_direction = (0, 0)  # Next direction to move
        self.color = (255, 255, 0)  # Bright yellow
        self.score = 0
        self.pellets_eaten = 0

    def is_aligned_to_tile(self):
        """Check if Pac-Man is aligned to the center of a tile"""
        # Check if position is at the center of a tile (within tolerance)
        center_x = self.x + self.size // 2
        center_y = self.y + self.size // 2

        # Calculate position relative to tile grid
        tile_offset_x = center_x % self.tile_size
        tile_offset_y = center_y % self.tile_size

        # Allow small tolerance for floating point precision
        tolerance = 2
        return (abs(tile_offset_x - self.tile_size // 2) < tolerance and
                abs(tile_offset_y - self.tile_size // 2) < tolerance)

    def set_direction(self, direction):
        """Set the next direction for Pac-Man"""
        # Only allow direction change if aligned to tile center
        if self.is_aligned_to_tile():
            self.next_direction = direction
        else:
            # Store the buffered direction but don't apply it yet
            self.next_direction = direction

    def update(self, maze):
        """Update Pac-Man's position"""
        # Only try to switch to next direction if aligned to tile
        if self.is_aligned_to_tile() and self.next_direction != (0, 0):
            next_x = self.x + self.next_direction[0] * self.speed
            next_y = self.y + self.next_direction[1] * self.speed

            if maze.can_move(next_x, next_y, self.size):
                self.direction = self.next_direction
                self.next_direction = (0, 0)

        # Move in current direction
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
        """Get Pac-Man's grid position"""
        return self.x // self.tile_size, self.y // self.tile_size

    def eat_pellet(self, points=10):
        """Add points and increment pellet counter"""
        self.score += points
        self.pellets_eaten += 1



