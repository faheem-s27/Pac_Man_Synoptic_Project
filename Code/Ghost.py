import pygame
from enum import Enum
from Code.Pathfinding import Pathfinding


class GhostState(Enum):
    CHASE = 1
    SCATTER = 2
    EATEN = 3
    FRIGHTENED = 4


def _get_opposite_dir(direction):
    """Returns the inverse vector to identify 180-degree turns."""
    return -direction[0], -direction[1]  #


class Ghost:
    PATH_RECALC_FRAMES = 10
    SCATTER_DURATION = 10 * 60
    CHASE_DURATION = 20 * 60

    def __init__(self, x, y, tile_size=40, speed=2, maze=None, name="Ghost"):
        self.tile_size = tile_size
        self.size = 30
        self.speed = speed
        self.maze = maze
        self.name = name
        self.offset = (self.tile_size - self.size) / 2

        self.x = float(x + self.offset)
        self.y = float(y + self.offset)

        self.current_dir = (0, 0)
        self.state = GhostState.SCATTER
        self.pathfinding = Pathfinding(maze)
        self.path = []
        self.path_index = 0
        self.path_update_counter = 0
        self.color = None

        self.mode_timer = 0
        self.is_scatter = True

    @property
    def grid_pos(self):
        center_x = self.x + self.size / 2
        center_y = self.y + self.size / 2
        return int(center_x // self.tile_size), int(center_y // self.tile_size)

    def is_at_center(self):
        """Strict check for tile center alignment."""
        tolerance = self.speed
        return (abs((self.x - self.offset) % self.tile_size) <= tolerance and
                abs((self.y - self.offset) % self.tile_size) <= tolerance)

    def update(self, pacman):
        self.mode_timer += 1

        # Only switch modes at intersections to prevent pathing glitches
        if self.is_at_center():
            if self.is_scatter and self.mode_timer >= self.SCATTER_DURATION:
                self.is_scatter = False
                self.state = GhostState.CHASE
                self.mode_timer = 0
                self.path = []
            elif not self.is_scatter and self.mode_timer >= self.CHASE_DURATION:
                self.is_scatter = True
                self.state = GhostState.SCATTER
                self.mode_timer = 0
                self.path = []

        self._execute_state_logic(pacman)

    def _execute_state_logic(self, pacman):
        """Modified to handle path exhaustion by forcing momentum."""
        self.path_update_counter += 1

        target_gx, target_gy = (self.get_scatter_target_tile() if self.is_scatter
                                else self.get_target_tile(pacman))

        # Check if we've arrived at the target tile
        current_gx, current_gy = self.grid_pos
        arrived_at_target = (current_gx == target_gx and current_gy == target_gy)

        # Pathfinding update: only if we aren't already at the target
        if self.path_update_counter >= self.PATH_RECALC_FRAMES and self.is_at_center():
            if not arrived_at_target:
                self.path = self.pathfinding.find_shortest_path(
                    current_gx, current_gy, target_gx, target_gy, self.current_dir
                )
                self.path_index = 0
            else:
                # If at target, clear path to let momentum/intersection logic take over
                self.path = []
            self.path_update_counter = 0

        # Movement implementation
        if self.path and self.path_index < len(self.path):
            next_tile = self.path[self.path_index]
            self._move_towards(next_tile[0] * self.tile_size + self.offset,
                               next_tile[1] * self.tile_size + self.offset)
        else:
            # Result: Force movement even when the A* path is empty
            self._apply_intersection_logic()

    def _apply_intersection_logic(self):
        """Forced turn logic when at an intersection with no path."""
        if self.is_at_center():
            # Get available directions excluding the 180-degree reversal
            opposite = _get_opposite_dir(self.current_dir)
            valid_dirs = []

            for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                if d == opposite and self.current_dir != (0, 0):
                    continue

                # Check if the next tile in this direction is a wall
                next_x = self.x + d[0] * self.speed
                next_y = self.y + d[1] * self.speed
                if self.maze.can_move(next_x, next_y, self.size):
                    valid_dirs.append(d)

            if valid_dirs:
                # If we can keep going straight, do so; otherwise pick a turn
                if self.current_dir in valid_dirs:
                    pass
                else:
                    import random
                    self.current_dir = random.choice(valid_dirs)
            else:
                # Absolute dead end: 180-degree turn as last resort
                self.current_dir = opposite

        # Apply the movement
        self.x += self.current_dir[0] * self.speed
        self.y += self.current_dir[1] * self.speed

    def _move_towards(self, target_px_x, target_px_y):
        dx = target_px_x - self.x
        dy = target_px_y - self.y

        if abs(dx) > 0.01:
            step = min(self.speed, abs(dx))
            self.x += step if dx > 0 else -step
            self.current_dir = (1 if dx > 0 else -1, 0)
        elif abs(dy) > 0.01:
            step = min(self.speed, abs(dy))
            self.y += step if dy > 0 else -step
            self.current_dir = (0, 1 if dy > 0 else -1)

        if abs(target_px_x - self.x) <= 0.01 and abs(target_px_y - self.y) <= 0.01:
            self.x, self.y = float(target_px_x), float(target_px_y)
            self.path_index += 1

    def _apply_momentum(self):
        """Unified forward momentum logic."""
        if self.current_dir == (0, 0):
            for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                if self.maze.can_move(self.x + d[0], self.y + d[1], self.size):
                    self.current_dir = d
                    break
            return

        next_x = self.x + self.current_dir[0] * self.speed
        next_y = self.y + self.current_dir[1] * self.speed

        if self.maze.can_move(next_x, next_y, self.size):
            self.x, self.y = next_x, next_y
        else:
            # Hit a wall? Force turn check by resetting counter
            self.path_update_counter = self.PATH_RECALC_FRAMES

    def get_target_tile(self, pacman):
        p_center_x = pacman.x + pacman.size / 2
        p_center_y = pacman.y + pacman.size / 2
        return int(p_center_x // self.tile_size), int(p_center_y // self.tile_size)

    def get_scatter_target_tile(self):
        return self.maze.width - 2, 1  # Blinky top-right corner inside walls

    def draw(self, surface):
        if self.color:
            pygame.draw.rect(surface, self.color, (int(self.x), int(self.y), self.size, self.size))


class Pinky(Ghost):
    """Pink ghost - targets 4 tiles ahead of Pac-Man's current direction"""

    def get_target_tile(self, pacman):
        p_gx, p_gy = pacman.get_grid_position()

        # Calculate 4 tiles ahead based on current heading
        target_gx = p_gx + (pacman.direction[0] // pacman.speed) * 4
        target_gy = p_gy + (pacman.direction[1] // pacman.speed) * 4

        # Standardize: Clamp to maze interior to avoid targeting border walls
        target_gx = max(1, min(target_gx, self.maze.width - 2))
        target_gy = max(1, min(target_gy, self.maze.height - 2))

        return int(target_gx), int(target_gy)

    def get_scatter_target_tile(self):
        # Targets the top-left corner area
        return 1, 1


class Clyde(Ghost):
    """Orange ghost - chases Pac-Man, but retreats to bottom-left when within 8 tiles"""

    def get_target_tile(self, pacman):
        p_gx, p_gy = pacman.get_grid_position()
        c_gx, c_gy = self.grid_pos

        # Manhattan Distance: |x1 - x2| + |y1 - y2|
        distance = abs(p_gx - c_gx) + abs(p_gy - c_gy)

        if distance > 8:
            return p_gx, p_gy
        else:
            return self.get_scatter_target_tile()

    def get_scatter_target_tile(self):
        # Targets the bottom-left corner area
        return 1, self.maze.height - 2


class Inky(Ghost):
    """Cyan ghost - targets based on Blinky's position and Pac-Man's position"""

    def __init__(self, x, y, tile_size=40, speed=2, maze=None, name="Inky", blinky=None):
        super().__init__(x, y, tile_size, speed, maze, name)
        self.blinky = blinky

    def get_target_tile(self, pacman):
        if not self.blinky:
            return super().get_target_tile(pacman)

        # 1. Get 2 tiles ahead of Pac-Man
        p_gx, p_gy = pacman.get_grid_position()
        p_ahead_x = p_gx + (pacman.direction[0] // pacman.speed) * 2
        p_ahead_y = p_gy + (pacman.direction[1] // pacman.speed) * 2

        # 2. Get vector from Blinky to that point
        b_gx, b_gy = self.blinky.grid_pos
        vec_x, vec_y = p_ahead_x - b_gx, p_ahead_y - b_gy

        # 3. Double the vector
        target_gx = b_gx + (vec_x * 2)
        target_gy = b_gy + (vec_y * 2)

        return (max(1, min(int(target_gx), self.maze.width - 2)),
                max(1, min(int(target_gy), self.maze.height - 2)))

    def get_scatter_target_tile(self):
        # Targets the bottom-right corner area
        return self.maze.width - 2, self.maze.height - 2
