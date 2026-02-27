import pygame
from enum import Enum
from Code.Pathfinding import Pathfinding


class GhostState(Enum):
    CHASE = 1
    SCATTER = 2
    EATEN = 3
    FRIGHTENED = 4


class Ghost:
    PATH_RECALC_FRAMES = 10

    def __init__(self, x, y, tile_size=40, speed=2, maze=None, name="Ghost"):
        self.tile_size = tile_size
        self.size = 30
        self.speed = speed
        self.maze = maze
        self.name = name
        self.offset = (self.tile_size - self.size) / 2

        self.x = float(x + self.offset)
        self.y = float(y + self.offset)

        # Track current direction vector to prevent 180-degree reversals
        self.current_dir = (0, 0)
        self.state = GhostState.CHASE
        self.pathfinding = Pathfinding(maze)
        self.path = []
        self.path_index = 0
        self.path_update_counter = 0
        self.color = None

    @property
    def grid_pos(self):
        """Returns the grid indices based on the true center of the ghost."""
        center_x = self.x + self.size / 2
        center_y = self.y + self.size / 2
        return int(center_x // self.tile_size), int(center_y // self.tile_size)

    def _get_opposite_dir(self, direction):
        """Returns the inverse vector to identify 180-degree turns."""
        return -direction[0], -direction[1]

    def update(self, pacman):
        match self.state:
            case GhostState.CHASE:
                self._update_chase(pacman)

    def get_target_tile(self, pacman):
        """Default Blinky behavior: target Pac-Man's tile."""
        p_center_x = pacman.x + pacman.size / 2
        p_center_y = pacman.y + pacman.size / 2
        return int(p_center_x // self.tile_size), int(p_center_y // self.tile_size)

    def _update_chase(self, pacman):
        self.path_update_counter += 1

        # Only allow path recalculation when centered to prevent 'jitter'
        at_center = abs((self.x - self.offset) % self.tile_size) < self.speed
        at_center &= abs((self.y - self.offset) % self.tile_size) < self.speed

        if self.path_update_counter >= self.PATH_RECALC_FRAMES and at_center:
            start_gx, start_gy = self.grid_pos
            target_gx, target_gy = self.get_target_tile(pacman)

            self.path = self.pathfinding.find_shortest_path(
                start_gx, start_gy, target_gx, target_gy, self.current_dir
            )
            self.path_index = 0
            self.path_update_counter = 0

        # Execute path movement or forced momentum
        if self.path and self.path_index < len(self.path):
            next_tile = self.path[self.path_index]
            self._move_towards(next_tile[0] * self.tile_size + self.offset,
                               next_tile[1] * self.tile_size + self.offset)
        else:
            # Result: Constant motion prevents the ghost from stopping at the target
            self._continue_forward()

    def _continue_forward(self):
        """Forces the ghost to maintain its current heading until it hits a wall."""
        if self.current_dir == (0, 0):
            # Fallback: Pick first available non-wall direction
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
            # Hit a wall? Reset counter to force a legal 90-degree turn check next frame
            self.path_update_counter = self.PATH_RECALC_FRAMES

    def _move_forward_until_intersection(self):
        """Forces the ghost forward and prevents stopping/bouncing at the target."""
        if self.current_dir == (0, 0):
            # Fallback if ghost is somehow static: pick any direction that isn't a wall
            for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                if self.maze.can_move(self.x + d[0] * self.speed, self.y + d[1] * self.speed, self.size):
                    self.current_dir = d
                    break
            return

        next_x = self.x + self.current_dir[0] * self.speed
        next_y = self.y + self.current_dir[1] * self.speed

        if self.maze.can_move(next_x, next_y, self.size):
            self.x, self.y = next_x, next_y
        else:
            # If we hit a wall while moving forward, force a path recalculation
            self.path_update_counter = self.PATH_RECALC_FRAMES

    def _move_towards(self, target_px_x, target_px_y):
        """Standard clamped movement on one axis at a time."""
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

    def _move_forward_momentum(self):
        """Forces the ghost forward into the wall to trigger a turn recalculation."""
        if self.current_dir == (0, 0): return

        next_x = self.x + self.current_dir[0] * self.speed
        next_y = self.y + self.current_dir[1] * self.speed

        if self.maze.can_move(next_x, next_y, self.size):
            self.x, self.y = next_x, next_y
        else:
            # Hit a dead end? Reset counter to find an available perpendicular turn.
            self.path_update_counter = self.PATH_RECALC_FRAMES

    def draw(self, surface):
        if self.color:
            pygame.draw.rect(surface, self.color, (int(self.x), int(self.y), self.size, self.size))


class Pinky(Ghost):
    """Pink ghost - targets 4 tiles ahead of Pac-Man's current direction"""

    def get_target_tile(self, pacman):
        """Target 4 tiles ahead of Pac-Man, decreasing to 3, 2, 1 if hitting walls"""
        p_center_x = pacman.x + pacman.size / 2
        p_center_y = pacman.y + pacman.size / 2

        # Get Pac-Man's current grid position
        pacman_gx = int(p_center_x // self.tile_size)
        pacman_gy = int(p_center_y // self.tile_size)

        # Try 4 tiles ahead, then 3, then 2, then 1 if hitting walls
        for distance in range(4, 0, -1):
            # Calculate offset based on Pac-Man's direction
            # pacman.direction is a unit vector (like (0, -1) or (1, 0))
            offset_x = int(pacman.direction[0] * distance)
            offset_y = int(pacman.direction[1] * distance)

            target_gx = pacman_gx + offset_x
            target_gy = pacman_gy + offset_y

            # Check if within maze bounds
            if 0 <= target_gx < self.maze.width and 0 <= target_gy < self.maze.height:
                # If this tile is not a wall, use it as target
                if not self.maze.is_wall(target_gx, target_gy):
                    return target_gx, target_gy

        # If all ahead tiles are walls, fall back to Pac-Man's position
        return pacman_gx, pacman_gy


class Clyde(Ghost):
    """Orange ghost - chases Pac-Man, but retreats to bottom-left when within 8 tiles"""

    def get_target_tile(self, pacman):
        """Target Pac-Man unless within 8 tiles, then target bottom-left corner"""
        p_center_x = pacman.x + pacman.size / 2
        p_center_y = pacman.y + pacman.size / 2

        # Get Pac-Man's current grid position
        pacman_gx = int(p_center_x // self.tile_size)
        pacman_gy = int(p_center_y // self.tile_size)

        # Get Clyde's current grid position
        clyde_gx, clyde_gy = self.grid_pos

        # Calculate distance to Pac-Man (Manhattan distance)
        distance = abs(clyde_gx - pacman_gx) + abs(clyde_gy - pacman_gy)

        # If within 8 tiles, retreat to bottom-left corner
        if distance < 8:
            # Target bottom-left corner
            target_gx = 0
            target_gy = self.maze.height - 1

            # Find nearest valid tile in bottom-left area if exact corner is a wall
            for offset in range(5):  # Try expanding search
                for dx in range(offset + 1):
                    for dy in range(offset + 1):
                        test_x = target_gx + dx
                        test_y = target_gy - dy
                        if 0 <= test_x < self.maze.width and 0 <= test_y < self.maze.height:
                            if not self.maze.is_wall(test_x, test_y):
                                return test_x, test_y

            # Fallback to Pac-Man's position if no valid corner found
            return pacman_gx, pacman_gy
        else:
            # Chase Pac-Man directly (Blinky behavior)
            return pacman_gx, pacman_gy


