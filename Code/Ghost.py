import pygame
from enum import Enum
from Code.Pathfinding import Pathfinding


class GhostState(Enum):
    CHASE = 1
    SCATTER = 2
    EATEN = 3
    FRIGHTENED = 4
    SPAWNING = 5


def _get_opposite_dir(direction):
    """Returns the inverse vector to identify 180-degree turns."""
    return -direction[0], -direction[1]  #


class Ghost:
    PATH_RECALC_FRAMES = 10
    SCATTER_DURATION = 10 * 60
    CHASE_DURATION = 20 * 60

    def __init__(self, x, y, tile_size=40, speed=2, maze=None, name="Ghost"):
        self.tile_size = tile_size
        self.size = tile_size
        self.speed = speed
        self.original_speed = speed  # Store original speed for restoration
        self.maze = maze
        self.name = name
        self.offset = (self.tile_size - self.size) / 2

        self.x = float(x + self.offset)
        self.y = float(y + self.offset)

        self.current_dir = (0, 0)
        self.state = GhostState.SPAWNING  # Start in spawning state
        self.pathfinding = Pathfinding(maze)
        self.path = []
        self.path_index = 0
        self.path_update_counter = 0
        self.color = None

        self.is_scatter = True  # Current mode (controlled by GameEngine globally)

        # Spawning system
        self.spawn_delay = 0  # Will be set by GameEngine
        self.spawn_timer = 0
        self.is_spawned = False

        # Frightened mode
        self.original_color = None
        self.previous_state = None
        self.frightened_warning = False

        # Eaten state - return to cage
        self.cage_x = None  # Will be set by GameEngine
        self.cage_y = None
        self.eaten_speed = speed * 2  # Move faster when returning to cage

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

    def reset_spawn(self):
        """Reset ghost to spawning state."""
        self.state = GhostState.SPAWNING
        self.spawn_timer = 0
        self.is_spawned = False
        self.current_dir = (0, 0)
        self.path = []
        self.is_scatter = True  # Reset to scatter mode
        # Restore original speed in case ghost was eaten while frightened
        self.speed = self.original_speed
        # Restore original color in case ghost was eaten while frightened
        if self.original_color:
            self.color = self.original_color
            self.original_color = None
        self.frightened_warning = False

    def enter_frightened_mode(self):
        """Enter frightened mode when power pellet is eaten."""
        if self.state != GhostState.SPAWNING:
            self.previous_state = self.state
            self.state = GhostState.FRIGHTENED
            self.original_color = self.color
            self.color = (0, 0, 255)  # Turn blue
            self.path = []  # Clear current path
            # Reverse direction (180-degree turn)
            self.current_dir = (-self.current_dir[0], -self.current_dir[1])
            # Halve the speed
            self.speed = self.original_speed / 2

    def exit_frightened_mode(self):
        """Exit frightened mode and return to previous behavior."""
        if self.state == GhostState.FRIGHTENED:
            self.state = self.previous_state if self.previous_state else GhostState.SCATTER
            self.color = self.original_color
            self.path = []  # Clear current path
            # Restore original speed
            self.speed = self.original_speed

    def enter_eaten_mode(self):
        """Enter eaten mode - ghost returns to cage before respawning."""
        self.state = GhostState.EATEN
        self.path = []
        self.speed = self.eaten_speed  # Move faster when returning

        # Calculate path from current position to cage
        if self.cage_x is not None and self.cage_y is not None:
            current_gx, current_gy = self.grid_pos
            cage_gx = int(self.cage_x // self.tile_size)
            cage_gy = int(self.cage_y // self.tile_size)

            # Find path to cage (no direction restriction for eaten ghosts, no wraparound)
            self.path = self.pathfinding.find_shortest_path(
                current_gx, current_gy, cage_gx, cage_gy, (0, 0), allow_wraparound=False
            )
            self.path_index = 0

    def set_mode(self, is_scatter):
        """Set scatter/chase mode globally (called by GameEngine)."""
        if self.state in (GhostState.CHASE, GhostState.SCATTER):
            old_mode = self.is_scatter
            self.is_scatter = is_scatter

            # Update state enum to match
            if is_scatter:
                self.state = GhostState.SCATTER
            else:
                self.state = GhostState.CHASE

            # Clear path when mode changes
            if old_mode != is_scatter:
                self.path = []

    def update(self, pacman):
        # Handle spawning state
        if self.state == GhostState.SPAWNING:
            self.spawn_timer += 1
            if self.spawn_timer >= self.spawn_delay:
                # Spawn complete, transition to scatter mode
                self.is_spawned = True
                self.state = GhostState.SCATTER
                self.spawn_timer = 0
            return  # Don't update while spawning

        # Handle eaten state - return to cage
        if self.state == GhostState.EATEN:
            self._return_to_cage()
            return

        # Execute movement logic (mode switching now handled by GameEngine)
        self._execute_state_logic(pacman)

    def _execute_state_logic(self, pacman):
        """Modified to handle path exhaustion by forcing momentum."""

        # Frightened mode: move randomly, avoid Pac-Man
        if self.state == GhostState.FRIGHTENED:
            self._frightened_movement()
            return

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

    def _frightened_movement(self):
        """Random movement when in frightened mode."""
        import random

        # Move in current direction until hitting a wall or intersection
        next_x = self.x + self.current_dir[0] * self.speed
        next_y = self.y + self.current_dir[1] * self.speed

        if self.maze.can_move(next_x, next_y, self.size):
            self.x, self.y = next_x, next_y
        else:
            # Hit a wall, choose a random direction
            valid_dirs = []
            for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                test_x = self.x + d[0] * self.speed
                test_y = self.y + d[1] * self.speed
                if self.maze.can_move(test_x, test_y, self.size):
                    valid_dirs.append(d)

            if valid_dirs:
                self.current_dir = random.choice(valid_dirs)
                next_x = self.x + self.current_dir[0] * self.speed
                next_y = self.y + self.current_dir[1] * self.speed
                if self.maze.can_move(next_x, next_y, self.size):
                    self.x, self.y = next_x, next_y

        # At intersections, randomly change direction (25% chance)
        if self.is_at_center() and random.random() < 0.25:
            opposite = _get_opposite_dir(self.current_dir)
            valid_dirs = []

            for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                if d == opposite:  # No 180-degree turns
                    continue
                test_x = self.x + d[0] * self.speed
                test_y = self.y + d[1] * self.speed
                if self.maze.can_move(test_x, test_y, self.size):
                    valid_dirs.append(d)

            if valid_dirs:
                self.current_dir = random.choice(valid_dirs)

    def _return_to_cage(self):
        """Move ghost back to cage after being eaten."""
        # Follow path back to cage
        if self.path and self.path_index < len(self.path):
            next_tile = self.path[self.path_index]
            self._move_towards(next_tile[0] * self.tile_size + self.offset,
                               next_tile[1] * self.tile_size + self.offset)
        else:
            # Reached cage, transition to spawning state
            current_gx, current_gy = self.grid_pos
            cage_gx = int(self.cage_x // self.tile_size)
            cage_gy = int(self.cage_y // self.tile_size)

            # Check if we're at the cage
            if current_gx == cage_gx and current_gy == cage_gy:
                # Reset to spawning state
                self.reset_spawn()
            else:
                # Recalculate path if needed (disable wraparound for eaten mode)
                self.path = self.pathfinding.find_shortest_path(
                    current_gx, current_gy, cage_gx, cage_gy, (0, 0), allow_wraparound=False
                )
                self.path_index = 0

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
        center_x = self.x + self.size // 2
        center_y = self.y + self.size // 2

        # Draw eaten ghosts (returning to cage) as eyes only
        if self.state == GhostState.EATEN:
            # calculate their color and mute it by 50%
            muted = (int(self.color[0] * 0.5), int(self.color[1] * 0.5), int(self.color[2] * 0.5))
            pygame.draw.circle(surface, muted, (int(center_x), int(center_y)), self.size // 4)
            return

        # Only draw spawned ghosts normally
        if self.is_spawned and self.color:
            # Flash between blue and white when frightened and warning
            if self.state == GhostState.FRIGHTENED and self.frightened_warning:
                import time
                flash = int(time.time() * 6) % 2 == 0  # Flash 6 times per second
                color = (255, 255, 255) if flash else (0, 0, 255)
                pygame.draw.circle(surface, color, (int(center_x), int(center_y)), self.size // 3)
            else:
                pygame.draw.circle(surface, self.color, (int(center_x), int(center_y)), self.size // 3)
        elif not self.is_spawned:
            # Draw spawning ghosts as dark gray
            pygame.draw.circle(surface, (50, 50, 50), (int(center_x), int(center_y)), self.size // 3)


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
