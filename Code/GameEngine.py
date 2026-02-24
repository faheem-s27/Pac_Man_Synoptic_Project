import pygame
from Code.Maze import Maze
from Code.PacMan import PacMan
from Code.Pathfinding import Pathfinding

class GameEngine:
    """Main game engine for Pac-Man gameplay"""

    def __init__(self, screen_width=800, screen_height=800, pacman_speed=2, paused=False, use_classic_maze=True, maze_algorithm="recursive_backtracking"):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.tile_size = 40

        # Calculate maze dimensions based on screen size
        # Ensure odd dimensions for proper centering (so there's a true center)
        maze_width = (screen_width // self.tile_size)
        maze_height = (screen_height // self.tile_size)

        # Make dimensions odd if they're even
        if maze_width % 2 == 0:
            maze_width -= 1
        if maze_height % 2 == 0:
            maze_height -= 1

        # Initialize game objects with procedural maze
        self.maze = Maze(self.tile_size, width=maze_width, height=maze_height,
                         use_classic=use_classic_maze, algorithm=maze_algorithm)

        # Find safe spawn position for Pac-Man at bottom center
        pacman_x, pacman_y = self._find_safe_spawn_bottom_center()
        self.pacman = PacMan(pacman_x, pacman_y, self.tile_size, speed=pacman_speed)

        # Game state
        self.game_over = False
        self.won = False
        self.pellets = self._initialize_pellets()
        self.power_ups = []
        self.paused = paused

        # Pathfinding system (tracks shortest path from top-left to Pac-Man)
        self.pathfinding = Pathfinding(self.maze)
        self.show_path = True  # Toggle to show/hide the path visualization

        # Sound effects
        self.pellet_sound = None
        try:
            self.pellet_sound = pygame.mixer.Sound("../Audio/pacman_chomp.wav")
        except Exception as e:
            print(f"Could not load pellet sound: {e}")

    def unpause(self):
        self.paused = False

    def _find_safe_spawn_bottom_center(self):
        """
        Find a safe spawn position for Pac-Man at the bottom center of the maze.
        Searches from bottom upward to find a valid path (0) not a wall (1).
        """
        center_x = self.maze.width // 2

        # Start from 3 rows from bottom and search upward
        for offset in range(0, self.maze.height // 3):
            y = self.maze.height - 3 - offset

            # Search horizontally around center
            for x_offset in range(0, self.maze.width // 4):
                # Try center first, then left, then right
                for test_x in [center_x, center_x - x_offset, center_x + x_offset]:
                    if 0 <= test_x < self.maze.width and 0 <= y < self.maze.height:
                        # Check if this position is a path (0) not a wall (1)
                        if self.maze.maze[y][test_x] == 0:
                            # Convert grid position to pixel position
                            pixel_x = test_x * self.tile_size
                            pixel_y = y * self.tile_size
                            return pixel_x, pixel_y

        # Fallback: spawn at top-left if no valid bottom-center position found
        print("Warning: No safe spawn position found at bottom, using fallback position")
        return self.tile_size, self.tile_size

    def _initialize_pellets(self):
        """Create pellets on all paths except in the ghost cage"""
        pellets = []

        # Calculate ghost cage boundaries (2x2 cage at center)
        cage_width = 4
        cage_height = 4
        cage_center_x = self.maze.width // 2
        cage_center_y = self.maze.height // 2
        cage_left = cage_center_x - cage_width // 2
        cage_top = cage_center_y - cage_height // 2
        cage_right = cage_left + cage_width
        cage_bottom = cage_top + cage_height

        for y in range(self.maze.height):
            for x in range(self.maze.width):
                # Skip if in ghost cage area
                if cage_left <= x < cage_right and cage_top <= y < cage_bottom:
                    continue

                if self.maze.maze[y][x] == 0:  # If it's a path
                    pellets.append((x * self.tile_size + self.tile_size // 2,
                                   y * self.tile_size + self.tile_size // 2))
        return pellets

    def handle_input(self, event):
        if self.paused:
            return
        """Handle keyboard input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.pacman.set_direction((0, -self.pacman.speed))
            elif event.key == pygame.K_DOWN:
                self.pacman.set_direction((0, self.pacman.speed))
            elif event.key == pygame.K_LEFT:
                self.pacman.set_direction((-self.pacman.speed, 0))
            elif event.key == pygame.K_RIGHT:
                self.pacman.set_direction((self.pacman.speed, 0))
            elif event.key == pygame.K_p:
                # Toggle path visualization
                self.show_path = not self.show_path
                print(f"Path visualization: {'ON' if self.show_path else 'OFF'}")

    def update(self):
        """Update game logic"""
        if self.game_over or self.won:
            return

        if self.paused:
            return

        # Update Pac-Man
        self.pacman.update(self.maze)

        # Check pellet collision
        pacman_x = self.pacman.x + self.pacman.size // 2
        pacman_y = self.pacman.y + self.pacman.size // 2

        pellets_to_remove = []
        for i, (px, py) in enumerate(self.pellets):
            distance = ((pacman_x - px) ** 2 + (pacman_y - py) ** 2) ** 0.5
            if distance < self.tile_size // 2:
                self.pacman.eat_pellet(10)
                # Play pellet sound effect
                if self.pellet_sound and not self.paused:
                    self.pellet_sound.play()
                pellets_to_remove.append(i)

        for i in reversed(pellets_to_remove):
            self.pellets.pop(i)

        # Check win condition
        if len(self.pellets) == 0:
            self.won = True

    def draw(self, surface):
        """Draw all game elements"""
        # Draw maze
        self.maze.draw(surface)

        # Draw pellets
        for px, py in self.pellets:
            pygame.draw.circle(surface, (184, 184, 184), (int(px), int(py)), 2)

        if self.paused:
            return

        # Draw Pac-Man
        self.pacman.draw(surface)

        # Draw shortest path visualization (from top-left to Pac-Man)
        if self.show_path:
            self.pathfinding.draw_path(surface, self.pacman.x, self.pacman.y, color=(255, 100, 100), line_width=2)

        # Draw score
        font = pygame.font.Font(None, 32)
        score_text = font.render(f"Score: {self.pacman.score}", True, (255, 255, 255))
        surface.blit(score_text, (10, 10))

        pellets_text = font.render(f"Pellets: {len(self.pellets)}", True, (255, 255, 255))
        surface.blit(pellets_text, (self.screen_width - 250, 10))

        # Draw game over message
        if self.game_over:
            game_over_font = pygame.font.Font(None, 64)
            game_over_text = game_over_font.render("GAME OVER!", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            surface.blit(game_over_text, text_rect)

        # Draw win message
        if self.won:
            win_font = pygame.font.Font(None, 64)
            win_text = win_font.render("YOU WIN!", True, (0, 255, 0))
            text_rect = win_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            surface.blit(win_text, text_rect)

