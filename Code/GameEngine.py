import pygame
from Code.Maze import Maze
from Code.PacMan import PacMan

class GameEngine:
    """Main game engine for Pac-Man gameplay"""

    def __init__(self, screen_width=800, screen_height=800, pacman_speed=2, paused=False):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.tile_size = 40

        # Initialize game objects
        self.maze = Maze(self.tile_size)
        self.pacman = PacMan(self.tile_size, self.tile_size, self.tile_size, speed=pacman_speed)

        # Game state
        self.game_over = False
        self.won = False
        self.pellets = self._initialize_pellets()
        self.power_ups = []
        self.paused = paused

        # Sound effects
        self.pellet_sound = None
        try:
            self.pellet_sound = pygame.mixer.Sound("../Audio/pacman_chomp.wav")
        except Exception as e:
            print(f"Could not load pellet sound: {e}")

    def _initialize_pellets(self):
        """Create pellets on all paths"""
        pellets = []
        for y in range(self.maze.height):
            for x in range(self.maze.width):
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

    def update(self):
        """Update game logic"""
        if self.game_over or self.won:
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

        # Draw Pac-Man
        if not self.paused:
            self.pacman.draw(surface)

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

