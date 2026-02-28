from enum import Enum
import pygame
from Code.Maze import Maze
from Code.PacMan import PacMan
from Code.Pathfinding import Pathfinding
from Code.Ghost import Ghost, Pinky, Clyde, Inky, GhostState

class GameState(Enum):
    MENU = 1
    GAME = 2
    GAME_OVER = 3
    QUIT = 4
    AUDIO_PLAYING = 5

class GameEngine:
    def __init__(self, screen_width=800, screen_height=800, pacman_speed=2, paused=False, use_classic_maze=True, maze_algorithm="recursive_backtracking", enable_ghosts=True):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.tile_size = 40
        self.enable_ghosts = enable_ghosts

        maze_width = (screen_width // self.tile_size)
        maze_height = (screen_height // self.tile_size)

        if maze_width % 2 == 0: maze_width -= 1
        if maze_height % 2 == 0: maze_height -= 1

        self.maze = Maze(self.tile_size, width=maze_width, height=maze_height,
                         use_classic=use_classic_maze, algorithm=maze_algorithm)

        pacman_x, pacman_y = self._find_safe_spawn_bottom_center()
        self.pacman = PacMan(pacman_x, pacman_y, self.tile_size, speed=pacman_speed)

        self.ghosts = []
        if self.enable_ghosts:
            self._initialize_ghosts()

        self.game_over = False
        self.won = False
        self.pellets = self._initialize_pellets()
        self.power_ups = []
        self.paused = paused

        self.pathfinding = Pathfinding(self.maze)
        self.show_path = True

        # Lives system
        self.lives = 3
        self.game_state = GameState.GAME
        self.life_lost_timer = 0
        self.life_lost_duration = 3 * 60  # 3 seconds at 60 FPS

        self.pellet_sound = None
        try:
            self.pellet_sound = pygame.mixer.Sound("../Audio/pacman_chomp.wav")
        except Exception as e:
            print(f"Audio Warning: {e}")

    def unpause(self):
        self.paused = False

    def _reset_positions(self):
        """Reset Pac-Man and ghosts to their starting positions after a life is lost."""
        pacman_x, pacman_y = self._find_safe_spawn_bottom_center()
        self.pacman.x = pacman_x
        self.pacman.y = pacman_y
        self.pacman.direction = (0, 0)

        if self.enable_ghosts:
            cage_center_x = (self.maze.width // 2) * self.tile_size
            cage_center_y = (self.maze.height // 2) * self.tile_size

            if len(self.ghosts) > 0:
                # Reset Blinky
                self.ghosts[0].x = cage_center_x + self.ghosts[0].offset
                self.ghosts[0].y = cage_center_y + self.ghosts[0].offset
                self.ghosts[0].current_dir = (0, 0)
            if len(self.ghosts) > 1:
                # Reset Pinky
                self.ghosts[1].x = cage_center_x + self.tile_size + self.ghosts[1].offset
                self.ghosts[1].y = cage_center_y + self.ghosts[1].offset
                self.ghosts[1].current_dir = (0, 0)
            if len(self.ghosts) > 2:
                # Reset Clyde
                self.ghosts[2].x = cage_center_x - self.tile_size + self.ghosts[2].offset
                self.ghosts[2].y = cage_center_y + self.ghosts[2].offset
                self.ghosts[2].current_dir = (0, 0)
            if len(self.ghosts) > 3:
                # Reset Inky
                self.ghosts[3].x = cage_center_x + self.ghosts[3].offset
                self.ghosts[3].y = cage_center_y + self.tile_size + self.ghosts[3].offset
                self.ghosts[3].current_dir = (0, 0)

    def _find_safe_spawn_bottom_center(self):
        center_x = self.maze.width // 2
        for offset in range(0, self.maze.height // 3):
            y = self.maze.height - 3 - offset
            for x_offset in range(0, self.maze.width // 4):
                for test_x in [center_x, center_x - x_offset, center_x + x_offset]:
                    if 0 <= test_x < self.maze.width and 0 <= y < self.maze.height:
                        if self.maze.maze[y][test_x] == 0:
                            return test_x * self.tile_size, y * self.tile_size
        return self.tile_size, self.tile_size

    def _initialize_ghosts(self):
        cage_center_x = (self.maze.width // 2) * self.tile_size
        cage_center_y = (self.maze.height // 2) * self.tile_size

        # Blinky (Red) - targets Pac-Man directly
        blinky = Ghost(cage_center_x, cage_center_y, self.tile_size, speed=2, maze=self.maze, name="Blinky")
        blinky.color = (255, 0, 0)
        self.ghosts.append(blinky)

        # Pinky (Pink) - targets 4 tiles ahead of Pac-Man
        # Spawn slightly offset from Blinky
        pinky = Pinky(cage_center_x + self.tile_size, cage_center_y, self.tile_size, speed=2, maze=self.maze, name="Pinky")
        pinky.color = (255, 184, 255)  # Pink color
        self.ghosts.append(pinky)

        # Clyde (Orange) - chases Pac-Man but retreats when within 8 tiles
        # Spawn on the other side
        clyde = Clyde(cage_center_x - self.tile_size, cage_center_y, self.tile_size, speed=2, maze=self.maze, name="Clyde")
        clyde.color = (255, 184, 82)  # Orange color
        self.ghosts.append(clyde)

        # Inky (Cyan) - targets based on vector from Blinky
        # Spawn below Blinky, pass Blinky reference for targeting
        inky = Inky(cage_center_x, cage_center_y + self.tile_size, self.tile_size, speed=2, maze=self.maze, name="Inky", blinky=blinky)
        inky.color = (0, 255, 255)  # Cyan color
        self.ghosts.append(inky)

    def _initialize_pellets(self):
        pellets = []
        cage_width, cage_height = 4, 4
        cage_center_x = self.maze.width // 2
        cage_center_y = self.maze.height // 2
        cage_left = cage_center_x - cage_width // 2
        cage_top = cage_center_y - cage_height // 2
        cage_right = cage_left + cage_width
        cage_bottom = cage_top + cage_height

        for y in range(self.maze.height):
            for x in range(self.maze.width):
                if cage_left <= x < cage_right and cage_top <= y < cage_bottom:
                    continue
                if self.maze.maze[y][x] == 0:
                    pellets.append((x * self.tile_size + self.tile_size // 2,
                                   y * self.tile_size + self.tile_size // 2))
        return pellets

    def handle_input(self, event):
        if self.paused: return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP: self.pacman.set_direction((0, -1))
            elif event.key == pygame.K_DOWN: self.pacman.set_direction((0, 1))
            elif event.key == pygame.K_LEFT: self.pacman.set_direction((-1, 0))
            elif event.key == pygame.K_RIGHT: self.pacman.set_direction((1, 0))
            elif event.key == pygame.K_p: self.show_path = not self.show_path

    def update(self):
        # Handle life lost timer and state transitions
        if self.game_state == GameState.AUDIO_PLAYING:
            self.life_lost_timer += 1
            if self.life_lost_timer >= self.life_lost_duration:
                # Transition back to GAME state
                if self.lives > 0:
                    self.game_state = GameState.GAME
                    self.life_lost_timer = 0
                else:
                    # No more lives, game over
                    self.game_over = True
                    self.game_state = GameState.GAME_OVER
            return

        if self.game_over or self.won or self.paused:
            return

        self.pacman.update(self.maze)

        for ghost in self.ghosts:
            ghost.update(self.pacman)

        pacman_x = self.pacman.x + self.pacman.size // 2
        pacman_y = self.pacman.y + self.pacman.size // 2
        collision_sq_threshold = (self.tile_size // 2) ** 2

        pellets_to_remove = []
        for i, (px, py) in enumerate(self.pellets):
            distance_sq = (pacman_x - px) ** 2 + (pacman_y - py) ** 2
            if distance_sq < collision_sq_threshold:
                self.pacman.eat_pellet(10)
                if self.pellet_sound:
                    pass
                pellets_to_remove.append(i)

        for i in reversed(pellets_to_remove):
            self.pellets.pop(i)

        if not self.pellets:
            self.won = True

        # Check ghost collision
        for ghost in self.ghosts:
            ghost_center_x = ghost.x + ghost.size // 2
            ghost_center_y = ghost.y + ghost.size // 2
            distance_sq = (pacman_x - ghost_center_x) ** 2 + (pacman_y - ghost_center_y) ** 2

            # Collision threshold: use tile_size / 2 as collision radius
            if distance_sq < collision_sq_threshold:
                # Lose a life
                self.lives -= 1
                if self.lives <= 0:
                    self.game_over = True
                    self.game_state = GameState.GAME_OVER
                else:
                    # Enter AUDIO_PLAYING state for 3 seconds
                    self.game_state = GameState.AUDIO_PLAYING
                    self.life_lost_timer = 0
                    self._reset_positions()
                break  # Only process one collision per frame

    def draw(self, surface):
        self.maze.draw(surface)

        for px, py in self.pellets:
            pygame.draw.circle(surface, (184, 184, 184), (int(px), int(py)), 2)

        self.pacman.draw(surface)

        for ghost in self.ghosts:
            ghost.draw(surface)

        if self.show_path and self.ghosts:
            for ghost in self.ghosts:
                if ghost.path:
                    ghost.pathfinding.draw_path(surface, ghost.path, ghost.color, line_width=2)

        font = pygame.font.Font(None, 32)
        surface.blit(font.render(f"Score: {self.pacman.score}", True, (255, 255, 255)), (10, 10))
        surface.blit(font.render(f"Pellets: {len(self.pellets)}", True, (255, 255, 255)), (self.screen_width - 250, 10))
        surface.blit(font.render(f"Lives: {self.lives}", True, (255, 255, 255)), (self.screen_width - 250, 50))

        # if self.paused and not (self.game_over or self.won):
        #     pause_text = font.render("PAUSED", True, (255, 255, 0))
        #     surface.blit(pause_text, pause_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2)))

        if self.game_over:
            game_over_font = pygame.font.Font(None, 64)
            surface.blit(game_over_font.render("GAME OVER!", True, (255, 0, 0)), game_over_font.render("GAME OVER!", True, (255, 0, 0)).get_rect(center=(self.screen_width // 2, self.screen_height // 2)))

        if self.won:
            win_font = pygame.font.Font(None, 64)
            surface.blit(win_font.render("YOU WIN!", True, (0, 255, 0)), win_font.render("YOU WIN!", True, (0, 255, 0)).get_rect(center=(self.screen_width // 2, self.screen_height // 2)))