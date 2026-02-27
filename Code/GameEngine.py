from enum import Enum
import pygame
from Code.Maze import Maze
from Code.PacMan import PacMan
from Code.Pathfinding import Pathfinding
from Code.Ghost import Ghost, Pinky, Clyde, GhostState

class GameState(Enum):
    MENU = 1
    GAME = 2
    GAME_OVER = 3
    QUIT = 4
    AUDIO_PLAYING = 5

class GameEngine:
    def __init__(self, screen_width=800, screen_height=800, pacman_speed=2, paused=False, use_classic_maze=True, maze_algorithm="recursive_backtracking"):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.tile_size = 40

        maze_width = (screen_width // self.tile_size)
        maze_height = (screen_height // self.tile_size)

        if maze_width % 2 == 0: maze_width -= 1
        if maze_height % 2 == 0: maze_height -= 1

        self.maze = Maze(self.tile_size, width=maze_width, height=maze_height,
                         use_classic=use_classic_maze, algorithm=maze_algorithm)

        pacman_x, pacman_y = self._find_safe_spawn_bottom_center()
        self.pacman = PacMan(pacman_x, pacman_y, self.tile_size, speed=pacman_speed)

        self.ghosts = []
        self._initialize_ghosts()

        self.game_over = False
        self.won = False
        self.pellets = self._initialize_pellets()
        self.power_ups = []
        self.paused = paused

        self.pathfinding = Pathfinding(self.maze)
        self.show_path = True

        self.pellet_sound = None
        try:
            self.pellet_sound = pygame.mixer.Sound("../Audio/pacman_chomp.wav")
        except Exception as e:
            print(f"Audio Warning: {e}")

    def unpause(self):
        self.paused = False

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
        blinky.state = GhostState.CHASE
        self.ghosts.append(blinky)

        # Pinky (Pink) - targets 4 tiles ahead of Pac-Man
        # Spawn slightly offset from Blinky
        pinky = Pinky(cage_center_x + self.tile_size, cage_center_y, self.tile_size, speed=2, maze=self.maze, name="Pinky")
        pinky.color = (255, 184, 255)  # Pink color
        pinky.state = GhostState.CHASE
        self.ghosts.append(pinky)

        # Clyde (Orange) - chases Pac-Man but retreats when within 8 tiles
        # Spawn on the other side
        clyde = Clyde(cage_center_x - self.tile_size, cage_center_y, self.tile_size, speed=2, maze=self.maze, name="Clyde")
        clyde.color = (255, 184, 82)  # Orange color
        clyde.state = GhostState.CHASE
        self.ghosts.append(clyde)

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

        # if self.paused and not (self.game_over or self.won):
        #     pause_text = font.render("PAUSED", True, (255, 255, 0))
        #     surface.blit(pause_text, pause_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2)))

        if self.game_over:
            game_over_font = pygame.font.Font(None, 64)
            surface.blit(game_over_font.render("GAME OVER!", True, (255, 0, 0)), game_over_font.render("GAME OVER!", True, (255, 0, 0)).get_rect(center=(self.screen_width // 2, self.screen_height // 2)))

        if self.won:
            win_font = pygame.font.Font(None, 64)
            surface.blit(win_font.render("YOU WIN!", True, (0, 255, 0)), win_font.render("YOU WIN!", True, (0, 255, 0)).get_rect(center=(self.screen_width // 2, self.screen_height // 2)))