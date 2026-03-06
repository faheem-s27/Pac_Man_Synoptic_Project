from enum import Enum
import pygame
from Code.Maze import Maze
from Code.PacMan import PacMan
from Code.Pathfinding import Pathfinding
from Code.Ghost import Ghost, Pinky, Clyde, Inky, GhostState

def parse_resolution(resolution_str):
    """Parse resolution string like '800x800' to (width, height)."""
    try:
        w, h = resolution_str.split('x')
        return int(w), int(h)
    except:
        return 800, 800

class GameState(Enum):
    MENU = 1
    GAME = 2
    GAME_OVER = 3
    QUIT = 4
    AUDIO_PLAYING = 5

class GameEngine:
    def __init__(self, screen_width=800, screen_height=800, pacman_speed=2, ghost_speed=-1,
                 paused=False, use_classic_maze=True,
                 maze_algorithm="recursive_backtracking",
                 enable_ghosts=True, tile_size=40, lives=3,
                 window_resolution="800x800",
                 god_mode=False, max_pellets=-1, pellets_to_win=-1,
                 scatter_duration=10, chase_duration=20,
                 always_chase=False,
                 level=1, ghost_speed_increment=0.1,
                 maze_seed=None,
                 enable_sound=True,
                 **kwargs):
        # Parse resolution string if provided
        if isinstance(window_resolution, str):
            screen_width, screen_height = parse_resolution(window_resolution)

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.tile_size = tile_size
        self.enable_ghosts = enable_ghosts
        self.god_mode = god_mode
        self.max_pellets = max_pellets  # legacy, unused — kept for **kwargs safety
        self.pellets_to_win = pellets_to_win  # -1 = eat all pellets to win
        self.pellets_eaten_this_level = 0
        self.pacman_speed = pacman_speed

        # Level system
        self.level = level
        self.ghost_speed_increment = ghost_speed_increment
        # If ghost_speed is -1, derive it automatically as pacman_speed - 0.1
        resolved_ghost_speed = (pacman_speed - 0.1) if ghost_speed < 0 else ghost_speed
        self.base_ghost_speed = resolved_ghost_speed
        # Each level beyond 1 adds ghost_speed_increment to ghost speed
        self.ghost_speed = resolved_ghost_speed + (level - 1) * ghost_speed_increment
        print(f"[Speed] Pac-Man: {pacman_speed}, Ghost base: {resolved_ghost_speed:.2f}, "
              f"Level {level} ghost speed: {self.ghost_speed:.2f}")

        # Global scatter/chase mode — must be set before _initialize_ghosts
        self.always_chase = always_chase
        self.scatter_chase_timer = 0
        self.scatter_duration = scatter_duration * 60   # Convert seconds to frames (60 FPS)
        self.chase_duration = chase_duration * 60       # Convert seconds to frames (60 FPS)
        self.global_scatter_mode = False if always_chase else True

        maze_width = (screen_width // self.tile_size)
        maze_height = (screen_height // self.tile_size)

        if maze_width % 2 == 0: maze_width -= 1
        if maze_height % 2 == 0: maze_height -= 1

        self.use_classic_maze = use_classic_maze
        self.maze_algorithm = maze_algorithm
        self.maze_seed = maze_seed
        self.maze = Maze(self.tile_size, width=maze_width, height=maze_height,
                         use_classic=use_classic_maze, algorithm=maze_algorithm, seed=maze_seed)

        pacman_x, pacman_y = self._find_safe_spawn_bottom_center()
        self.pacman = PacMan(pacman_x, pacman_y, self.tile_size, speed=pacman_speed)

        self.ghosts = []
        if self.enable_ghosts:
            self._initialize_ghosts()
            self._sync_ghost_modes()  # Apply always_chase / initial scatter mode

        self.game_over = False
        self.won = False
        self.pellets = self._initialize_pellets()
        self.power_pellets = self._initialize_power_pellets()
        self.paused = paused

        self.pathfinding = Pathfinding(self.maze)
        self.show_path = True

        # Lives system
        self.lives = lives
        self.game_state = GameState.GAME
        self.life_lost_timer = 0
        self.life_lost_duration = 3 * 60  # 3 seconds at 60 FPS

        # Power pellet / Frightened mode system
        self.frightened_mode = False
        self.frightened_timer = 0
        self.frightened_duration = 10 * 60  # 10 seconds at 60 FPS
        self.frightened_warning_threshold = 3 * 60  # Start flashing 3 seconds before end
        self.ghosts_eaten_combo = 0  # Track combo for increasing points


        self.pellet_sounds = []
        self.pellet_sound_index = 0
        self.pellet_channel = None
        self.death_sound = None
        self.death_channel = None
        self.ghost_turn_blue_sound = None
        self.ghost_turn_blue_channel = None
        self.ghost_return_to_cage_sound = None
        self.ghost_return_to_cage_channel = None
        self.eat_ghost_sound = None
        self.eat_ghost_channel = None

        if enable_sound:
            try:
                self.pellet_sounds = [
                    pygame.mixer.Sound("../Audio/eat_dot_0.wav"),
                    pygame.mixer.Sound("../Audio/eat_dot_1.wav"),
                ]
                self.pellet_channel = pygame.mixer.Channel(0)
            except Exception as e:
                print(f"Audio Warning (pellet): {e}")

            try:
                self.death_sound = pygame.mixer.Sound("../Audio/pacman_death.mp3")
                self.death_channel = pygame.mixer.Channel(1)
            except Exception as e:
                print(f"Audio Warning (death): {e}")

            try:
                self.ghost_turn_blue_sound = pygame.mixer.Sound("../Audio/ghost_turn_blue.mp3")
                self.ghost_turn_blue_channel = pygame.mixer.Channel(2)
            except Exception as e:
                print(f"Audio Warning (ghost turn blue): {e}")

            try:
                self.ghost_return_to_cage_sound = pygame.mixer.Sound("../Audio/ghost_return_to_cage.mp3")
                self.ghost_return_to_cage_channel = pygame.mixer.Channel(3)
            except Exception as e:
                print(f"Audio Warning (ghost return to cage): {e}")

            try:
                self.eat_ghost_sound = pygame.mixer.Sound("../Audio/pacman_eatghost.wav")
                self.eat_ghost_channel = pygame.mixer.Channel(4)
            except Exception as e:
                print(f"Audio Warning (eat ghost): {e}")

    def unpause(self):
        self.paused = False

    def next_level(self):
        """Advance to the next level — increase ghost speed and reset the board."""
        self.level += 1
        self.ghost_speed = self.base_ghost_speed + (self.level - 1) * self.ghost_speed_increment
        print(f"[Level] Advancing to level {self.level} — ghost speed: {self.ghost_speed:.2f}")

        # Update speed on existing ghosts
        for ghost in self.ghosts:
            ghost.speed = self.ghost_speed
            ghost.original_speed = self.ghost_speed
            ghost.eaten_speed = self.ghost_speed * 2

        # Regenerate maze — use the stored seed so the layout stays constant across levels
        self.maze = Maze(self.tile_size, width=self.maze.width, height=self.maze.height,
                         use_classic=self.use_classic_maze, algorithm=self.maze_algorithm, seed=self.maze_seed)
        self.pellets = self._initialize_pellets()
        self.power_pellets = self._initialize_power_pellets()

        # Reset pacman and ghost positions
        self.won = False
        self.game_over = False
        self.frightened_mode = False
        self.frightened_timer = 0
        self.ghosts_eaten_combo = 0
        self.scatter_chase_timer = 0
        self.global_scatter_mode = False if self.always_chase else True
        self.pellets_eaten_this_level = 0

        pacman_x, pacman_y = self._find_safe_spawn_bottom_center()
        self.pacman.x = pacman_x
        self.pacman.y = pacman_y
        self.pacman.direction = (0, 0)
        self.pacman.next_direction = (0, 0)

        if self.enable_ghosts:
            ts = self.tile_size
            m  = self.maze
            interior_py = (m.cage_top + 1) * ts
            interior_centre_x = (m.cage_left + m.cage_right) // 2
            blinky_px = float(m.door_x * ts)
            blinky_py = float((m.cage_top - 1) * ts)

            for ghost in self.ghosts:
                ghost.maze = self.maze
                ghost.pathfinding = Pathfinding(self.maze)
            if len(self.ghosts) > 0:
                self.ghosts[0].x = blinky_px
                self.ghosts[0].y = blinky_py
                self.ghosts[0].reset_spawn()
            if len(self.ghosts) > 1:
                self.ghosts[1].x = float(interior_centre_x * ts)
                self.ghosts[1].y = float(interior_py)
                self.ghosts[1].reset_spawn()
            if len(self.ghosts) > 2:
                self.ghosts[2].x = float((interior_centre_x - 1) * ts)
                self.ghosts[2].y = float(interior_py)
                self.ghosts[2].reset_spawn()
            if len(self.ghosts) > 3:
                self.ghosts[3].x = float((interior_centre_x + 1) * ts)
                self.ghosts[3].y = float(interior_py)
                self.ghosts[3].reset_spawn()
            self._sync_ghost_modes()

        self.pathfinding = Pathfinding(self.maze)

    def _reset_positions(self):
        """Reset Pac-Man and ghosts to their starting positions after a life is lost."""
        pacman_x, pacman_y = self._find_safe_spawn_bottom_center()
        self.pacman.x = pacman_x
        self.pacman.y = pacman_y
        self.pacman.direction = (0, 0)

        # Reset global scatter/chase timer
        self.scatter_chase_timer = 0
        self.global_scatter_mode = False if self.always_chase else True

        if self.enable_ghosts:
            ts = self.tile_size
            m  = self.maze
            interior_py = (m.cage_top + 1) * ts
            interior_centre_x = (m.cage_left + m.cage_right) // 2

            if len(self.ghosts) > 0:
                # Reset Blinky — above door
                self.ghosts[0].x = float(m.door_x * ts)
                self.ghosts[0].y = float((m.cage_top - 1) * ts)
                self.ghosts[0].reset_spawn()
            if len(self.ghosts) > 1:
                # Reset Pinky — cage interior centre
                self.ghosts[1].x = float(interior_centre_x * ts)
                self.ghosts[1].y = float(interior_py)
                self.ghosts[1].reset_spawn()
            if len(self.ghosts) > 2:
                # Reset Inky — left of centre
                self.ghosts[2].x = float((interior_centre_x - 1) * ts)
                self.ghosts[2].y = float(interior_py)
                self.ghosts[2].reset_spawn()
            if len(self.ghosts) > 3:
                # Reset Clyde — right of centre
                self.ghosts[3].x = float((interior_centre_x + 1) * ts)
                self.ghosts[3].y = float(interior_py)
                self.ghosts[3].reset_spawn()
            self._sync_ghost_modes()  # Reapply always_chase / scatter mode after reset

    def _sync_ghost_modes(self):
        """Stamp the current global scatter/chase mode onto every ghost's is_scatter flag."""
        for ghost in self.ghosts:
            ghost.is_scatter = self.global_scatter_mode

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
        ts = self.tile_size
        m  = self.maze

        # Blinky starts just ABOVE the door (outside the cage) — exits immediately
        blinky_gx = m.door_x
        blinky_gy = m.cage_top - 1          # one tile above the door
        blinky_px = blinky_gx * ts
        blinky_py = blinky_gy * ts

        # Interior ghosts sit on the single interior row of the cage
        interior_y  = m.cage_top + 1        # the open row inside the cage
        interior_py = interior_y * ts

        # Centre the three interior ghosts across the cage interior
        # cage is 6 wide, interior is 4 wide (cage_left+1 .. cage_right-1)
        interior_centre_x = (m.cage_left + m.cage_right) // 2
        pinky_gx  = interior_centre_x
        inky_gx   = interior_centre_x - 1
        clyde_gx  = interior_centre_x + 1

        pinky_px  = pinky_gx  * ts
        inky_px   = inky_gx   * ts
        clyde_px  = clyde_gx  * ts

        # Return-to-cage target: interior centre
        cage_home_px = interior_centre_x * ts
        cage_home_py = interior_py

        # Blinky (Red) — spawns immediately, starts above cage
        blinky = Ghost(blinky_px, blinky_py, ts, speed=self.ghost_speed, maze=self.maze, name="Blinky")
        blinky.color     = (255, 0, 0)
        blinky.spawn_delay = 0
        blinky.cage_x    = cage_home_px
        blinky.cage_y    = cage_home_py
        self.ghosts.append(blinky)

        # Pinky (Pink) — centre of cage interior
        pinky = Pinky(pinky_px, interior_py, ts, speed=self.ghost_speed, maze=self.maze, name="Pinky")
        pinky.color      = (255, 184, 255)
        pinky.spawn_delay = 5 * 60
        pinky.cage_x     = pinky_px
        pinky.cage_y     = interior_py
        self.ghosts.append(pinky)

        # Inky (Cyan) — left of centre
        inky = Inky(inky_px, interior_py, ts, speed=self.ghost_speed, maze=self.maze, name="Inky", blinky=blinky)
        inky.color       = (0, 255, 255)
        inky.spawn_delay = 10 * 60
        inky.cage_x      = inky_px
        inky.cage_y      = interior_py
        self.ghosts.append(inky)

        # Clyde (Orange) — right of centre
        clyde = Clyde(clyde_px, interior_py, ts, speed=self.ghost_speed, maze=self.maze, name="Clyde")
        clyde.color      = (255, 184, 82)
        clyde.spawn_delay = 15 * 60
        clyde.cage_x     = clyde_px
        clyde.cage_y     = interior_py
        self.ghosts.append(clyde)

    def _initialize_pellets(self):
        pellets = []

        # Exclude the entire cage box (including one tile margin around it)
        cage_exclusion = set()
        if self.enable_ghosts:
            m = self.maze
            for gy in range(m.cage_top - 1, m.cage_bottom + 2):
                for gx in range(m.cage_left - 1, m.cage_right + 2):
                    cage_exclusion.add((gx, gy))

        for y in range(self.maze.height):
            for x in range(self.maze.width):
                if (x, y) in cage_exclusion:
                    continue
                if self.maze.maze[y][x] == 0:
                    pellets.append((x * self.tile_size + self.tile_size // 2,
                                    y * self.tile_size + self.tile_size // 2))

        total_possible = len(pellets)
        print(f"[Pellets] Placing all {total_possible} pellets. "
              f"Pellets to win: {'all' if self.pellets_to_win < 0 else self.pellets_to_win}.")
        return pellets

    def _initialize_power_pellets(self):
        """Initialize power pellets dynamically based on maze size."""
        power_pellets = []

        # Calculate number of power pellets based on maze area
        # Formula: 1 power pellet per ~400 tiles (adjustable)
        maze_area = self.maze.width * self.maze.height
        tiles_per_pellet = 400  # Adjust this to control density

        # Minimum 4, maximum 12 power pellets for balance
        num_power_pellets = max(4, min(12, int(maze_area / tiles_per_pellet)))

        # Define strategic positions based on maze divisions
        # Divide maze into grid sections and place power pellets in corners of sections
        if num_power_pellets == 4:
            # Classic 4-corner layout for small mazes
            corner_positions = [
                (1, 1, 1, 1),                                        # Top-left
                (self.maze.width - 2, 1, -1, 1),                    # Top-right
                (1, self.maze.height - 2, 1, -1),                   # Bottom-left
                (self.maze.width - 2, self.maze.height - 2, -1, -1) # Bottom-right
            ]
        else:
            # For larger mazes, distribute power pellets more evenly
            corner_positions = []

            # Calculate grid divisions (2x2, 2x3, 3x3, 3x4, 4x4 based on count)
            if num_power_pellets <= 4:
                grid_x, grid_y = 2, 2
            elif num_power_pellets <= 6:
                grid_x, grid_y = 2, 3
            elif num_power_pellets <= 9:
                grid_x, grid_y = 3, 3
            elif num_power_pellets <= 12:
                grid_x, grid_y = 3, 4
            else:
                grid_x, grid_y = 4, 4

            # Calculate section sizes
            section_width = self.maze.width // grid_x
            section_height = self.maze.height // grid_y

            # Place power pellets in corners of grid sections
            pellets_placed = 0
            for gy in range(grid_y):
                for gx in range(grid_x):
                    if pellets_placed >= num_power_pellets:
                        break

                    # Determine which corner of this section to use
                    # Alternate pattern for variety
                    if (gx + gy) % 2 == 0:
                        # Place in top-left of section
                        start_x = gx * section_width + 1
                        start_y = gy * section_height + 1
                        dx, dy = 1, 1
                    else:
                        # Place in bottom-right of section
                        start_x = (gx + 1) * section_width - 2
                        start_y = (gy + 1) * section_height - 2
                        dx, dy = -1, -1

                    corner_positions.append((start_x, start_y, dx, dy))
                    pellets_placed += 1

                if pellets_placed >= num_power_pellets:
                    break

        # Search radius scales with tile size (larger tiles = larger search area)
        search_radius = max(5, int(10 / (self.tile_size / 20)))

        # Place power pellets at calculated positions
        for start_x, start_y, dx, dy in corner_positions:
            # Search for the nearest open space from this position
            found = False
            for radius in range(search_radius):
                if found:
                    break
                for offset_x in range(radius + 1):
                    offset_y = radius - offset_x
                    test_x = start_x + (offset_x * dx)
                    test_y = start_y + (offset_y * dy)

                    # Check if within bounds and not a wall
                    if 0 <= test_x < self.maze.width and 0 <= test_y < self.maze.height:
                        if self.maze.maze[test_y][test_x] == 0:
                            # Found valid position, add power pellet
                            px = test_x * self.tile_size + self.tile_size // 2
                            py = test_y * self.tile_size + self.tile_size // 2
                            power_pellets.append((px, py))

                            # Remove regular pellet from this position if it exists
                            pellet_to_remove = (px, py)
                            if pellet_to_remove in self.pellets:
                                self.pellets.remove(pellet_to_remove)

                            found = True
                            break

        print(f"Power Pellets: {len(power_pellets)} placed (maze: {self.maze.width}x{self.maze.height}, tile: {self.tile_size}px)")
        return power_pellets

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

        # Update global scatter/chase timer (when not in frightened mode)
        if not self.frightened_mode and not self.always_chase:
            self.scatter_chase_timer += 1

            # Switch modes globally for all ghosts
            if self.global_scatter_mode and self.scatter_chase_timer >= self.scatter_duration:
                # Switch from scatter to chase
                self.global_scatter_mode = False
                self.scatter_chase_timer = 0
                for ghost in self.ghosts:
                    if ghost.is_spawned:
                        ghost.set_mode(False)  # Chase mode
            elif not self.global_scatter_mode and self.scatter_chase_timer >= self.chase_duration:
                # Switch from chase to scatter
                self.global_scatter_mode = True
                self.scatter_chase_timer = 0
                for ghost in self.ghosts:
                    if ghost.is_spawned:
                        ghost.set_mode(True)  # Scatter mode

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
                self.pellets_eaten_this_level += 1
                # Play alternating pellet sounds only if channel is not busy
                if self.pellet_sounds and self.pellet_channel and not self.pellet_channel.get_busy():
                    sound = self.pellet_sounds[self.pellet_sound_index % len(self.pellet_sounds)]
                    self.pellet_channel.play(sound)
                    self.pellet_sound_index += 1
                pellets_to_remove.append(i)

        for i in reversed(pellets_to_remove):
            self.pellets.pop(i)

        # Check power pellet collision
        power_pellets_to_remove = []
        for i, (px, py) in enumerate(self.power_pellets):
            distance_sq = (pacman_x - px) ** 2 + (pacman_y - py) ** 2
            if distance_sq < collision_sq_threshold:
                self.pacman.eat_pellet(50)  # Power pellets worth more points
                power_pellets_to_remove.append(i)

                if not self.frightened_mode:
                    # First power pellet - activate frightened mode
                    self.frightened_mode = True
                    self.frightened_timer = 0
                    self.ghosts_eaten_combo = 0  # Reset combo counter

                    # Play frightened entry sound once per frightened cycle
                    if self.ghost_turn_blue_sound and self.ghost_turn_blue_channel:
                        self.ghost_turn_blue_channel.play(self.ghost_turn_blue_sound)

                    # Set all spawned ghosts to frightened state
                    for ghost in self.ghosts:
                        if ghost.is_spawned and ghost.state != GhostState.SPAWNING:
                            ghost.enter_frightened_mode()
                else:
                    # Already in frightened mode - extend the duration
                    # Reset timer to extend duration, but keep the combo
                    self.frightened_timer = 0

                    # Only frighten ghosts that aren't already frightened or spawning
                    for ghost in self.ghosts:
                        if ghost.is_spawned and ghost.state != GhostState.FRIGHTENED and ghost.state != GhostState.SPAWNING:
                            ghost.enter_frightened_mode()

        for i in reversed(power_pellets_to_remove):
            self.power_pellets.pop(i)

        # Update frightened mode timer
        if self.frightened_mode:
            self.frightened_timer += 1

            # Set warning flag on ghosts when timer is low
            is_warning = self.frightened_timer >= (self.frightened_duration - self.frightened_warning_threshold)
            for ghost in self.ghosts:
                if ghost.is_spawned and ghost.state == GhostState.FRIGHTENED:
                    ghost.frightened_warning = is_warning

            if self.frightened_timer >= self.frightened_duration:
                # End frightened mode
                self.frightened_mode = False
                self.frightened_timer = 0
                for ghost in self.ghosts:
                    if ghost.is_spawned:
                        ghost.exit_frightened_mode()
                        ghost.frightened_warning = False
                        # Restore global scatter/chase mode
                        ghost.set_mode(self.global_scatter_mode)

        # Win condition — eat pellets_to_win pellets, or all pellets if -1
        total_spawned = len(self.pellets) + len(self.power_pellets) + self.pellets_eaten_this_level
        if self.pellets_to_win < 0:
            # Classic: eat every pellet
            if not self.pellets and not self.power_pellets:
                self.won = True
        else:
            target = min(self.pellets_to_win, total_spawned)
            if self.pellets_to_win > total_spawned:
                print(f"[Pellets] pellets_to_win ({self.pellets_to_win}) exceeds total spawned "
                      f"({total_spawned}). Winning when all {total_spawned} are eaten.")
            if self.pellets_eaten_this_level >= target:
                self.won = True

        # Check ghost collision
        for ghost in self.ghosts:
            # Only check collision with spawned ghosts (not eaten ghosts returning to cage)
            if not ghost.is_spawned or ghost.state == GhostState.EATEN:
                continue

            ghost_center_x = ghost.x + ghost.size // 2
            ghost_center_y = ghost.y + ghost.size // 2
            distance_sq = (pacman_x - ghost_center_x) ** 2 + (pacman_y - ghost_center_y) ** 2

            # Collision threshold: use tile_size / 2 as collision radius
            if distance_sq < collision_sq_threshold:
                if self.frightened_mode and ghost.state == GhostState.FRIGHTENED:
                    # Eat the ghost!
                    ghost_points = 200 * (2 ** self.ghosts_eaten_combo)  # 200, 400, 800, 1600
                    self.pacman.eat_pellet(ghost_points)
                    self.ghosts_eaten_combo += 1

                    # Play eat ghost sound
                    if self.eat_ghost_sound and self.eat_ghost_channel:
                        self.eat_ghost_channel.play(self.eat_ghost_sound)

                    # Play sound for ghost entering return-to-cage state
                    if self.ghost_return_to_cage_sound and self.ghost_return_to_cage_channel:
                        self.ghost_return_to_cage_channel.play(self.ghost_return_to_cage_sound)

                    # Enter eaten mode - ghost will return to cage then respawn
                    ghost.enter_eaten_mode()

                elif not self.god_mode:
                    # Ghost catches Pac-Man - lose a life
                    self.lives -= 1

                    # Play death sound
                    if self.death_sound and self.death_channel:
                        self.death_channel.play(self.death_sound)

                    if self.lives <= 0:
                        self.game_over = True
                        self.game_state = GameState.GAME_OVER
                    else:
                        # Enter AUDIO_PLAYING state for 3 seconds
                        self.game_state = GameState.AUDIO_PLAYING
                        self.life_lost_timer = 0
                        self._reset_positions()

                        # Reset frightened mode when life is lost
                        self.frightened_mode = False
                        self.frightened_timer = 0
                    break  # Only process one collision per frame

    def draw(self, surface):
        if surface is None:
            return  # Headless — skip all rendering and image loading

        self.maze.draw(surface)

        # Draw regular pellets
        for px, py in self.pellets:
            pygame.draw.circle(surface, (184, 184, 184), (int(px), int(py)), 2)

        # Draw power pellets with flashing effect
        import time
        flash = int(time.time() * 3) % 2 == 0  # Flash 3 times per second
        for px, py in self.power_pellets:
            if flash:
                pygame.draw.circle(surface, (255, 255, 200), (int(px), int(py)), 6)
            else:
                pygame.draw.circle(surface, (255, 255, 100), (int(px), int(py)), 5)

        self.pacman.draw(surface)

        for ghost in self.ghosts:
            ghost.draw(surface)

        if self.show_path and self.ghosts:
            for ghost in self.ghosts:
                if ghost.path:
                    ghost.pathfinding.draw_path(surface, ghost.path, ghost.color, line_width=2)

        font = pygame.font.Font(None, 32)
        surface.blit(font.render(f"Score: {self.pacman.score}", True, (255, 255, 255)), (10, 10))
        surface.blit(font.render(f"Level: {self.level}", True, (255, 255, 0)), (10, 40))
        surface.blit(font.render(f"Lives: {self.lives}", True, (255, 255, 255)), (self.screen_width - 250, 50))

        if self.pellets_to_win >= 0:
            total_spawned = len(self.pellets) + len(self.power_pellets) + self.pellets_eaten_this_level
            target = min(self.pellets_to_win, total_spawned)
            surface.blit(font.render(f"Pellets: {self.pellets_eaten_this_level}/{target}", True, (255, 255, 255)), (self.screen_width - 250, 10))
        else:
            surface.blit(font.render(f"Pellets: {len(self.pellets) + len(self.power_pellets)}", True, (255, 255, 255)), (self.screen_width - 250, 10))

        # Display frightened mode timer
        if self.frightened_mode:
            time_left = (self.frightened_duration - self.frightened_timer) // 60
            is_warning = self.frightened_timer >= (self.frightened_duration - self.frightened_warning_threshold)
            color = (255, 100, 100) if is_warning else (100, 200, 255)
            surface.blit(font.render(f"Power Mode: {time_left}s", True, color), (10, 50))

        # if self.paused and not (self.game_over or self.won):
        #     pause_text = font.render("PAUSED", True, (255, 255, 0))
        #     surface.blit(pause_text, pause_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2)))

        if self.game_over:
            game_over_font = pygame.font.Font(None, 64)
            surface.blit(game_over_font.render("GAME OVER!", True, (255, 0, 0)), game_over_font.render("GAME OVER!", True, (255, 0, 0)).get_rect(center=(self.screen_width // 2, self.screen_height // 2)))

        if self.won:
            win_font = pygame.font.Font(None, 64)
            surface.blit(win_font.render("YOU WIN!", True, (0, 255, 0)), win_font.render("YOU WIN!", True, (0, 255, 0)).get_rect(center=(self.screen_width // 2, self.screen_height // 2)))