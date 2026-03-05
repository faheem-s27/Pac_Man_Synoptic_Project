import pygame
from PIL import Image
import os


class PacMan:
    ANIMATION_FRAME_DELAY = 6  # 60 FPS / 6 ~= 10 animation frames per second
    VISUAL_SCALE = 0.82

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

        self.render_size = max(8, int(self.size * self.VISUAL_SCALE))
        self.render_offset = (self.size - self.render_size) // 2

        self.pacman_images = {}
        self.animation_counter = 0
        self.last_facing_direction = "right"
        self._load_pacman_images()

    def _load_pacman_images(self):
        """Load directional GIF images for Pac-Man and extract all frames."""
        directions = ["up", "down", "left", "right"]
        images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Images"))

        for direction in directions:
            try:
                gif_path = os.path.join(images_dir, f"pacman_{direction}.gif")
                frames = []
                pil_image = Image.open(gif_path)

                frame_index = 0
                while True:
                    try:
                        pil_image.seek(frame_index)
                        frame = pil_image.convert("RGBA")
                        frame_data = pygame.image.fromstring(frame.tobytes(), frame.size, "RGBA")
                        frame_scaled = pygame.transform.scale(frame_data, (self.render_size, self.render_size))
                        frames.append(frame_scaled)
                        frame_index += 1
                    except EOFError:
                        break

                if frames:
                    self.pacman_images[direction] = frames
                    print(f"Loaded {len(frames)} frames for Pac-Man {direction} animation")
            except Exception as e:
                print(f"Note: Could not load Pac-Man {direction} GIF: {e}")

    def _get_current_direction_name(self):
        dx, dy = self.direction

        if dx == 0 and dy == 0:
            return self.last_facing_direction

        if abs(dy) > abs(dx):
            direction = "up" if dy < 0 else "down"
        else:
            direction = "left" if dx < 0 else "right"

        self.last_facing_direction = direction
        return direction

    def is_aligned_to_tile(self):
        center_x = self.x + self.size // 2
        center_y = self.y + self.size // 2
        tile_offset_x = center_x % self.tile_size
        tile_offset_y = center_y % self.tile_size

        # Tolerance must be at least as large as speed so we never overshoot without triggering
        tolerance = max(self.speed, self.tile_size // 10, 3)

        return (abs(tile_offset_x - self.tile_size // 2) < tolerance and
                abs(tile_offset_y - self.tile_size // 2) < tolerance)

    def _snap_to_axis(self):
        """Snap Pac-Man's position along the non-moving axis to the nearest tile centre.
        This prevents sub-pixel drift that blocks turns at non-divisible speeds."""
        dx, dy = self.direction
        half = self.tile_size // 2

        if dx != 0:  # Moving horizontally — snap Y to tile centre
            center_y = self.y + self.size // 2
            snapped_tile_y = (center_y // self.tile_size) * self.tile_size + half
            self.y = snapped_tile_y - self.size // 2

        elif dy != 0:  # Moving vertically — snap X to tile centre
            center_x = self.x + self.size // 2
            snapped_tile_x = (center_x // self.tile_size) * self.tile_size + half
            self.x = snapped_tile_x - self.size // 2

    def set_direction(self, direction):
        """Allow direction changes more freely - don't require perfect tile alignment"""
        # Always allow storing the next direction
        # The update() method will apply it when safe
        self.next_direction = direction

    def update(self, maze):
        self.animation_counter += 1

        if self.next_direction != (0, 0):
            is_reverse = (self.next_direction[0] == -self.direction[0] and
                          self.next_direction[1] == -self.direction[1])

            if is_reverse:
                # Reverse direction instantly — no tile alignment needed
                self.direction = self.next_direction
                self.next_direction = (0, 0)
            elif self.is_aligned_to_tile():
                # Snap to the tile centre first so the move check is clean
                self._snap_to_axis()
                next_x = self.x + self.next_direction[0] * self.speed
                next_y = self.y + self.next_direction[1] * self.speed

                if maze.can_move(next_x, next_y, self.size):
                    self.direction = self.next_direction
                    self.next_direction = (0, 0)

        # Move — but clamp each step so we never overshoot a tile centre
        dx, dy = self.direction
        if dx != 0:
            # Moving horizontally: advance X, keep Y snapped
            next_x = self.x + dx * self.speed
            next_y = self.y
            tile_half = self.tile_size // 2
            if maze.can_move(next_x, next_y, self.size):
                # Clamp: don't overshoot the next tile centre
                center_x = self.x + self.size // 2
                next_center_x = next_x + self.size // 2
                tile_half = self.tile_size // 2
                cur_offset = center_x % self.tile_size
                nxt_offset = next_center_x % self.tile_size
                # If we crossed the tile-centre boundary, snap to it
                if (cur_offset < tile_half <= nxt_offset) or (cur_offset > tile_half >= nxt_offset):
                    snapped = (center_x // self.tile_size) * self.tile_size + tile_half
                    if abs(next_center_x - snapped) <= self.speed:
                        next_x = snapped - self.size // 2
                self.x = next_x
            else:
                # Snap to the tile centre so we sit flush against the wall
                center_x = self.x + self.size // 2
                snapped = (center_x // self.tile_size) * self.tile_size + tile_half
                self.x = snapped - self.size // 2

        elif dy != 0:
            # Moving vertically: advance Y, keep X snapped
            next_x = self.x
            next_y = self.y + dy * self.speed
            if maze.can_move(next_x, next_y, self.size):
                # Clamp: don't overshoot the next tile centre
                center_y = self.y + self.size // 2
                next_center_y = next_y + self.size // 2
                tile_half = self.tile_size // 2
                cur_offset = center_y % self.tile_size
                nxt_offset = next_center_y % self.tile_size
                if (cur_offset < tile_half <= nxt_offset) or (cur_offset > tile_half >= nxt_offset):
                    snapped = (center_y // self.tile_size) * self.tile_size + tile_half
                    if abs(next_center_y - snapped) <= self.speed:
                        next_y = snapped - self.size // 2
                self.y = next_y
            else:
                # Snap to the tile centre so we sit flush against the wall
                center_y = self.y + self.size // 2
                tile_half = self.tile_size // 2
                snapped = (center_y // self.tile_size) * self.tile_size + tile_half
                self.y = snapped - self.size // 2

        # Handle teleportation at maze edges
        self.x, self.y = maze.handle_teleportation(self.x, self.y)

    def draw(self, surface):
        center_x = self.x + self.size // 2
        center_y = self.y + self.size // 2

        if self.pacman_images:
            direction = self._get_current_direction_name()
            if direction in self.pacman_images and self.pacman_images[direction]:
                frames = self.pacman_images[direction]
                frame_index = (self.animation_counter // self.ANIMATION_FRAME_DELAY) % len(frames)
                image = frames[frame_index]
                surface.blit(image, (self.x + self.render_offset, self.y + self.render_offset))
                return

        pygame.draw.circle(surface, self.color, (int(center_x), int(center_y)), self.render_size // 3)

    def get_grid_position(self):
        return self.x // self.tile_size, self.y // self.tile_size

    def eat_pellet(self, points=10):
        self.score += points
        self.pellets_eaten += 1

