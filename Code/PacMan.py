import pygame
from PIL import Image
import os


class PacMan:
    ANIMATION_FRAME_DELAY = 6  # 60 FPS / 6 ~= 10 animation frames per second
    VISUAL_SCALE = 0.75

    def __init__(self, x, y, tile_size=40, speed=2):
        self.x = x
        self.y = y
        self.tile_size = tile_size
        self.size = tile_size
        self.speed = speed

        # ✅ FIX 1: Start with a valid movement direction
        self.direction = (1, 0)  # RIGHT
        self.next_direction = (0, 0)

        self.color = (255, 255, 0)
        self.score = 0
        self.pellets_eaten = 0

        self.render_size = max(8, int(self.size * self.VISUAL_SCALE))
        self.render_offset = (self.size - self.render_size) // 2

        self.pacman_images = {}
        self.animation_counter = 0
        self.last_facing_direction = "right"
        self._images_loaded = False
        self.score_popups = []
        self.score_popup_duration_frames = 45

    def _update_score_popups(self):
        if not self.score_popups:
            return

        updated = []
        for popup in self.score_popups:
            popup["ttl"] -= 1
            popup["y"] -= 0.6
            if popup["ttl"] > 0:
                updated.append(popup)
        self.score_popups = updated

    def _draw_score_popups(self, surface):
        if not self.score_popups:
            return

        font = pygame.font.Font(None, max(18, self.tile_size // 2))
        for popup in self.score_popups:
            alpha = max(0, int(255 * (popup["ttl"] / popup["max_ttl"])))
            text_surface = font.render(str(popup["points"]), True, (255, 255, 255))
            text_surface.set_alpha(alpha)
            rect = text_surface.get_rect(center=(int(popup["x"]), int(popup["y"])))
            surface.blit(text_surface, rect)

    def _load_pacman_images(self):
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
                        frame_scaled = pygame.transform.scale(
                            frame_data, (self.render_size, self.render_size)
                        )
                        frames.append(frame_scaled)
                        frame_index += 1
                    except EOFError:
                        break

                if frames:
                    self.pacman_images[direction] = frames
            except Exception:
                pass

        self._images_loaded = True

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

        half = self.tile_size // 2
        tolerance = max(1, min(self.speed, half))

        return (
            abs(tile_offset_x - half) <= tolerance
            and abs(tile_offset_y - half) <= tolerance
        )

    def _snap_to_axis(self):
        dx, dy = self.direction
        half = self.tile_size // 2

        if dx != 0:
            center_y = self.y + self.size // 2
            snapped_tile_y = (center_y // self.tile_size) * self.tile_size + half
            self.y = snapped_tile_y - self.size // 2

        elif dy != 0:
            center_x = self.x + self.size // 2
            snapped_tile_x = (center_x // self.tile_size) * self.tile_size + half
            self.x = snapped_tile_x - self.size // 2

    def set_direction(self, direction):
        # Keep simple — engine handles validity
        self.next_direction = direction

    def update(self, maze):
        self.animation_counter += 1
        self._update_score_popups()

        if self.next_direction != (0, 0):
            is_reverse = (
                self.next_direction[0] == -self.direction[0]
                and self.next_direction[1] == -self.direction[1]
            )

            if is_reverse:
                self.direction = self.next_direction
                self.next_direction = (0, 0)

            # ✅ FIX 2: Allow turning when stationary
            elif self.is_aligned_to_tile() or self.direction == (0, 0):
                self._snap_to_axis()

                next_x = self.x + self.next_direction[0] * self.speed
                next_y = self.y + self.next_direction[1] * self.speed

                if maze.can_move(next_x, next_y, self.size):
                    self.direction = self.next_direction
                    self.next_direction = (0, 0)

        dx, dy = self.direction

        if dx != 0:
            next_x = self.x + dx * self.speed
            next_y = self.y

            if maze.can_move(next_x, next_y, self.size):
                center_x = self.x + self.size // 2
                next_center_x = next_x + self.size // 2
                half = self.tile_size // 2

                cur_offset = center_x % self.tile_size
                nxt_offset = next_center_x % self.tile_size

                if (cur_offset < half <= nxt_offset) or (cur_offset > half >= nxt_offset):
                    snapped = (center_x // self.tile_size) * self.tile_size + half
                    if abs(next_center_x - snapped) <= self.speed:
                        next_x = snapped - self.size // 2

                self.x = next_x
            else:
                center_x = self.x + self.size // 2
                snapped = (center_x // self.tile_size) * self.tile_size + (self.tile_size // 2)
                self.x = snapped - self.size // 2

        elif dy != 0:
            next_x = self.x
            next_y = self.y + dy * self.speed

            if maze.can_move(next_x, next_y, self.size):
                center_y = self.y + self.size // 2
                next_center_y = next_y + self.size // 2
                half = self.tile_size // 2

                cur_offset = center_y % self.tile_size
                nxt_offset = next_center_y % self.tile_size

                if (cur_offset < half <= nxt_offset) or (cur_offset > half >= nxt_offset):
                    snapped = (center_y // self.tile_size) * self.tile_size + half
                    if abs(next_center_y - snapped) <= self.speed:
                        next_y = snapped - self.size // 2

                self.y = next_y
            else:
                center_y = self.y + self.size // 2
                snapped = (center_y // self.tile_size) * self.tile_size + (self.tile_size // 2)
                self.y = snapped - self.size // 2

        # ✅ FIX 3: Fallback movement (prevents deadlock)
        if self.direction == (0, 0):
            for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx = self.x + d[0] * self.speed
                ny = self.y + d[1] * self.speed
                if maze.can_move(nx, ny, self.size):
                    self.direction = d
                    break

        self.x, self.y = maze.handle_teleportation(self.x, self.y)

    def draw(self, surface):
        if not self._images_loaded:
            self._load_pacman_images()

        center_x = self.x + self.size // 2
        center_y = self.y + self.size // 2

        if self.pacman_images:
            direction = self._get_current_direction_name()
            if direction in self.pacman_images and self.pacman_images[direction]:
                frames = self.pacman_images[direction]
                frame_index = (self.animation_counter // self.ANIMATION_FRAME_DELAY) % len(frames)
                image = frames[frame_index]
                surface.blit(image, (self.x + self.render_offset, self.y + self.render_offset))
                self._draw_score_popups(surface)
                return

        pygame.draw.circle(
            surface,
            self.color,
            (int(center_x), int(center_y)),
            self.render_size // 3,
        )
        self._draw_score_popups(surface)

    def get_grid_position(self):
        # ✅ FIX 4: Use centre, not top-left
        center_x = self.x + self.size // 2
        center_y = self.y + self.size // 2
        return center_x // self.tile_size, center_y // self.tile_size

    def eat_pellet(self, points=10):
        self.score += points
        self.pellets_eaten += 1

        # Show arcade-style popup for bonuses (fruit, ghost, etc.).
        self.score_popups.append({
            "points": points,
            "x": self.x + self.size // 2,
            "y": self.y + self.size // 2,
            "ttl": self.score_popup_duration_frames,
            "max_ttl": self.score_popup_duration_frames,
        })