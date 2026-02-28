import pygame

class UILabel:
    """Simple text label for the UI."""

    def __init__(self, x, y, text, font_size=24, color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.text = text
        self.font_size = font_size
        self.color = color
        self.font = pygame.font.Font(None, font_size)

    def draw(self, surface):
        text_surface = self.font.render(self.text, True, self.color)
        surface.blit(text_surface, (self.x, self.y))

class Slider:
    """Horizontal slider for numeric settings."""

    def __init__(self, x, y, width, min_val, max_val, current_val, label="", step=1):
        self.x = x
        self.y = y
        self.width = width
        self.height = 20
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = current_val
        self.label = label
        self.step = step
        self.rect = pygame.Rect(x, y, width, self.height)
        self.dragging = False
        self.font = pygame.font.Font(None, 20)
        self.value_changed = False

    def update(self, mouse_pos, mouse_pressed):
        """Update slider state based on mouse input."""
        self.value_changed = False

        if mouse_pressed and self.rect.collidepoint(mouse_pos):
            self.dragging = True

        if not mouse_pressed:
            self.dragging = False

        if self.dragging:
            # Calculate position within slider
            relative_x = max(0, min(mouse_pos[0] - self.x, self.width))
            # Map to value range
            ratio = relative_x / self.width
            new_val = self.min_val + (self.max_val - self.min_val) * ratio
            # Apply step
            new_val = round(new_val / self.step) * self.step
            new_val = max(self.min_val, min(self.max_val, new_val))

            if new_val != self.current_val:
                self.current_val = new_val
                self.value_changed = True

    def draw(self, surface):
        # Draw background track
        pygame.draw.rect(surface, (50, 50, 50), self.rect)

        # Draw handle
        handle_ratio = (self.current_val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.x + (self.width * handle_ratio)
        handle_rect = pygame.Rect(handle_x - 5, self.y - 3, 10, self.height + 6)
        pygame.draw.rect(surface, (255, 255, 0), handle_rect)
        pygame.draw.rect(surface, (255, 200, 0), handle_rect, 2)

        # Draw label and value
        if self.label:
            label_text = f"{self.label}: {self.current_val}"
            text_surface = self.font.render(label_text, True, (255, 255, 255))
            surface.blit(text_surface, (self.x, self.y - 25))

class Toggle:
    """Toggle button for boolean settings."""

    def __init__(self, x, y, label="", initial_state=False):
        self.x = x
        self.y = y
        self.label = label
        self.state = initial_state
        self.width = 40
        self.height = 20
        self.rect = pygame.Rect(x, y, self.width, self.height)
        self.font = pygame.font.Font(None, 20)
        self.value_changed = False

    def update(self, mouse_pos, event):
        """Update toggle state based on mouse click."""
        self.value_changed = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(mouse_pos):
                self.state = not self.state
                self.value_changed = True

    def draw(self, surface):
        # Draw background
        bg_color = (100, 200, 100) if self.state else (150, 50, 50)
        pygame.draw.rect(surface, bg_color, self.rect)
        pygame.draw.rect(surface, (255, 255, 255), self.rect, 2)

        # Draw toggle indicator
        indicator_x = self.x + (self.width - 15) if self.state else self.x + 2
        indicator_rect = pygame.Rect(indicator_x, self.y + 2, 15, 16)
        pygame.draw.rect(surface, (255, 255, 255), indicator_rect)

        # Draw label
        label_text = f"{self.label}: {'ON' if self.state else 'OFF'}"
        text_surface = self.font.render(label_text, True, (255, 255, 255))
        surface.blit(text_surface, (self.x + self.width + 10, self.y - 2))

class Dropdown:
    """Dropdown selection for options."""

    def __init__(self, x, y, options, current_option, label=""):
        self.x = x
        self.y = y
        self.options = options
        self.current_option = current_option
        self.label = label
        self.width = 150
        self.height = 25
        self.rect = pygame.Rect(x, y, self.width, self.height)
        self.is_open = False
        self.font = pygame.font.Font(None, 18)
        self.value_changed = False
        self.option_rects = []

    def update(self, mouse_pos, event):
        """Update dropdown state based on mouse input."""
        self.value_changed = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(mouse_pos):
                self.is_open = not self.is_open
            elif self.is_open:
                # Check if clicked on an option
                for i, opt_rect in enumerate(self.option_rects):
                    if opt_rect.collidepoint(mouse_pos):
                        self.current_option = self.options[i]
                        self.is_open = False
                        self.value_changed = True
                        break
                if not any(rect.collidepoint(mouse_pos) for rect in self.option_rects):
                    self.is_open = False

    def draw(self, surface):
        # Draw main button
        bg_color = (100, 100, 150) if self.is_open else (80, 80, 120)
        pygame.draw.rect(surface, bg_color, self.rect)
        pygame.draw.rect(surface, (255, 255, 255), self.rect, 2)

        # Draw current option text
        text_surface = self.font.render(self.current_option, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

        # Draw label
        if self.label:
            label_surface = self.font.render(self.label + ":", True, (255, 255, 255))
            surface.blit(label_surface, (self.x - 150, self.y))

        # Draw dropdown options if open
        self.option_rects = []
        if self.is_open:
            for i, option in enumerate(self.options):
                opt_y = self.y + self.height + (i * self.height)
                opt_rect = pygame.Rect(self.x, opt_y, self.width, self.height)
                self.option_rects.append(opt_rect)

                opt_color = (120, 120, 180) if option == self.current_option else (80, 80, 120)
                pygame.draw.rect(surface, opt_color, opt_rect)
                pygame.draw.rect(surface, (200, 200, 200), opt_rect, 1)

                opt_text = self.font.render(option, True, (255, 255, 255))
                opt_text_rect = opt_text.get_rect(center=opt_rect.center)
                surface.blit(opt_text, opt_text_rect)

