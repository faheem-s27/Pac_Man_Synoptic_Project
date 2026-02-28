import pygame

class Button:
    def __init__(self, x=300, y=375, width=200, height=50, text="Start Game", button_type="primary"):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.button_type = button_type

        # Pac-Man arcade style colors
        if button_type == "primary":
            self.color = (0, 0, 0)  # Black background
            self.hover_color = (255, 255, 0)  # Yellow on hover
            self.text_color = (255, 255, 0)  # Yellow text
            self.border_color = (255, 255, 0)  # Yellow border
        elif button_type == "secondary":
            self.color = (0, 0, 0)
            self.hover_color = (255, 100, 100)  # Red on hover
            self.text_color = (255, 100, 100)
            self.border_color = (255, 100, 100)
        else:  # "settings"
            self.color = (0, 0, 0)
            self.hover_color = (100, 200, 255)  # Cyan on hover
            self.text_color = (100, 200, 255)
            self.border_color = (100, 200, 255)

        self.border_width = 3
        self.font_size = 28
        self.font = None
        self.text = text
        self.is_hovered = False
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, mouse_pos):
        self.rect.x = self.x
        self.rect.y = self.y
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def draw(self, surface, font=None):
        if font:
            self.font = font

        current_color = self.hover_color if self.is_hovered else self.text_color
        border_color = self.hover_color if self.is_hovered else self.border_color

        # Draw outer border (glow effect)
        glow_rect = pygame.Rect(
            self.x - 2,
            self.y - 2,
            self.width + 4,
            self.height + 4
        )
        pygame.draw.rect(surface, border_color, glow_rect, 2)

        # Draw main border
        pygame.draw.rect(surface, border_color, self.rect, self.border_width)

        # Draw inner filled area
        inner_rect = pygame.Rect(
            self.x + self.border_width,
            self.y + self.border_width,
            self.width - 2 * self.border_width,
            self.height - 2 * self.border_width
        )
        pygame.draw.rect(surface, self.color, inner_rect)

        if self.font:
            text_surface = self.font.render(self.text, True, current_color)
            text_rect = text_surface.get_rect(center=self.rect.center)
            surface.blit(text_surface, text_rect)

    def is_clicked(self, mouse_pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(mouse_pos):
                return True
        return False