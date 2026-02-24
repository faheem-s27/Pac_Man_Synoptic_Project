import pygame

class Button:
    def __init__(self, x=300, y=375, width=200, height=50, text="Start Game"):
        self.width = width
        self.height = height
        self.x = x
        self.y = y

        # Pac-Man arcade style colors
        self.color = (33, 33, 222)  # Arcade pink/magenta (classic Pac-Man color)
        self.hover_color = (255, 255, 0)  # Bright yellow (Pac-Man's color)
        self.text_color = (0, 0, 0)  # Black text for contrast
        self.border_color = (33, 33, 222)  # Deep blue (arcade style)
        self.border_width = 4

        self.font_size = 32
        self.font = None
        self.text = text

        self.is_hovered = False
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, mouse_pos):
        """Update button state based on mouse position"""
        self.rect.x = self.x
        self.rect.y = self.y
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def draw(self, surface, font=None):
        """Draw the button with Pac-Man arcade styling"""
        if font:
            self.font = font

        # Use hover color if hovered, otherwise use base color
        current_color = self.hover_color if self.is_hovered else self.color

        # Draw border (for arcade effect)
        pygame.draw.rect(surface, self.border_color, self.rect, self.border_width)

        # Draw button background with a slight rounded effect by drawing a smaller rect
        inner_rect = pygame.Rect(
            self.x + self.border_width,
            self.y + self.border_width,
            self.width - 2 * self.border_width,
            self.height - 2 * self.border_width
        )
        pygame.draw.rect(surface, current_color, inner_rect)

        # Draw text if font is available
        if self.font:
            text_surface = self.font.render(self.text, True, self.text_color)
            text_rect = text_surface.get_rect(center=self.rect.center)
            surface.blit(text_surface, text_rect)

    def is_clicked(self, mouse_pos, event):
        """Check if button is clicked"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(mouse_pos):
                return True
        return False

