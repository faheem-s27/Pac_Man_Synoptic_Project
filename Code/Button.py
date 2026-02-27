import pygame

class Button:
    def __init__(self, x=300, y=375, width=200, height=50, text="Start Game"):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.color = (33, 33, 222)
        self.hover_color = (255, 255, 0)
        self.text_color = (0, 0, 0)
        self.border_color = (33, 33, 222)
        self.border_width = 4
        self.font_size = 32
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

        current_color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, self.border_color, self.rect, self.border_width)

        inner_rect = pygame.Rect(
            self.x + self.border_width,
            self.y + self.border_width,
            self.width - 2 * self.border_width,
            self.height - 2 * self.border_width
        )
        pygame.draw.rect(surface, current_color, inner_rect)

        if self.font:
            text_surface = self.font.render(self.text, True, self.text_color)
            text_rect = text_surface.get_rect(center=self.rect.center)
            surface.blit(text_surface, text_rect)

    def is_clicked(self, mouse_pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(mouse_pos):
                return True
        return False