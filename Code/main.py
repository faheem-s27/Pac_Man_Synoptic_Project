import sys
import pygame
from Code.GameEngine import GameState, GameEngine
from Code.Button import Button
from Code.Settings import Settings

FPS = 60


def parse_resolution(resolution_str):
    """Parse resolution string like '800x800' to (width, height)."""
    try:
        w, h = resolution_str.split('x')
        return int(w), int(h)
    except:
        return 800, 800


class PacManMenu:
    """Simple arcade-styled main menu. Settings are edited via game_settings.json."""

    def __init__(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height

        cx = window_width // 2

        self.btn_start = Button(cx - 100, window_height // 2 - 40, 200, 50, "START GAME", "primary")
        self.btn_quit  = Button(cx - 100, window_height // 2 + 30, 200, 50, "QUIT",       "secondary")

        self.font_title    = pygame.font.Font(None, 72)
        self.font_subtitle = pygame.font.Font(None, 32)
        self.font_button   = pygame.font.Font(None, 28)
        self.font_hint     = pygame.font.Font(None, 22)

    def update(self, mouse_pos, events):
        self.btn_start.update(mouse_pos)
        self.btn_quit.update(mouse_pos)

        for event in events:
            if self.btn_start.is_clicked(mouse_pos, event):
                return "start_game"
            if self.btn_quit.is_clicked(mouse_pos, event):
                return "quit"
        return None

    def draw(self, surface):
        surface.fill((0, 0, 0))
        w, h = self.window_width, self.window_height

        # Title
        title = self.font_title.render("PAC-MAN", True, (255, 255, 0))
        surface.blit(title, title.get_rect(center=(w // 2, h // 2 - 160)))

        # Subtitle
        sub = self.font_subtitle.render("SYNOPTIC PROJECT", True, (100, 200, 255))
        surface.blit(sub, sub.get_rect(center=(w // 2, h // 2 - 100)))

        # Buttons
        self.btn_start.draw(surface, self.font_button)
        self.btn_quit.draw(surface, self.font_button)

        # Hint
        hint = self.font_hint.render("Edit game_settings.json to change settings", True, (80, 80, 80))
        surface.blit(hint, hint.get_rect(center=(w // 2, h - 30)))


def main():
    print("Pac-Man Environment - Initialization")

    pygame.init()
    pygame.mixer.init()

    # Load settings from JSON — this is the only source of truth
    settings = Settings("game_settings.json")
    game_config = settings.get_all()

    resolution = parse_resolution(settings.get("window_resolution", "800x800"))
    window_width, window_height = resolution

    screen = pygame.display.set_mode(resolution)
    pygame.display.set_caption("Pac-Man Environment")
    clock = pygame.time.Clock()

    menu = PacManMenu(window_width, window_height)

    current_state = GameState.MENU
    game_engine = None

    menu_music_loaded = False
    try:
        pygame.mixer.music.load("../Audio/pacman_intermission.wav")
        menu_music_loaded = True
    except Exception as e:
        print(f"Audio Warning (Menu): {e}")

    run = True
    while run:
        mouse_pos = pygame.mouse.get_pos()
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                run = False
                continue

            if current_state == GameState.MENU:
                menu_action = menu.update(mouse_pos, events)
                if menu_action == "start_game":
                    pygame.mixer.music.stop()
                    # Re-read JSON fresh every time the game starts
                    settings = Settings("game_settings.json")
                    game_config = settings.get_all()
                    try:
                        pygame.mixer.music.load("../Audio/pacman_beginning.wav")
                        pygame.mixer.music.play()
                        current_state = GameState.AUDIO_PLAYING
                        game_engine = GameEngine(**game_config, paused=True)
                    except Exception as e:
                        print(f"Audio Warning: {e}. Skipping intro.")
                        current_state = GameState.GAME
                        game_engine = GameEngine(**game_config, paused=False)
                        pygame.mouse.set_visible(False)
                elif menu_action == "quit":
                    run = False

            elif current_state == GameState.GAME:
                if game_engine:
                    game_engine.handle_input(event)

            elif current_state == GameState.GAME_OVER:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    current_state = GameState.MENU
                    game_engine = None
                    pygame.mouse.set_visible(True)

        # --- State updates ---
        if current_state == GameState.MENU:
            if menu_music_loaded and not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(loops=-1)

        elif current_state == GameState.AUDIO_PLAYING:
            if not pygame.mixer.music.get_busy():
                current_state = GameState.GAME
                if game_engine:
                    game_engine.unpause()
                pygame.mouse.set_visible(False)

        elif current_state == GameState.GAME:
            if game_engine:
                game_engine.update()
                if game_engine.game_state == GameState.AUDIO_PLAYING:
                    current_state = GameState.AUDIO_PLAYING
                elif game_engine.game_over or game_engine.won:
                    current_state = GameState.GAME_OVER
                    pygame.mouse.set_visible(True)

        # --- Drawing ---
        screen.fill((0, 0, 0))

        if current_state == GameState.MENU:
            menu.draw(screen)

        elif current_state in (GameState.AUDIO_PLAYING, GameState.GAME, GameState.GAME_OVER):
            if game_engine:
                game_engine.draw(screen)

        if current_state == GameState.GAME_OVER:
            font_go   = pygame.font.Font(None, 64)
            font_inst = pygame.font.Font(None, 32)
            if game_engine and game_engine.won:
                go_text = font_go.render("YOU WIN!", True, (255, 255, 0))
            else:
                go_text = font_go.render("GAME OVER", True, (255, 100, 100))
            screen.blit(go_text, go_text.get_rect(center=(window_width // 2, window_height // 2 - 50)))
            inst = font_inst.render("Click to return to menu", True, (100, 200, 255))
            screen.blit(inst, inst.get_rect(center=(window_width // 2, window_height // 2 + 50)))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
