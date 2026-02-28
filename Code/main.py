import sys
import pygame
from Code.GameEngine import GameState, GameEngine
from Code.Button import Button
from Code.Settings import Settings
from Code.UIElements import Slider, Toggle, Dropdown, UILabel

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
FPS = 60

class MenuState:
    """Enum for menu screens."""
    MAIN_MENU = 1
    SETTINGS_MENU = 2
    GAME = 3
    GAME_OVER = 4
    AUDIO_PLAYING = 5

class PacManMenu:
    """Enhanced menu system with settings."""

    def __init__(self, settings):
        self.settings = settings
        self.state = MenuState.MAIN_MENU

        # Main menu buttons
        self.btn_start = Button(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 - 80, 200, 50, "START GAME", "primary")
        self.btn_settings = Button(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2, 200, 50, "SETTINGS", "settings")
        self.btn_quit = Button(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 + 80, 200, 50, "QUIT", "secondary")

        # Settings menu buttons
        self.btn_save = Button(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT - 100, 200, 50, "SAVE & BACK", "primary")
        self.btn_reset = Button(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT - 40, 200, 50, "RESET", "secondary")

        # Settings UI elements
        self._initialize_settings_ui()

        # Fonts
        self.font_title = pygame.font.Font(None, 64)
        self.font_subtitle = pygame.font.Font(None, 32)
        self.font_button = pygame.font.Font(None, 28)
        self.font_label = pygame.font.Font(None, 20)

    def _initialize_settings_ui(self):
        """Initialize all settings UI elements."""
        start_y = 120
        spacing = 60

        # Tile Size slider
        self.slider_tile_size = Slider(
            200, start_y, 300, 10, 50,
            self.settings.get("tile_size"),
            "Tile Size", 5
        )

        # Pacman Speed slider
        self.slider_pacman_speed = Slider(
            200, start_y + spacing, 300, 1, 5,
            int(self.settings.get("pacman_speed")),
            "Pacman Speed", 1
        )

        # Ghost Speed slider
        self.slider_ghost_speed = Slider(
            200, start_y + spacing * 2, 300, 1, 3,
            int(self.settings.get("ghost_speed") * 10) / 10,
            "Ghost Speed", 0.1
        )

        # Classic Maze toggle
        self.toggle_classic = Toggle(
            200, start_y + spacing * 3,
            "Classic Maze",
            self.settings.get("use_classic_maze")
        )

        # Enable Ghosts toggle
        self.toggle_ghosts = Toggle(
            200, start_y + spacing * 4,
            "Enable Ghosts",
            self.settings.get("enable_ghosts")
        )

        # God Mode toggle
        self.toggle_god = Toggle(
            200, start_y + spacing * 5,
            "God Mode",
            self.settings.get("god_mode")
        )

        # Lives slider
        self.slider_lives = Slider(
            200, start_y + spacing * 6, 300, 1, 9,
            self.settings.get("lives"),
            "Lives", 1
        )

        # Maze Algorithm dropdown
        self.dropdown_algorithm = Dropdown(
            200, start_y + spacing * 7,
            ["recursive_backtracking", "prims"],
            self.settings.get("maze_algorithm"),
            "Algorithm"
        )

        # Window Resolution dropdown
        self.dropdown_resolution = Dropdown(
            200, start_y + spacing * 8,
            ["1000x800", "1200x1000", "1920x1080"],
            self.settings.get("window_resolution"),
            "Resolution"
        )

    def update(self, mouse_pos, events):
        """Update menu state based on input."""
        mouse_pressed = pygame.mouse.get_pressed()[0]

        if self.state == MenuState.MAIN_MENU:
            self.btn_start.update(mouse_pos)
            self.btn_settings.update(mouse_pos)
            self.btn_quit.update(mouse_pos)

            for event in events:
                if self.btn_start.is_clicked(mouse_pos, event):
                    return "start_game"
                if self.btn_settings.is_clicked(mouse_pos, event):
                    self.state = MenuState.SETTINGS_MENU
                if self.btn_quit.is_clicked(mouse_pos, event):
                    return "quit"

        elif self.state == MenuState.SETTINGS_MENU:
            # Update all sliders
            self.slider_tile_size.update(mouse_pos, mouse_pressed)
            self.slider_pacman_speed.update(mouse_pos, mouse_pressed)
            self.slider_ghost_speed.update(mouse_pos, mouse_pressed)
            self.slider_lives.update(mouse_pos, mouse_pressed)

            # Update toggles and dropdowns
            for event in events:
                self.toggle_classic.update(mouse_pos, event)
                self.toggle_ghosts.update(mouse_pos, event)
                self.toggle_god.update(mouse_pos, event)
                self.dropdown_algorithm.update(mouse_pos, event)
                self.dropdown_resolution.update(mouse_pos, event)

            self.btn_save.update(mouse_pos)
            self.btn_reset.update(mouse_pos)

            for event in events:
                if self.btn_save.is_clicked(mouse_pos, event):
                    self._save_settings()
                    self.state = MenuState.MAIN_MENU
                if self.btn_reset.is_clicked(mouse_pos, event):
                    self.settings.reset_to_defaults()
                    self._initialize_settings_ui()

    def _save_settings(self):
        """Save current UI values to settings."""
        self.settings.set("tile_size", int(self.slider_tile_size.current_val))
        self.settings.set("pacman_speed", int(self.slider_pacman_speed.current_val))
        self.settings.set("ghost_speed", round(self.slider_ghost_speed.current_val, 1))
        self.settings.set("use_classic_maze", self.toggle_classic.state)
        self.settings.set("enable_ghosts", self.toggle_ghosts.state)
        self.settings.set("god_mode", self.toggle_god.state)
        self.settings.set("lives", int(self.slider_lives.current_val))
        self.settings.set("maze_algorithm", self.dropdown_algorithm.current_option)
        self.settings.set("window_resolution", self.dropdown_resolution.current_option)

    def draw(self, surface):
        """Draw the current menu screen."""
        surface.fill((0, 0, 0))

        if self.state == MenuState.MAIN_MENU:
            self._draw_main_menu(surface)
        elif self.state == MenuState.SETTINGS_MENU:
            self._draw_settings_menu(surface)

    def _draw_main_menu(self, surface):
        """Draw main menu with arcade styling."""
        # Draw title
        title_text = self.font_title.render("PAC-MAN", True, (255, 255, 0))
        title_rect = title_text.get_rect(center=(WINDOW_WIDTH // 2, 80))
        surface.blit(title_text, title_rect)

        # Draw subtitle
        subtitle_text = self.font_subtitle.render("SYNOPTIC PROJECT", True, (100, 200, 255))
        subtitle_rect = subtitle_text.get_rect(center=(WINDOW_WIDTH // 2, 150))
        surface.blit(subtitle_text, subtitle_rect)

        # Draw buttons
        self.btn_start.draw(surface, self.font_button)
        self.btn_settings.draw(surface, self.font_button)
        self.btn_quit.draw(surface, self.font_button)

        # Draw some arcade flavor text
        arcade_text = self.font_label.render("Use your keyboard to move and navigate", True, (100, 100, 100))
        arcade_rect = arcade_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 50))
        surface.blit(arcade_text, arcade_rect)

    def _draw_settings_menu(self, surface):
        """Draw settings menu."""
        # Draw title
        title_text = self.font_title.render("SETTINGS", True, (100, 200, 255))
        title_rect = title_text.get_rect(center=(WINDOW_WIDTH // 2, 30))
        surface.blit(title_text, title_rect)

        # Draw all sliders
        self.slider_tile_size.draw(surface)
        self.slider_pacman_speed.draw(surface)
        self.slider_ghost_speed.draw(surface)
        self.slider_lives.draw(surface)

        # Draw toggles
        self.toggle_classic.draw(surface)
        self.toggle_ghosts.draw(surface)
        self.toggle_god.draw(surface)

        # Draw dropdowns
        self.dropdown_algorithm.draw(surface)
        self.dropdown_resolution.draw(surface)

        # Draw buttons
        self.btn_save.draw(surface, self.font_button)
        self.btn_reset.draw(surface, self.font_button)


def parse_resolution(resolution_str):
    """Parse resolution string like '800x800' to (width, height)."""
    try:
        w, h = resolution_str.split('x')
        return (int(w), int(h))
    except:
        return (800, 800)


def main():
    print("Pac-Man Environment - Initialization")

    pygame.init()
    pygame.mixer.init()

    # Load settings
    settings = Settings("game_settings.json")

    # Parse initial resolution
    resolution = parse_resolution(settings.get("window_resolution"))

    # Create screen with initial resolution
    screen = pygame.display.set_mode(resolution)
    pygame.display.set_caption("Pac-Man Environment")
    clock = pygame.time.Clock()

    menu = PacManMenu(settings)

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

            match current_state:
                case GameState.MENU:
                    menu_action = menu.update(mouse_pos, events)
                    if menu_action == "start_game":
                        pygame.mixer.music.stop()
                        try:
                            pygame.mixer.music.load("../Audio/pacman_beginning.wav")
                            pygame.mixer.music.play()
                            current_state = GameState.AUDIO_PLAYING
                            # Create game engine with settings
                            game_config = settings.get_all()
                            game_engine = GameEngine(**game_config, paused=True)
                        except Exception as e:
                            print(f"Audio Warning: {e}. Skipping intro.")
                            current_state = GameState.GAME
                            game_config = settings.get_all()
                            game_engine = GameEngine(**game_config, paused=False)
                            pygame.mouse.set_visible(False)
                    elif menu_action == "quit":
                        run = False

                case GameState.GAME:
                    if game_engine:
                        game_engine.handle_input(event)

                case GameState.GAME_OVER:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        current_state = GameState.MENU
                        game_engine = None
                        pygame.mouse.set_visible(True)
                        menu.state = MenuState.MAIN_MENU

        # Check for resolution changes
        current_resolution = parse_resolution(settings.get("window_resolution"))
        current_screen_size = screen.get_size()
        current_screen_flags = screen.get_flags()

        if current_resolution != current_screen_size:
            screen = pygame.display.set_mode(current_resolution)
            pygame.display.set_caption("Pac-Man Environment")

        match current_state:
            case GameState.MENU:
                if menu_music_loaded and not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(loops=-1)
                menu.draw(screen)

            case GameState.AUDIO_PLAYING:
                if not pygame.mixer.music.get_busy():
                    current_state = GameState.GAME
                    if game_engine:
                        game_engine.unpause()
                    pygame.mouse.set_visible(False)
                menu.draw(screen)

            case GameState.GAME:
                if game_engine:
                    game_engine.update()
                    if game_engine.game_over or game_engine.won:
                        current_state = GameState.GAME_OVER
                        pygame.mouse.set_visible(True)

            case GameState.GAME_OVER:
                # Draw game over screen
                pass

        screen.fill((0, 0, 0))

        if current_state in (GameState.AUDIO_PLAYING, GameState.GAME, GameState.GAME_OVER) and game_engine:
            game_engine.draw(screen)
        elif current_state == GameState.MENU:
            menu.draw(screen)

        match current_state:
            case GameState.GAME_OVER:
                # Draw game over UI
                font_gameover = pygame.font.Font(None, 64)
                font_instruction = pygame.font.Font(None, 32)

                if game_engine.won:
                    gameover_text = font_gameover.render("YOU WIN!", True, (255, 255, 0))
                else:
                    gameover_text = font_gameover.render("GAME OVER", True, (255, 100, 100))

                gameover_rect = gameover_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
                screen.blit(gameover_text, gameover_rect)

                instruction_text = font_instruction.render("Click to return to menu", True, (100, 200, 255))
                instruction_rect = instruction_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
                screen.blit(instruction_text, instruction_rect)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

