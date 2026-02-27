import sys
import pygame
from Code.GameEngine import GameState, GameEngine
from Code.Button import Button

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 900
FPS = 60

def main():
    print("Pac-Man Environment - Initialization")

    pygame.init()
    pygame.mixer.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Pac-Man Environment")
    clock = pygame.time.Clock()

    current_state = GameState.MENU
    game_engine = None

    btn_start = Button(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2, 200, 50, "Start Game")
    font_button = pygame.font.Font(None, btn_start.font_size)
    font_loading = pygame.font.Font(None, 48)

    menu_music_loaded = False
    try:
        pygame.mixer.music.load("../Audio/pacman_intermission.wav")
        menu_music_loaded = True
    except Exception as e:
        print(f"Audio Warning (Menu): {e}")

    ENGINE_CONFIG = {
        "screen_width": WINDOW_WIDTH,
        "screen_height": WINDOW_HEIGHT,
        "use_classic_maze": False,
        "maze_algorithm": "prims"
    }

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
                    if btn_start.is_clicked(mouse_pos, event):
                        pygame.mixer.music.stop()
                        try:
                            pygame.mixer.music.load("../Audio/pacman_beginning.wav")
                            pygame.mixer.music.play()
                            current_state = GameState.AUDIO_PLAYING
                            game_engine = GameEngine(**ENGINE_CONFIG, paused=True)
                        except Exception as e:
                            print(f"Audio Warning: {e}. Skipping intro.")
                            current_state = GameState.GAME
                            game_engine = GameEngine(**ENGINE_CONFIG, paused=False)
                            pygame.mouse.set_visible(False)

                case GameState.GAME:
                    if game_engine:
                        game_engine.handle_input(event)

                case GameState.GAME_OVER:
                    if btn_start.is_clicked(mouse_pos, event):
                        pygame.mixer.music.stop()
                        current_state = GameState.MENU
                        game_engine = None
                        pygame.mouse.set_visible(True)

        match current_state:
            case GameState.MENU:
                if menu_music_loaded and not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(loops=-1)
                btn_start.text = "Start Game"
                btn_start.update(mouse_pos)

            case GameState.AUDIO_PLAYING:
                if not pygame.mixer.music.get_busy():
                    current_state = GameState.GAME
                    if game_engine:
                        game_engine.unpause()
                    pygame.mouse.set_visible(False)

            case GameState.GAME:
                if game_engine:
                    game_engine.update()
                    if game_engine.game_over or game_engine.won:
                        current_state = GameState.GAME_OVER
                        pygame.mouse.set_visible(True)

            case GameState.GAME_OVER:
                btn_start.text = "Back to Menu"
                btn_start.update(mouse_pos)

        screen.fill((0, 0, 0))

        if current_state in (GameState.AUDIO_PLAYING, GameState.GAME, GameState.GAME_OVER) and game_engine:
            game_engine.draw(screen)

        match current_state:
            case GameState.MENU | GameState.GAME_OVER:
                btn_start.draw(screen, font_button)
            case GameState.AUDIO_PLAYING:
                loading_text = font_loading.render("GET READY!", True, (255, 255, 0))
                text_rect = loading_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
                screen.blit(loading_text, text_rect)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()