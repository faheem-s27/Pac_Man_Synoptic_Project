from enum import Enum
import pygame

from Code.Button import Button
from Code.GameEngine import GameEngine

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

class GameState(Enum):
    MENU = 1
    GAME = 2
    GAME_OVER = 3
    QUIT = 4
    AUDIO_PLAYING = 5


def main():
    GAME_STATE = GameState.MENU
    print("Pac-Man Environment")

    # make a window that is 800x800 pixels
    pygame.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Pac-Man Environment")
    run = True

    # Menu button
    startGamebutton = Button(300, 500, 200, 50, "Start Game")
    button_font = pygame.font.Font(None, startGamebutton.font_size)
    pygame.mouse.set_visible(True)

    # Game engine (initialized only when game starts)
    game_engine = None

    # Audio setup
    pygame.mixer.init()

    # Load intermission music once at startup
    menu_music_loaded = False
    try:
        pygame.mixer.music.load("../Audio/pacman_intermission.wav")
        menu_music_loaded = True
    except Exception as e:
        print(f"Could not load intermission music: {e}")

    while run:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                print("User exited")

            # Handle button clicks in the menu state
            if GAME_STATE == GameState.MENU and startGamebutton.is_clicked(mouse_pos, event):
                # Stop menu music and load game audio
                pygame.mixer.music.stop()
                try:
                    pygame.mixer.music.load("../Audio/pacman_beginning.wav")
                    pygame.mixer.music.play()
                    GAME_STATE = GameState.AUDIO_PLAYING
                    game_engine = GameEngine(paused=True, screen_width=WINDOW_WIDTH, screen_height=WINDOW_HEIGHT)
                    print("Playing audio...")
                except Exception as e:
                    print(f"Could not load audio: {e}")
                    GAME_STATE = GameState.GAME
                    game_engine = GameEngine(screen_width=WINDOW_WIDTH, screen_height=WINDOW_HEIGHT)
                    pygame.mouse.set_visible(False)
                    print("Starting game...")

            # Handle game input
            if GAME_STATE == GameState.GAME and game_engine:
                game_engine.handle_input(event)

            # Return to menu from game over
            if GAME_STATE == GameState.GAME_OVER and startGamebutton.is_clicked(mouse_pos, event):
                pygame.mixer.music.stop()
                GAME_STATE = GameState.MENU
                game_engine = None
                pygame.mouse.set_visible(True)
                print("Returning to menu...")

        # just black screen
        screen.fill((0, 0, 0))

        if GAME_STATE == GameState.MENU:
            # Play menu music only once
            if menu_music_loaded and not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(loops=-1)  # -1 means loop infinitely

            # Update and draw button
            startGamebutton.text = "Start Game"
            startGamebutton.update(mouse_pos)
            startGamebutton.draw(screen, button_font)
        elif GAME_STATE == GameState.AUDIO_PLAYING:
            # Display loading/waiting message
            loading_font = pygame.font.Font(None, 48)
            loading_text = loading_font.render("Get Ready!", True, (255, 255, 0))
            text_rect = loading_text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
            screen.blit(loading_text, text_rect)

            if game_engine:
                game_engine.update()
                game_engine.draw(screen)

            # Check if audio has finished playing
            if not pygame.mixer.music.get_busy():
                GAME_STATE = GameState.GAME
                game_engine = GameEngine(use_classic_maze=False, maze_algorithm="prims", screen_width=WINDOW_WIDTH, screen_height=WINDOW_HEIGHT)
                pygame.mouse.set_visible(False)
                print("Starting game...")
        elif GAME_STATE == GameState.GAME:
            if game_engine:
                game_engine.update()
                game_engine.draw(screen)

                # Check if game ended
                if game_engine.game_over or game_engine.won:
                    GAME_STATE = GameState.GAME_OVER
        elif GAME_STATE == GameState.GAME_OVER:
            # Display game over screen with a button to return to menu
            if game_engine:
                game_engine.draw(screen)

            # Draw restart button
            pygame.mouse.set_visible(True)
            startGamebutton.update(mouse_pos)
            startGamebutton.text = "Back to Menu"
            startGamebutton.draw(screen, button_font)
        else:
            print("Invalid game state.")

        pygame.display.flip()

        # runs at 60fps for human perception
        clock.tick(60)

if __name__ == "__main__":
    main()