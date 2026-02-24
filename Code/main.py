from enum import Enum
import pygame

from Code.Button import Button
from Code.GameEngine import GameEngine


class GameState(Enum):
    MENU = 1
    GAME = 2
    GAME_OVER = 3
    QUIT = 4


def main():
    GAME_STATE = GameState.MENU
    print("Pac-Man Environment")

    # make a window that is 800x800 pixels
    pygame.init()

    screen = pygame.display.set_mode((800, 800))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Pac-Man Environment")
    run = True

    # Menu button
    startGamebutton = Button(300, 500, 200, 50, "Start Game")
    button_font = pygame.font.Font(None, startGamebutton.font_size)
    pygame.mouse.set_visible(True)

    # Game engine (initialized only when game starts)
    game_engine = None

    while run:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                print("User exited")

            # Handle button clicks in the menu state
            if GAME_STATE == GameState.MENU and startGamebutton.is_clicked(mouse_pos, event):
                GAME_STATE = GameState.GAME
                game_engine = GameEngine(800, 800)
                pygame.mouse.set_visible(False)
                print("Starting game...")

            # Handle game input
            if GAME_STATE == GameState.GAME and game_engine:
                game_engine.handle_input(event)

            # Return to menu from game over
            if GAME_STATE == GameState.GAME_OVER and startGamebutton.is_clicked(mouse_pos, event):
                GAME_STATE = GameState.MENU
                game_engine = None
                pygame.mouse.set_visible(True)
                print("Returning to menu...")

        # just black screen
        screen.fill((0, 0, 0))

        if GAME_STATE == GameState.MENU:
            # Update and draw button
            startGamebutton.text = "Start Game"
            startGamebutton.update(mouse_pos)
            startGamebutton.draw(screen, button_font)
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