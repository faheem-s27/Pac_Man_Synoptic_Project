from enum import Enum

import pygame
from pip._internal.utils.misc import enum

class GameState(Enum):
    MENU = 1
    GAME = 2
    GAME_OVER = 3
    QUIT = 4

GAME_STATE = GameState.MENU

def main():
    print("Pac-Man Environment")

    # make a window that is 800x800 pixels
    pygame.init()

    screen = pygame.display.set_mode((800, 800))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Pac-Man Environment")
    pygame.mouse.set_visible(False)
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                print("User exited")

        # just black screen
        screen.fill((0, 0, 0))

        if GAME_STATE == GameState.MENU:
            print("Menu State")
        elif GAME_STATE == GameState.GAME:
            print("Game State")
        elif GAME_STATE == GameState.GAME_OVER:
            print("Game Over State")
        else:
            print("Invalid game state.")

        pygame.display.flip()

        # runs at 60fps for human perception
        clock.tick(60)

if __name__ == "__main__":
    main()