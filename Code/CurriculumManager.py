import os
from Code.Settings import Settings

# Always resolve game_settings.json relative to this file (Code/ folder)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_PATH = os.path.join(_HERE, "game_settings.json")

class CurriculumManager:
    def __init__(self, settings_path: str = None):
        # Use the provided path, or fall back to the sibling game_settings.json
        path = settings_path or _SETTINGS_PATH
        # Delegate to the existing Settings class — it handles missing files gracefully
        self.base_settings = Settings(path).get_all()

    def get_settings_for_generation(self, generation_id: int):
        """Return curriculum settings for the given generation.

        Design:
        - Each curriculum step lasts 200 generations.
        - First phase: increase pellets required with no ghosts.
          e.g. 10, 20, 30, 40, 50 pellets (each for 200 gens).
        - Then require full-clear (pellets_to_win = -1) with 0 ghosts.
        - Then gradually add ghosts: 1, 2, 3, 4 (full game).
        - After all stages, remain at full-game config.
        """

        settings = self.base_settings.copy()

        stage = generation_id // 200  # 0-based stage index, 200 gens per stage

        # Base common tweaks
        settings['lives'] = 1
        # By default, disable power pellets; selectively enable from stage 5 onwards.
        settings['enable_power_pellets'] = False

        if stage == 0:
            print(f"--- CURRICULUM STAGE 0 (Gen {generation_id}): 10 Pellets, No Ghosts ---")
            settings['enable_ghosts'] = False
            settings['active_ghost_count'] = 0
            settings['pellets_to_win'] = 10
            settings['max_episode_steps'] = 500

        elif stage == 1:
            print(f"--- CURRICULUM STAGE 1 (Gen {generation_id}): 20 Pellets, No Ghosts ---")
            settings['enable_ghosts'] = False
            settings['active_ghost_count'] = 0
            settings['pellets_to_win'] = 20
            settings['max_episode_steps'] = 800

        elif stage == 2:
            print(f"--- CURRICULUM STAGE 2 (Gen {generation_id}): 30 Pellets, No Ghosts ---")
            settings['enable_ghosts'] = False
            settings['active_ghost_count'] = 0
            settings['pellets_to_win'] = 30
            settings['max_episode_steps'] = 1200

        elif stage == 3:
            print(f"--- CURRICULUM STAGE 3 (Gen {generation_id}): 40 Pellets, No Ghosts ---")
            settings['enable_ghosts'] = False
            settings['active_ghost_count'] = 0
            settings['pellets_to_win'] = 40
            settings['max_episode_steps'] = 1600

        elif stage == 4:
            print(f"--- CURRICULUM STAGE 4 (Gen {generation_id}): 50 Pellets, No Ghosts ---")
            settings['enable_ghosts'] = False
            settings['active_ghost_count'] = 0
            settings['pellets_to_win'] = 80
            settings['max_episode_steps'] = 2000

        elif stage == 5:
            print(f"--- CURRICULUM STAGE 5 (Gen {generation_id}): Full Clear, No Ghosts ---")
            settings['enable_ghosts'] = False
            settings['active_ghost_count'] = 0
            settings['pellets_to_win'] = -1  # must clear the map
            settings['max_episode_steps'] = 2500
            settings['enable_power_pellets'] = True

        elif stage == 6:
            print(f"--- CURRICULUM STAGE 6 (Gen {generation_id}): Full Clear, 1 Ghost ---")
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 1
            settings['pellets_to_win'] = -1
            settings['max_episode_steps'] = 3000
            settings['enable_power_pellets'] = True

        elif stage == 7:
            print(f"--- CURRICULUM STAGE 7 (Gen {generation_id}): Full Clear, 2 Ghosts ---")
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 2
            settings['pellets_to_win'] = -1
            settings['max_episode_steps'] = 3500
            settings['enable_power_pellets'] = True

        elif stage == 8:
            print(f"--- CURRICULUM STAGE 8 (Gen {generation_id}): Full Clear, 3 Ghosts ---")
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 3
            settings['pellets_to_win'] = -1
            settings['max_episode_steps'] = 4000
            settings['enable_power_pellets'] = True

        else:
            # stage >= 9 → Full game forever (4 ghosts, full clear)
            print(f"--- CURRICULUM STAGE 9+ (Gen {generation_id}): Full Game, 4 Ghosts ---")
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 4
            settings['lives'] = 3
            settings['pellets_to_win'] = -1
            settings['max_episode_steps'] = 5000
            settings['enable_power_pellets'] = True

        return settings