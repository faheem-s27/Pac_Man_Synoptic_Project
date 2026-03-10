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

    def get_settings_for_generation(self, generation_id):
        """
        Returns a modified settings dictionary based on the current generation.
        """
        settings = self.base_settings.copy()

        # --- STAGE 1: Baby Steps (Gens 0-9) ---
        # Goal: Learn to move and eat. No danger.
        if generation_id < 10:
            print(f"--- CURRICULUM STAGE 1 (Gen {generation_id}): Eat 10 Pellets, No Ghosts, 1 Life ---")
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = 10
            settings['lives'] = 1  # Fail fast if they get stuck
            settings['active_ghost_count'] = 0

        # --- STAGE 2: Endurance (Gens 10-24) ---
        # Goal: Learn to clear the whole map. No danger.
        elif generation_id < 25:
            print(f"--- CURRICULUM STAGE 2 (Gen {generation_id}): Clear Map, No Ghosts ---")
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = -1  # -1 means "eat all"
            settings['active_ghost_count'] = 0
            settings['lives'] = 1

        # --- STAGE 3: Introduction to Fear (Gens 25-39) ---
        # Goal: Eat all pellets while avoiding ONE ghost (usually Blinky).
        elif generation_id < 40:
            print(f"--- CURRICULUM STAGE 3 (Gen {generation_id}): Clear Map, 1 Ghost ---")
            settings['enable_ghosts'] = True
            settings['pellets_to_win'] = -1
            settings['active_ghost_count'] = 1
            settings['lives'] = 1

        # --- STAGE 4: Rising Difficulty (Gens 40-54) ---
        # Goal: Handle 2 ghosts.
        elif generation_id < 55:
            print(f"--- CURRICULUM STAGE 4 (Gen {generation_id}): Clear Map, 2 Ghosts ---")
            settings['enable_ghosts'] = True
            settings['pellets_to_win'] = -1
            settings['active_ghost_count'] = 2
            settings['lives'] = 2

        # --- STAGE 5: Full Game (Gens 55+) ---
        # Goal: Standard Pacman gameplay with all 4 ghosts.
        else:
            print(f"--- CURRICULUM STAGE 5 (Gen {generation_id}): Full Difficulty (4 Ghosts) ---")
            settings['enable_ghosts'] = True
            settings['pellets_to_win'] = -1
            settings['active_ghost_count'] = 4
            settings['lives'] = 3

        return settings

