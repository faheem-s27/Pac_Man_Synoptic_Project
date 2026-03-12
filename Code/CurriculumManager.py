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

        # --- STAGE 1: Baby Steps (Gens 0-19) ---
        if generation_id < 20:
            print(f"--- CURRICULUM STAGE 1 (Gen {generation_id}): Eat 10 Pellets, No Ghosts, 1 Life ---")
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = 20
            settings['lives'] = 1
            settings['active_ghost_count'] = 0
            settings['enable_power_pellets'] = False  # ACTION: Remove the perimeter exploit
            settings['max_episode_steps'] = 500

        # --- STAGE 2: Spatial Mastery (Gens 20-74) ---
        elif generation_id < 75:
            print(f"--- CURRICULUM STAGE 2 (Gen {generation_id}): Clear Map, No Ghosts ---")
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = -1
            settings['active_ghost_count'] = 0
            settings['lives'] = 1
            settings['enable_power_pellets'] = False  # ACTION: Force internal maze navigation
            settings['max_episode_steps'] = 2000

        # --- STAGE 3: Introduction to Fear (Gens 75-119) ---
        elif generation_id < 120:
            print(f"--- CURRICULUM STAGE 3 (Gen {generation_id}): Clear Map, 1 Ghost ---")
            settings['enable_ghosts'] = True
            settings['pellets_to_win'] = -1
            settings['active_ghost_count'] = 1
            settings['lives'] = 1
            settings['enable_power_pellets'] = True  # ACTION: Restore for threat mitigation
            settings['max_episode_steps'] = 3000

        # --- STAGE 4: Rising Difficulty (Gens 120-159) ---
        elif generation_id < 160:
            print(f"--- CURRICULUM STAGE 4 (Gen {generation_id}): Clear Map, 2 Ghosts ---")
            settings['enable_ghosts'] = True
            settings['pellets_to_win'] = -1
            settings['active_ghost_count'] = 2
            settings['lives'] = 2
            settings['enable_power_pellets'] = True
            settings['max_episode_steps'] = 4000

        # --- STAGE 5: Full Game (Gens 160+) ---
        else:
            print(f"--- CURRICULUM STAGE 5 (Gen {generation_id}): Full Difficulty (4 Ghosts) ---")
            settings['enable_ghosts'] = True
            settings['pellets_to_win'] = -1
            settings['active_ghost_count'] = 4
            settings['lives'] = 3
            settings['enable_power_pellets'] = True
            settings['max_episode_steps'] = 5000

        return settings