import os
from Code.Settings import Settings

_HERE = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_PATH = os.path.join(_HERE, "game_settings.json")

class CurriculumManager:
    def __init__(self, settings_path: str = None, stage_duration: int = 250):
        path = settings_path or _SETTINGS_PATH
        self.base_settings = Settings(path).get_all()
        # Increased stage duration to give more time for weights to settle per difficulty
        self.stage_duration = stage_duration
        self.current_stage = -1

    def get_settings_for_generation(self, step_id: int):
        settings = self.base_settings.copy()
        stage = step_id // self.stage_duration

        if stage != self.current_stage:
            print(f"\n--- TOUGHENED CURRICULUM: Stage {stage} (Episode/Gen {step_id}) ---")
            self.current_stage = stage

        # GLOBAL HARDCORE RULES: 1 Life only for training (forces perfection)
        settings['lives'] = 1
        settings['enable_power_pellets'] = False

        # --- PHASE 1: RAPID FORAGING (No Ghosts) ---
        if stage == 0:
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = 25
            settings['max_episode_steps'] = 1500 # High ceiling to allow exploration

        elif stage == 1:
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = 75
            settings['max_episode_steps'] = 2000

        elif stage == 2:
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = -1 # FULL CLEAR MANDATORY
            settings['max_episode_steps'] = 3000

        # --- PHASE 2: INTRODUCING LETHALITY (The Survival Test) ---
        elif stage == 3:
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 1
            settings['pellets_to_win'] = -1
            settings['max_episode_steps'] = 3500

        elif stage == 4:
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 2
            settings['pellets_to_win'] = -1
            settings['max_episode_steps'] = 4000

        elif stage == 5:
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 3
            settings['pellets_to_win'] = -1
            settings['max_episode_steps'] = 4500

        # --- PHASE 3: FULL CONTEXT (The Hunt) ---
        elif stage == 6:
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 4
            settings['pellets_to_win'] = -1
            settings['max_episode_steps'] = 5000
            settings['enable_power_pellets'] = True # Learn to use power pellets under max pressure

        elif stage >= 7:
            # THE GAUNTLET: 3 Lives restored but ghosts are at max aggression
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 4
            settings['lives'] = 3
            settings['pellets_to_win'] = -1
            settings['max_episode_steps'] = 6000 # Maximum time for full strategic clears
            settings['enable_power_pellets'] = True

        return settings