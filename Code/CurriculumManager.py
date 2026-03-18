import os
import collections
import copy
from Code.Settings import Settings

_HERE = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_PATH = os.path.join(_HERE, "game_settings.json")


class CurriculumManager:
    def __init__(self, settings_path: str = None):
        path = settings_path or _SETTINGS_PATH
        self.base_settings = Settings(path).get_all()

        # Rolling performance window
        self.recent_results = collections.deque(maxlen=50)

        self.current_stage = 0

    def update_performance(self, won: bool):
        self.recent_results.append(bool(won))

    # ---------------- PROMOTION ----------------
    def check_promotion(self) -> bool:
        if len(self.recent_results) < self.recent_results.maxlen:
            return False

        win_rate = sum(self.recent_results) / len(self.recent_results)

        # Stability check: last 10 must also be strong
        recent_10 = list(self.recent_results)[-10:]
        recent_10_rate = sum(recent_10) / 10

        if win_rate >= 0.80 and recent_10_rate >= 0.80:
            self.current_stage += 1
            print(f"\n--- CURRICULUM PROMOTION: Stage {self.current_stage} (win_rate={win_rate:.2f}) ---")
            self.recent_results.clear()
            return True

        return False

    # ---------------- DEMOTION ----------------
    def check_demotion(self) -> bool:
        if len(self.recent_results) < self.recent_results.maxlen:
            return False

        win_rate = sum(self.recent_results) / len(self.recent_results)

        if win_rate <= 0.20 and self.current_stage > 0:
            self.current_stage -= 1
            print(f"\n--- CURRICULUM DEMOTION: Stage {self.current_stage} (win_rate={win_rate:.2f}) ---")
            self.recent_results.clear()
            return True

        return False

    # ---------------- SETTINGS ----------------
    def get_settings(self):
        settings = copy.deepcopy(self.base_settings)
        stage = self.current_stage

        # ---------- STAGE 0: Tutorial ----------
        if stage == 0:
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = 20
            settings['max_episode_steps'] = 1000
            settings['enable_power_pellets'] = False

        # ---------- STAGE 1: Expansion ----------
        elif stage == 1:
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = 80
            settings['max_episode_steps'] = 4000
            settings['enable_power_pellets'] = False

        # ---------- STAGE 2: Map Mastery (FIXED) ----------
        elif stage == 2:
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = 200   # NOT full clear anymore
            settings['max_episode_steps'] = 8000
            settings['enable_power_pellets'] = False

        # ---------- STAGE 3: Ghost Intro ----------
        elif stage == 3:
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 1
            settings['ghost_speed'] = 1  # slow but real movement
            settings['pellets_to_win'] = 100
            settings['max_episode_steps'] = 6000
            settings['enable_power_pellets'] = False

        # ---------- STAGE 4: First Hunt ----------
        elif stage == 4:
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 2
            settings['ghost_speed'] = self.base_settings.get('ghost_speed', 2)
            settings['scatter_duration'] = 7
            settings['chase_duration'] = 20
            settings['pellets_to_win'] = 120  # encourage power pellet usage
            settings['enable_power_pellets'] = True
            settings['max_episode_steps'] = 8000

        # ---------- STAGE 5: Final ----------
        else:
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 4
            settings['ghost_speed'] = self.base_settings.get('ghost_speed', 2) + 0.5
            settings['pellets_to_win'] = -1  # full clear
            settings['enable_power_pellets'] = True
            settings['scatter_duration'] = 7
            settings['chase_duration'] = 20
            settings['max_episode_steps'] = max(settings.get('max_episode_steps', 15000), 15000)

        return settings

    # ---------------- UTILITY ----------------
    def get_stage(self):
        return self.current_stage