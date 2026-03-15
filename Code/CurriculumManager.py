import os
import collections
from Code.Settings import Settings

_HERE = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_PATH = os.path.join(_HERE, "game_settings.json")

class CurriculumManager:
    def __init__(self, settings_path: str = None):
        path = settings_path or _SETTINGS_PATH
        self.base_settings = Settings(path).get_all()
        # Rolling window of recent episode outcomes (True = win, False = loss)
        self.recent_results = collections.deque(maxlen=50)
        self.current_stage = 0

    def update_performance(self, won: bool):
        """Record the outcome of a completed episode.

        Call this after each episode with won=True if the agent cleared the level,
        or won=False otherwise. This drives dynamic curriculum promotion.
        """
        self.recent_results.append(bool(won))

    def check_promotion(self) -> bool:
        """Check whether the agent should be promoted to the next stage.

        Promotion rule:
            - Once we have a full window of 50 results, compute
              win_rate = sum(recent_results) / len(recent_results).
            - If win_rate >= 0.80, clear the history, increment current_stage,
              print a promotion message, and return True.
            - Otherwise, return False.
        """
        if not self.recent_results:
            return False

        if len(self.recent_results) < self.recent_results.maxlen:
            return False

        win_rate = sum(self.recent_results) / len(self.recent_results)
        if win_rate >= 0.80:
            self.current_stage += 1
            print(f"\n--- CURRICULUM PROMOTION: Stage {self.current_stage} (win_rate={win_rate:.2f}) ---")
            self.recent_results.clear()
            return True

        return False

    def get_settings(self):
        """Return environment settings for the current curriculum stage.

        Balanced Progression:
            0: Tutorial      — 20 pellets, No ghosts. (Learning to move)
            1: Expansion     — 60 pellets, No ghosts. (Learning corridors)
            2: Ghost Intro   — 60 pellets, 1 Static Ghost. (Learning 'Danger' exists)
            3: First Hunt    — 80 pellets, 1 Active Ghost. (Learning Evasion)
            4: Power Play    — 100 pellets, 2 Active Ghosts + Power Pellets. (Learning Aggression)
            5: The Gauntlet  — Full Clear, 4 Active Ghosts. (Mastery)
        """
        settings = self.base_settings.copy()
        stage = self.current_stage  # No longer capping at 4

        if stage == 0:
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = 20
            settings['max_episode_steps'] = 1000
            settings['enable_power_pellets'] = False

        elif stage == 1:
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = 60
            settings['max_episode_steps'] = 2000
            settings['enable_power_pellets'] = False

        elif stage == 2:
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 1
            # PRO-TIP: Set ghost speed to 0.5 for this stage to make it a slow obstacle
            settings['ghost_speed'] = 0.5
            settings['pellets_to_win'] = 60
            settings['max_episode_steps'] = 3000
            settings['enable_power_pellets'] = False

        elif stage == 3:
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 1
            settings['ghost_speed'] = self.base_settings.get('ghost_speed', 2)
            settings['pellets_to_win'] = 80
            settings['max_episode_steps'] = 4000
            settings['enable_power_pellets'] = False

        elif stage == 4:
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 2
            settings['pellets_to_win'] = 100
            settings['enable_power_pellets'] = True
            settings['max_episode_steps'] = 5000

        else:  # Stage 5 Mastery
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 4
            settings['pellets_to_win'] = -1  # Full Clear
            settings['enable_power_pellets'] = True
            settings['max_episode_steps'] = 10000

        return settings

