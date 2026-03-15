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

        Stages:
            0: Rapid Foraging      — small pellet target, no ghosts
            1: Map Mastery        — full clear, no ghosts
            2: Evasion Bridge     — 1 ghost, partial pellet target
            3: Hunting Context    — 2 ghosts, power pellets enabled
            4+: The Gauntlet      — 4 ghosts, full clear with power pellets, long horizon
        """
        settings = self.base_settings.copy()

        # Fallback: any stage > 4 uses Stage 4 configuration
        stage = min(self.current_stage, 4)

        if stage == 0:
            # Stage 0 — Rapid Foraging
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = 25
            settings['lives'] = 1
            settings['enable_power_pellets'] = False

        elif stage == 1:
            # Stage 1 — Map Mastery (full clear)
            settings['enable_ghosts'] = False
            settings['pellets_to_win'] = -1  # full clear
            settings['lives'] = 1
            settings['enable_power_pellets'] = False

        elif stage == 2:
            # Stage 2 — Evasion Bridge
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 1
            settings['pellets_to_win'] = 20
            settings['lives'] = 3
            settings['enable_power_pellets'] = False

        elif stage == 3:
            # Stage 3 — Hunting Context
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 2
            settings['pellets_to_win'] = 20
            settings['lives'] = 3
            settings['enable_power_pellets'] = True

        else:  # stage == 4 or higher
            # Stage 4 — The Gauntlet (fallback for all higher stages)
            settings['enable_ghosts'] = True
            settings['active_ghost_count'] = 4
            settings['pellets_to_win'] = -1
            settings['lives'] = 3
            settings['enable_power_pellets'] = True
            settings['max_episode_steps'] = 6000

        return settings

