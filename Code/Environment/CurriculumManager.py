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

        # Operational Tracker for Grace Periods
        self.episodes_in_current_stage = 0

        # Slow-and-steady difficulty ladder: small maps first, full game last.
        self.stage_profiles = [
            # 0-4: routing fundamentals, no ghosts.
            {
                "window_resolution": "600x600",
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.12,
                "pellets_to_win": -1,
                "max_episode_steps": 1800,
            },
            {
                "window_resolution": "600x600",
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.22,
                "pellets_to_win": -1,
                "max_episode_steps": 2200,
            },
            {
                "window_resolution": "600x600",
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.35,
                "pellets_to_win": -1,
                "max_episode_steps": 2600,
            },
            {
                "window_resolution": "700x700",
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.45,
                "pellets_to_win": -1,
                "max_episode_steps": 3200,
            },
            {
                "window_resolution": "700x700",
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.60,
                "pellets_to_win": -1,
                "max_episode_steps": 4200,
            },
            # 5-7: gentle ghost introduction.
            {
                "window_resolution": "800x800",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "ghost_speed": 1.2,
                "scatter_duration": 5,
                "chase_duration": 15,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.45,
                "pellets_to_win": -1,
                "max_episode_steps": 6200,
            },
            {
                "window_resolution": "800x800",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": True,
                "inky_active": False,
                "clyde_active": False,
                "ghost_speed": 1.45,
                "scatter_duration": 5,
                "chase_duration": 15,
                "enable_power_pellets": True,
                "pellets_to_win_ratio": 0.60,
                "pellets_to_win": -1,
                "max_episode_steps": 7600,
            },
            {
                "window_resolution": "900x900",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": True,
                "inky_active": True,
                "clyde_active": False,
                "ghost_speed": 1.65,
                "scatter_duration": 10,
                "chase_duration": 10,
                "enable_power_pellets": True,
                "pellets_to_win_ratio": 0.75,
                "pellets_to_win": -1,
                "max_episode_steps": 9800,
            },
            # 8-9: full board and full roster.
            {
                "window_resolution": "1000x1000",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": True,
                "inky_active": True,
                "clyde_active": True,
                "ghost_speed": 1.85,
                "scatter_duration": 8,
                "chase_duration": 12,
                "enable_power_pellets": True,
                "pellets_to_win_ratio": 0.90,
                "pellets_to_win": -1,
                "max_episode_steps": 12500,
            },
            {
                "window_resolution": "1000x1000",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": True,
                "inky_active": True,
                "clyde_active": True,
                "ghost_speed": self.base_settings.get("ghost_speed", 2),
                "scatter_duration": 7,
                "chase_duration": 20,
                "enable_power_pellets": True,
                "pellets_to_win_ratio": 1.0,
                "pellets_to_win": -1,
                "max_episode_steps": 15000,
            },
        ]

    def _promotion_threshold(self) -> float:
        if self.current_stage <= 2:
            return 0.70
        if self.current_stage <= 5:
            return 0.75
        return 0.80

    def _demotion_threshold(self) -> float:
        if self.current_stage <= 2:
            return 0.05
        if self.current_stage <= 5:
            return 0.10
        return 0.15

    def _demotion_grace_episodes(self) -> int:
        if self.current_stage <= 2:
            return 80
        if self.current_stage <= 5:
            return 120
        return 160

    def update_performance(self, won: bool):
        self.recent_results.append(bool(won))
        self.episodes_in_current_stage += 1

    # ---------------- PROMOTION ----------------
    def check_promotion(self) -> bool:
        if len(self.recent_results) < self.recent_results.maxlen:
            return False

        # Cap progression at last configured stage.
        if self.current_stage >= len(self.stage_profiles) - 1:
            return False

        win_rate = sum(self.recent_results) / len(self.recent_results)
        threshold = self._promotion_threshold()

        # Stability check: last 10 must also be strong.
        recent_10 = list(self.recent_results)[-10:]
        recent_10_rate = sum(recent_10) / 10

        if win_rate >= threshold and recent_10_rate >= threshold:
            self.current_stage += 1
            print(
                f"\n--- CURRICULUM PROMOTION: Stage {self.current_stage} "
                f"(win_rate={win_rate:.2f}, threshold={threshold:.2f}) ---"
            )
            self.recent_results.clear()
            self.episodes_in_current_stage = 0
            return True

        return False

    # ---------------- DEMOTION ----------------
    def check_demotion(self) -> bool:
        if self.current_stage == 0:
            return False

        if self.episodes_in_current_stage < self._demotion_grace_episodes():
            return False

        if len(self.recent_results) < self.recent_results.maxlen:
            return False

        win_rate = sum(self.recent_results) / len(self.recent_results)
        threshold = self._demotion_threshold()

        if win_rate <= threshold:
            self.current_stage -= 1
            print(
                f"\n--- CURRICULUM DEMOTION: Stage {self.current_stage} "
                f"(win_rate={win_rate:.2f}, threshold={threshold:.2f}) ---"
            )
            self.recent_results.clear()
            self.episodes_in_current_stage = 0
            return True

        return False

    # ---------------- SETTINGS ----------------
    def get_settings(self):
        settings = copy.deepcopy(self.base_settings)

        stage_idx = min(self.current_stage, len(self.stage_profiles) - 1)
        stage_profile = self.stage_profiles[stage_idx]

        # Always randomise maze for generalisation.
        settings["maze_seed"] = None

        settings.update(stage_profile)
        return settings

    # ---------------- UTILITY ----------------
    def get_stage(self):
        return self.current_stage