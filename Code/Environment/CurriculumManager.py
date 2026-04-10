import os
import collections
import copy
from Code.Settings import Settings

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _resolve_settings_path(settings_path: str | None) -> str:
    if settings_path:
        if os.path.isabs(settings_path):
            return settings_path
        candidates = [
            os.path.join(_ROOT, settings_path),
            os.path.join(_HERE, settings_path),
            os.path.join(os.getcwd(), settings_path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    default_candidates = [
        os.path.join(_ROOT, "game_settings.json"),
        os.path.join(_HERE, "game_settings.json"),
        os.path.join(os.getcwd(), "game_settings.json"),
    ]
    for candidate in default_candidates:
        if os.path.exists(candidate):
            return candidate
    return default_candidates[0]


class CurriculumManager:
    def __init__(self, settings_path: str = None):
        path = _resolve_settings_path(settings_path)
        self.base_settings = Settings(path).get_all()

        # Starvation budget is derived from required map-clear ratio:
        # starvation_limit_ticks = base + ratio * scale
        self.starvation_base_ticks = int(self.base_settings.get("starvation_base_ticks", 900))
        self.starvation_ratio_scale = int(self.base_settings.get("starvation_ratio_scale", 900))

        # Rolling performance window
        self.recent_results = collections.deque(maxlen=50)
        self.current_stage = 0

        # Operational Tracker for Grace Periods
        self.episodes_in_current_stage = 0

        # Slow-and-steady difficulty ladder: small maps first, full game last.
        self.stage_profiles = [
            # 0-4: routing fundamentals, no ghosts.
            {
                "window_resolution": "700x700",
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.10,
                "pellets_to_win": -1,
                "max_episode_steps": 1800,
                "starvation_limit_ticks": 900,
            },
            {
                "window_resolution": "700x700",
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.20,
                "pellets_to_win": -1,
                "max_episode_steps": 2200,
                "starvation_limit_ticks": 1000,
            },
            {
                "window_resolution": "800x800",
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.30,
                "pellets_to_win": -1,
                "max_episode_steps": 2600,
                "starvation_limit_ticks": 1100,
            },
            {
                "window_resolution": "800x800",
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.40,
                "pellets_to_win": -1,
                "max_episode_steps": 3200,
                "starvation_limit_ticks": 1200,
            },
            {
                "window_resolution": "800x800",
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.50,
                "pellets_to_win": -1,
                "max_episode_steps": 3600,
                "starvation_limit_ticks": 1300,
            },
            # 5-7: gentle ghost introduction.
            {
                "window_resolution": "900x900",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "ghost_speed": 1.2,
                "scatter_duration": 5,
                "chase_duration": 15,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.50,
                "pellets_to_win": -1,
                "max_episode_steps": 6200,
                "starvation_limit_ticks": 1400,
            },
            {
                "window_resolution": "1000x1000",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": True,
                "inky_active": False,
                "clyde_active": False,
                "ghost_speed": 1.45,
                "scatter_duration": 5,
                "chase_duration": 15,
                "enable_power_pellets": True,
                "pellets_to_win_ratio": 0.50,
                "pellets_to_win": -1,
                "max_episode_steps": 7600,
                "starvation_limit_ticks": 1500,
            },
            {
                "window_resolution": "1000x1000",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": True,
                "inky_active": True,
                "clyde_active": False,
                "ghost_speed": 1.65,
                "scatter_duration": 10,
                "chase_duration": 10,
                "enable_power_pellets": True,
                "pellets_to_win_ratio": 0.65,
                "pellets_to_win": -1,
                "max_episode_steps": 9800,
                "starvation_limit_ticks": 1600,
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
                "pellets_to_win_ratio": 0.80,
                "pellets_to_win": -1,
                "max_episode_steps": 12500,
                "starvation_limit_ticks": 1700,
            },
            {
                "window_resolution": "1100x1100",
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
                "starvation_limit_ticks": 1800,
            },
        ]

    def _promotion_threshold(self) -> float:
        if self.current_stage <= 2:
            return 0.65
        if self.current_stage <= 5:
            return 0.70
        return 0.65

    def _demotion_threshold(self) -> float:
        if self.current_stage <= 2:
            return 0.02
        if self.current_stage <= 5:
            return 0.05
        return 0.10

    def _demotion_grace_episodes(self) -> int:
        if self.current_stage <= 2:
            return 60
        if self.current_stage <= 5:
            return 90
        return 120

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

        # Stability check: last 5 should stay close to the stage threshold.
        recent_5 = list(self.recent_results)[-5:]
        recent_5_rate = sum(recent_5) / max(1, len(recent_5))
        tail_threshold = max(0.0, threshold - 0.05)

        if win_rate >= threshold and recent_5_rate >= tail_threshold:
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
    def _compute_starvation_limit_ticks(self, settings: dict) -> int:
        ratio = float(settings.get("pellets_to_win_ratio", 0.0) or 0.0)
        ratio = max(0.0, min(1.0, ratio))
        ticks = int(round(self.starvation_base_ticks + ratio * self.starvation_ratio_scale))
        return max(300, ticks)

    def get_settings(self):
        settings = copy.deepcopy(self.base_settings)

        stage_idx = min(self.current_stage, len(self.stage_profiles) - 1)
        stage_profile = self.stage_profiles[stage_idx]

        # Always randomise maze for generalisation.
        settings["maze_seed"] = None

        settings.update(stage_profile)
        settings["starvation_limit_ticks"] = self._compute_starvation_limit_ticks(settings)
        return settings

    # ---------------- UTILITY ----------------
    def get_stage(self):
        return self.current_stage