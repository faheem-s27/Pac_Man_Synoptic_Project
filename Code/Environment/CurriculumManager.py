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
    def __init__(
        self,
        settings_path: str = None,
        recent_window: int = 150,
        promotion_threshold_stages_0_2: float = 0.75,
        promotion_threshold_stages_3_5: float = 0.70,
        promotion_threshold_stages_6_plus: float = 0.60,
        promotion_threshold_all_stages: float | None = None,
        tail_check_size: int = 5,
        tail_check_enabled: bool = True,
        tail_threshold_margin: float = 0.05,
    ):
        path = _resolve_settings_path(settings_path)
        self.base_settings = Settings(path).get_all()

        # Starvation budget is derived from required map-clear ratio:
        # starvation_limit_ticks = base + ratio * scale
        self.starvation_base_ticks = int(self.base_settings.get("starvation_base_ticks", 900))
        self.starvation_ratio_scale = int(self.base_settings.get("starvation_ratio_scale", 900))

        # Rolling performance window (algorithm-specific in suite).
        self.recent_results = collections.deque(maxlen=max(1, int(recent_window)))
        self.current_stage = 0

        self._promotion_threshold_stages_0_2 = float(promotion_threshold_stages_0_2)
        self._promotion_threshold_stages_3_5 = float(promotion_threshold_stages_3_5)
        self._promotion_threshold_stages_6_plus = float(promotion_threshold_stages_6_plus)
        self._promotion_threshold_all_stages = (
            None if promotion_threshold_all_stages is None else float(promotion_threshold_all_stages)
        )

        self._tail_check_enabled = bool(tail_check_enabled)
        self._tail_check_size = max(0, int(tail_check_size))
        self._tail_threshold_margin = max(0.0, float(tail_threshold_margin))

        # Expanded 8-stage ladder with an extra one-ghost bridge before two-ghost play.
        self.stage_profiles = [
            # Stage 0: Basic navigation, tiny pellet target.
            {
                "window_resolution": "800x800",
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.20,
                "pellets_to_win": -1,
                "max_episode_steps": 2500,
                "starvation_limit_ticks": 1000,
            },
            # Stage 1: Full navigation mastery, no ghosts.
            {
                "window_resolution": "800x800",
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.60,
                "pellets_to_win": -1,
                "max_episode_steps": 4000,
                "starvation_limit_ticks": 1300,
            },
            # Stage 2: One slow ghost, mostly scatter, no power pellets.
            {
                "window_resolution": "800x800",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "ghost_speed": 0.9,
                "scatter_duration": 20,
                "chase_duration": 5,
                "enable_power_pellets": False,
                "pellets_to_win_ratio": 0.50,
                "pellets_to_win": -1,
                "max_episode_steps": 5500,
                "starvation_limit_ticks": 1350,
            },
            # Stage 3: Ghost-eating bridge with explicit bonus.
            {
                "window_resolution": "800x800",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "ghost_speed": 1.1,
                "scatter_duration": 10,
                "chase_duration": 10,
                "enable_power_pellets": True,
                "ghost_eat_bonus": 150.0,
                "pellets_to_win_ratio": 0.55,
                "pellets_to_win": -1,
                "max_episode_steps": 6500,
                "starvation_limit_ticks": 1400,
            },
            # Stage 4: One ghost bridge (faster, no explicit ghost-eat bonus).
            {
                "window_resolution": "800x800",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "ghost_speed": 1.4,
                "scatter_duration": 10,
                "chase_duration": 12,
                "enable_power_pellets": True,
                "pellets_to_win_ratio": 0.55,
                "pellets_to_win": -1,
                "max_episode_steps": 7000,
                "starvation_limit_ticks": 1450,
            },
            # Stage 5: Two ghosts, power pellets, no bridge bonus.
            {
                "window_resolution": "800x800",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": True,
                "inky_active": False,
                "clyde_active": False,
                "ghost_speed": 1.2,
                "scatter_duration": 8,
                "chase_duration": 15,
                "enable_power_pellets": True,
                "pellets_to_win_ratio": 0.55,
                "pellets_to_win": -1,
                "max_episode_steps": 8500,
                "starvation_limit_ticks": 1550,
            },
            # Stage 6: Three ghosts, higher speed.
            {
                "window_resolution": "800x800",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": True,
                "inky_active": True,
                "clyde_active": False,
                "ghost_speed": 1.55,
                "scatter_duration": 7,
                "chase_duration": 18,
                "enable_power_pellets": True,
                "pellets_to_win_ratio": 0.70,
                "pellets_to_win": -1,
                "max_episode_steps": 10000,
                "starvation_limit_ticks": 1600,
            },
            # Stage 7: Full game.
            {
                "window_resolution": "800x800",
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": True,
                "inky_active": True,
                "clyde_active": True,
                "ghost_speed": 1.85,
                "scatter_duration": 7,
                "chase_duration": 20,
                "enable_power_pellets": True,
                "pellets_to_win_ratio": 1.0,
                "pellets_to_win": -1,
                "max_episode_steps": 13000,
                "starvation_limit_ticks": 1800,
            },
        ]

    def _promotion_threshold(self) -> float:
        if self._promotion_threshold_all_stages is not None:
            return self._promotion_threshold_all_stages
        if self.current_stage <= 2:
            return self._promotion_threshold_stages_0_2
        if self.current_stage <= 5:
            return self._promotion_threshold_stages_3_5
        return self._promotion_threshold_stages_6_plus

    def update_performance(self, won: bool):
        self.recent_results.append(bool(won))

    # ---------------- PROMOTION ----------------
    def check_promotion(self) -> bool:
        if len(self.recent_results) < self.recent_results.maxlen:
            return False

        # Cap progression at last configured stage.
        if self.current_stage >= len(self.stage_profiles) - 1:
            return False

        win_rate = sum(self.recent_results) / len(self.recent_results)
        threshold = self._promotion_threshold()

        promote = win_rate >= threshold
        if promote and self._tail_check_enabled and self._tail_check_size > 0:
            # Stability check over a configurable tail window.
            tail_n = min(self._tail_check_size, len(self.recent_results))
            recent_tail = list(self.recent_results)[-tail_n:]
            recent_tail_rate = sum(recent_tail) / max(1, len(recent_tail))
            tail_threshold = max(0.0, threshold - self._tail_threshold_margin)
            promote = recent_tail_rate >= tail_threshold

        if promote:
            self.current_stage += 1
            print(
                f"\n--- CURRICULUM PROMOTION: Stage {self.current_stage} "
                f"(win_rate={win_rate:.2f}, threshold={threshold:.2f}) ---"
            )
            self.recent_results.clear()
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