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

    def update_performance(self, won: bool):
        self.recent_results.append(bool(won))
        self.episodes_in_current_stage += 1

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
            self.episodes_in_current_stage = 0  # Reset for grace period
            return True

        return False

    # ---------------- DEMOTION ----------------
    def check_demotion(self) -> bool:
        # VITAL: Provide a 150-episode grace period to allow the DQN to adjust
        # weights to the newly introduced MDP complexities (e.g. faster ghosts)
        if self.episodes_in_current_stage < 150:
            return False

        if len(self.recent_results) < self.recent_results.maxlen:
            return False

        win_rate = sum(self.recent_results) / len(self.recent_results)

        # Dropped demotion threshold to 15% to tolerate high-variance exploration
        if win_rate <= 0.15 and self.current_stage > 0:
            self.current_stage -= 1
            print(f"\n--- CURRICULUM DEMOTION: Stage {self.current_stage} (win_rate={win_rate:.2f}) ---")
            self.recent_results.clear()
            self.episodes_in_current_stage = 0  # Reset for stabilization
            return True

        return False

    # ---------------- SETTINGS ----------------
    def get_settings(self):
        settings = copy.deepcopy(self.base_settings)
        stage = self.current_stage

        # Always randomise maze for generalisation
        settings["maze_seed"] = None

        # ---------- STAGE 0: Early routing (no ghosts) ----------
        if stage == 0:
            settings.update({
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                # Relative target: clear 20% of spawned pellets on this map.
                "pellets_to_win_ratio": 0.20,
                "pellets_to_win": -1,
                "max_episode_steps": 2500,
                "enable_power_pellets": False,
            })

        # ---------- STAGE 1: Bigger clear targets (no ghosts) ----------
        elif stage == 1:
            settings.update({
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                # Relative target: clear 50% of spawned pellets on this map.
                "pellets_to_win_ratio": 0.50,
                "pellets_to_win": -1,
                "max_episode_steps": 5000,
                "enable_power_pellets": False,
            })

        # ---------- STAGE 2: Full-map mastery before ghosts ----------
        elif stage == 2:
            settings.update({
                "enable_ghosts": False,
                "blinky_active": False,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                # Relative target: clear all spawned pellets on this map.
                "pellets_to_win_ratio": 0.90,
                "pellets_to_win": -1,
                "max_episode_steps": 9000,
                "enable_power_pellets": False,
            })

        # ---------- STAGE 3: Ghost comfort (Blinky only) ----------
        elif stage == 3:
            settings.update({
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": False,
                "inky_active": False,
                "clyde_active": False,
                "ghost_speed": 1.3,
                "scatter_duration": 14,
                "chase_duration": 6,
                "pellets_to_win": -1,
                "max_episode_steps": 12000,
                "enable_power_pellets": False,
            })

        # ---------- STAGE 4: Two-ghost adaptation ----------
        elif stage == 4:
            settings.update({
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": True,
                "inky_active": False,
                "clyde_active": False,
                "ghost_speed": 1.7,
                "scatter_duration": 10,
                "chase_duration": 10,
                "pellets_to_win": -1,
                "enable_power_pellets": True,
                "max_episode_steps": 13000,
            })

        # ---------- STAGE 5+: Full game ----------
        else:
            settings.update({
                "enable_ghosts": True,
                "blinky_active": True,
                "pinky_active": True,
                "inky_active": True,
                "clyde_active": True,
                "ghost_speed": self.base_settings.get("ghost_speed", 2),
                "pellets_to_win": -1,
                "enable_power_pellets": True,
                "scatter_duration": 7,
                "chase_duration": 20,
                "max_episode_steps": 15000,
            })

        return settings

    # ---------------- UTILITY ----------------
    def get_stage(self):
        return self.current_stage