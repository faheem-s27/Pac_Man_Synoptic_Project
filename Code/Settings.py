import json
import os

class Settings:
    """Manages game settings and persists them to a JSON file."""

    DEFAULT_SETTINGS = {
        "screen_width": 800,
        "screen_height": 800,
        "tile_size": 20,
        "pacman_speed": 2,
        "ghost_speed": -1,
        "maze_algorithm": "recursive_backtracking",
        "enable_ghosts": True,
        "lives": 3,
        "god_mode": False,
        "window_resolution": "800x800",
        "max_pellets": -1,
        "pellets_to_win": -1,
        "scatter_duration": 10,
        "chase_duration": 20,
        "always_chase": False,
        "level": 1,
        "ghost_speed_increment": 0.1,
        "enable_power_pellets": True,
        "blinky_active": True,
        "pinky_active": True,
        "inky_active": True,
        "clyde_active": True
    }

    @staticmethod
    def _apply_ghost_activation_migration(settings: dict) -> dict:
        """Normalize per-ghost toggles and migrate legacy active_ghost_count if present."""
        has_new_keys = any(
            key in settings
            for key in ("blinky_active", "pinky_active", "inky_active", "clyde_active")
        )

        if not has_new_keys and "active_ghost_count" in settings:
            try:
                count = int(settings.get("active_ghost_count", 4))
            except Exception:
                count = 4
            settings["blinky_active"] = count > 0
            settings["pinky_active"] = count > 1
            settings["inky_active"] = count > 2
            settings["clyde_active"] = count > 3

        settings["blinky_active"] = bool(settings.get("blinky_active", True))
        settings["pinky_active"] = bool(settings.get("pinky_active", True))
        settings["inky_active"] = bool(settings.get("inky_active", True))
        settings["clyde_active"] = bool(settings.get("clyde_active", True))

        # Keep a derived count for any still-legacy consumers.
        settings["active_ghost_count"] = int(
            settings["blinky_active"]
        ) + int(settings["pinky_active"]) + int(settings["inky_active"]) + int(settings["clyde_active"])
        return settings

    def __init__(self, settings_file="game_settings.json"):
        self.settings_file = settings_file
        self.settings = self.load_settings()

    def load_settings(self):
        """Load settings from file, or use defaults if file doesn't exist."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    loaded = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    settings = self.DEFAULT_SETTINGS.copy()
                    settings.update(loaded)
                    return self._apply_ghost_activation_migration(settings)
            except Exception as e:
                print(f"Error loading settings: {e}. Using defaults.")

        return self._apply_ghost_activation_migration(self.DEFAULT_SETTINGS.copy())

    def save_settings(self):
        """Save current settings to file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def get(self, key, default=None):
        """Get a setting value."""
        return self.settings.get(key, default)

    def set(self, key, value):
        """Set a setting value."""
        self.settings[key] = value
        self.save_settings()

    def get_all(self):
        """Get all settings as a dictionary."""
        return self.settings.copy()

    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.settings = self._apply_ghost_activation_migration(self.DEFAULT_SETTINGS.copy())
        self.save_settings()

