import json
import os
from pathlib import Path

class Settings:
    """Manages game settings and persists them to a JSON file."""

    DEFAULT_SETTINGS = {
        "screen_width": 800,
        "screen_height": 800,
        "tile_size": 20,
        "pacman_speed": 2,
        "ghost_speed": -1,
        "use_classic_maze": False,
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
        "enable_power_pellets": True
    }

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
                    return settings
            except Exception as e:
                print(f"Error loading settings: {e}. Using defaults.")

        return self.DEFAULT_SETTINGS.copy()

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
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.save_settings()

