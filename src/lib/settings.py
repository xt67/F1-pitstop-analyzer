"""
Settings manager for F1 Pit Stop Analyzer application.
Handles loading, saving, and accessing application configuration.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional


class SettingsManager:
    """Manages application settings with JSON file persistence."""

    DEFAULTS = {
        "cache_path": "cache",
        "charts_path": "charts",
        "ml_models_path": "ml_models",
        "default_year": 2025,
        "auto_save_charts": False,
        "theme": "dark",
        "window_width": 1400,
        "window_height": 900,
    }

    _instance: Optional["SettingsManager"] = None

    def __new__(cls) -> "SettingsManager":
        """Singleton pattern to ensure only one settings instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._settings: dict = {}
        self._settings_file = self._get_settings_file_path()
        self.load()

    def _get_settings_file_path(self) -> Path:
        """Get the path to the settings file.

        Settings are stored in the user's app data directory for persistence
        across different working directories.
        """
        if os.name == "nt":  # Windows
            app_data = os.environ.get("APPDATA", os.path.expanduser("~"))
            settings_dir = Path(app_data) / "F1PitLab"
        else:  # macOS/Linux
            settings_dir = Path.home() / ".config" / "f1-pitlab"

        settings_dir.mkdir(parents=True, exist_ok=True)
        return settings_dir / "settings.json"

    def load(self) -> None:
        """Load settings from the JSON file."""
        self._settings = dict(self.DEFAULTS)

        if self._settings_file.exists():
            try:
                with open(self._settings_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self._settings.update(loaded)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load settings file: {e}")

    def save(self) -> None:
        """Save current settings to the JSON file."""
        try:
            with open(self._settings_file, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save settings file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value by key."""
        return self._settings.get(key, default if default is not None else self.DEFAULTS.get(key))

    def set(self, key: str, value: Any) -> None:
        """Set a setting value and save."""
        self._settings[key] = value
        self.save()

    def reset(self) -> None:
        """Reset all settings to defaults."""
        self._settings = dict(self.DEFAULTS)
        self.save()

    @property
    def cache_path(self) -> str:
        return self.get("cache_path")

    @property
    def charts_path(self) -> str:
        return self.get("charts_path")

    @property
    def ml_models_path(self) -> str:
        return self.get("ml_models_path")

    @property
    def default_year(self) -> int:
        return self.get("default_year")


def get_settings() -> SettingsManager:
    """Get the global settings manager instance."""
    return SettingsManager()
