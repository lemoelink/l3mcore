import json
import os
import logging

logger = logging.getLogger("ConfigManager")

class ConfigManager:
    _instance = None
    _config = {}
    CONFIG_FILE = 'config/config.json'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.load()
        return cls._instance

    def load(self):
        """Loads configuration from JSON file."""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self._config = loaded if loaded is not None else {}
            else:
                logger.warning(f"Configuration file {self.CONFIG_FILE} not found. Using defaults.")
                self._config = {}
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._config = {}

    def save(self):
        """Saves current configuration to JSON file."""
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def get(self, key, default=None):
        """Retrieves a configuration value."""
        return self._config.get(key, default)

    def set(self, key, value):
        """Sets a configuration value and saves."""
        self._config[key] = value
        self.save()

    def get_all(self):
        """Returns the full configuration dictionary."""
        return self._config
