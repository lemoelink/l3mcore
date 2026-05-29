import json
import os
import threading
import logging

logger = logging.getLogger("ConfigManager")

_CONFIG_FILE = 'config/config.json'


class ConfigManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super(ConfigManager, cls).__new__(cls)
                    instance._config = {}
                    instance.load()
                    cls._instance = instance
        return cls._instance

    def load(self):
        try:
            if os.path.exists(_CONFIG_FILE):
                with open(_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self._config = loaded if loaded is not None else {}
            else:
                logger.warning(f"Configuration file {_CONFIG_FILE} not found. Using defaults.")
                self._config = {}
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._config = {}

    def save(self):
        try:
            tmp_path = _CONFIG_FILE + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4)
            os.replace(tmp_path, _CONFIG_FILE)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def get(self, key, default=None):
        return self._config.get(key, default)

    def set(self, key, value):
        self._config[key] = value
        self.save()

    def get_all(self):
        return self._config
