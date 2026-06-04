import json
import os
import threading
import logging
import time

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
                    instance._file_mtime = 0.0
                    instance.load()
                    cls._instance = instance
        return cls._instance

    def load(self):
        try:
            if os.path.exists(_CONFIG_FILE):
                with open(_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self._config = loaded if loaded is not None else {}
                self._file_mtime = os.path.getmtime(_CONFIG_FILE)
            else:
                logger.warning(f"Configuration file {_CONFIG_FILE} not found. Using defaults.")
                self._config = {}
                self._file_mtime = 0.0
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

    def check_for_changes(self):
        """Logs a warning if config.json on disk is newer than the loaded version."""
        try:
            if os.path.exists(_CONFIG_FILE):
                current_mtime = os.path.getmtime(_CONFIG_FILE)
                if current_mtime > self._file_mtime:
                    age = int(current_mtime - self._file_mtime)
                    logger.warning(
                        f"config.json has been modified on disk ({age}s ago) but the running "
                        "instance still uses the old version. Restart the server to apply changes."
                    )
        except Exception as e:
            logger.debug(f"ConfigManager.check_for_changes: {e}")
