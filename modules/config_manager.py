import json
import os
import threading
import logging
import time
import base64

logger = logging.getLogger("ConfigManager")

_CONFIG_FILE = 'config/config.json'


def _deobfuscate_value(val):
    if isinstance(val, str):
        if val.startswith("env:"):
            env_var = val[4:]
            return os.getenv(env_var, "")
        elif val.startswith("base64:"):
            try:
                decoded = base64.b64decode(val[7:]).decode("utf-8")
                return decoded
            except Exception as e:
                logger.error(f"Error decoding base64 value '{val}': {e}")
                return val
        elif val.startswith("obfuscated:"):
            try:
                decoded = base64.b64decode(val[11:]).decode("utf-8")
                return decoded
            except Exception as e:
                logger.error(f"Error decoding obfuscated value '{val}': {e}")
                return val
    elif isinstance(val, dict):
        return {k: _deobfuscate_value(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_deobfuscate_value(v) for v in val]
    return val


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
        val = self._config.get(key, default)
        return _deobfuscate_value(val)

    def set(self, key, value):
        self._config[key] = value
        self.save()

    def get_all(self):
        return _deobfuscate_value(self._config)

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
