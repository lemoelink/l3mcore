import os
import re
import sys
import threading
import importlib.util
from .logger import app_logger

_PLUGIN_NAME_RE = re.compile(r'^[a-zA-Z0-9_-]+$')


class PluginManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super(PluginManager, cls).__new__(cls)
                    instance._plugins = []
                    instance._plugin_accepts_label = []
                    instance._load_plugins()
                    cls._instance = instance
        return cls._instance

    def _load_plugins(self):
        plugin_dir = os.path.join(os.getcwd(), 'plugins')
        if not os.path.exists(plugin_dir) or not os.path.isdir(plugin_dir):
            app_logger.debug("Plugin directory not found. Plugins disabled.")
            return

        for filename in sorted(os.listdir(plugin_dir)):
            if not filename.endswith(".py") or filename.startswith("__"):
                continue

            plugin_name = filename[:-3]

            if not _PLUGIN_NAME_RE.match(plugin_name):
                app_logger.warning(
                    f"Plugin filename '{filename}' contains invalid characters. Skipped."
                )
                continue

            namespace_key = f"lemoe_plugin.{plugin_name}"
            if namespace_key in sys.modules:
                app_logger.warning(
                    f"Plugin '{plugin_name}' is already registered. Skipped duplicate."
                )
                continue

            file_path = os.path.join(plugin_dir, filename)
            try:
                spec = importlib.util.spec_from_file_location(namespace_key, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[namespace_key] = module
                    spec.loader.exec_module(module)
                    self._plugins.append(module)

                    accepts_label = False
                    if hasattr(module, 'after_generation'):
                        try:
                            import inspect
                            sig = inspect.signature(module.after_generation)
                            accepts_label = len(sig.parameters) >= 2
                        except Exception:
                            accepts_label = False
                    self._plugin_accepts_label.append(accepts_label)

                    app_logger.info(f"Loaded plugin: {plugin_name}")
            except Exception as e:
                sys.modules.pop(namespace_key, None)
                app_logger.error(f"Failed to load plugin {plugin_name}: {e}")

    def hook_before_routing(self, prompt: str) -> str:
        for plugin in self._plugins:
            if hasattr(plugin, 'before_routing'):
                try:
                    result = plugin.before_routing(prompt)
                    if isinstance(result, str):
                        prompt = result
                    else:
                        app_logger.warning(
                            f"Plugin {plugin.__name__} (before_routing) returned "
                            f"{type(result).__name__} instead of str. Ignored."
                        )
                except Exception as e:
                    app_logger.error(f"Error in plugin {plugin.__name__} (before_routing): {e}")
        return prompt

    def hook_override_route(self, messages: list) -> str | None:
        for plugin in self._plugins:
            if hasattr(plugin, 'override_route'):
                try:
                    target = plugin.override_route(messages)
                    if target:
                        return target
                except Exception as e:
                    app_logger.error(f"Error in plugin {plugin.__name__} (override_route): {e}")
        return None

    def hook_after_generation(self, response: str, expert_label: str = None) -> str:
        for i, plugin in enumerate(self._plugins):
            if hasattr(plugin, 'after_generation'):
                try:
                    if self._plugin_accepts_label[i]:
                        result = plugin.after_generation(response, expert_label)
                    else:
                        result = plugin.after_generation(response)

                    if isinstance(result, str):
                        response = result
                    else:
                        app_logger.warning(
                            f"Plugin {plugin.__name__} (after_generation) returned "
                            f"{type(result).__name__} instead of str. Ignored."
                        )
                except Exception as e:
                    app_logger.error(f"Error in plugin {plugin.__name__} (after_generation): {e}")
        return response
