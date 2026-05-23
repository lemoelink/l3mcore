import os
import inspect
import importlib.util
from .logger import app_logger

class PluginManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(cls)
            cls._instance._plugins = []
            cls._instance._load_plugins()
        return cls._instance

    def _load_plugins(self):
        plugin_dir = os.path.join(os.getcwd(), 'plugins')
        if not os.path.exists(plugin_dir) or not os.path.isdir(plugin_dir):
            app_logger.debug("Plugin directory not found. Plugins disabled.")
            return

        for filename in os.listdir(plugin_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                plugin_name = filename[:-3]
                file_path = os.path.join(plugin_dir, filename)
                try:
                    spec = importlib.util.spec_from_file_location(plugin_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        self._plugins.append(module)
                        app_logger.info(f"Loaded plugin: {plugin_name}")
                except Exception as e:
                    app_logger.error(f"Failed to load plugin {plugin_name}: {e}")

    def hook_before_routing(self, prompt: str) -> str:
        """Called before the semantic router decides where to send the prompt."""
        for plugin in self._plugins:
            if hasattr(plugin, 'before_routing'):
                try:
                    prompt = plugin.before_routing(prompt)
                except Exception as e:
                    app_logger.error(f"Error in plugin {plugin.__name__} (before_routing): {e}")
        return prompt

    def hook_override_route(self, messages: list) -> str | None:
        """Called to let plugins completely bypass the semantic router based on the raw payload."""
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
        """Called after the expert generates the response, before returning to user."""
        for plugin in self._plugins:
            if hasattr(plugin, 'after_generation'):
                try:
                    sig = inspect.signature(plugin.after_generation)
                    if len(sig.parameters) >= 2:
                        response = plugin.after_generation(response, expert_label)
                    else:
                        response = plugin.after_generation(response)
                except Exception as e:
                    app_logger.error(f"Error in plugin {plugin.__name__} (after_generation): {e}")
        return response
