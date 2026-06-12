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
                    instance._tools = []
                    instance._plugin_accepts_label = []
                    instance._load_plugins()
                    instance._load_tools()
                    cls._instance = instance
        return cls._instance

    def _load_tools(self):
        tool_dir = os.path.join(os.getcwd(), 'tools')
        if not os.path.exists(tool_dir) or not os.path.isdir(tool_dir):
            app_logger.debug("Tools directory not found. Tools disabled.")
            return

        for filename in sorted(os.listdir(tool_dir)):
            if not filename.endswith(".py") or filename.startswith("__"):
                continue

            tool_name = filename[:-3]

            if not _PLUGIN_NAME_RE.match(tool_name):
                app_logger.warning(
                    f"Tool filename '{filename}' contains invalid characters. Skipped."
                )
                continue

            namespace_key = f"l3mcore_tool.{tool_name}"
            if namespace_key in sys.modules:
                continue

            file_path = os.path.join(tool_dir, filename)
            try:
                spec = importlib.util.spec_from_file_location(namespace_key, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[namespace_key] = module
                    spec.loader.exec_module(module)
                    self._tools.append(module)
                    app_logger.info(f"Loaded tool: {tool_name}")
            except Exception as e:
                sys.modules.pop(namespace_key, None)
                app_logger.error(f"Failed to load tool {tool_name}: {e}")

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

            namespace_key = f"l3mcore_plugin.{plugin_name}"
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

    def hook_before_expert(self, messages: list, expert_config: dict) -> None:
        """
        Runs right before an expert executes, allowing plugins to inspect or modify
        the messages with knowledge of the chosen expert's configuration.
        """
        for plugin in self._plugins:
            if hasattr(plugin, 'before_expert'):
                try:
                    plugin.before_expert(messages, expert_config)
                except Exception as e:
                    app_logger.error(f"Error in plugin {plugin.__name__} (before_expert): {e}")

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

    def hook_before_request(self, request):
        for plugin in self._plugins:
            if hasattr(plugin, 'before_request'):
                try:
                    result = plugin.before_request(request)
                    if result is not None:
                        return result
                except Exception as e:
                    app_logger.error(f"Error in plugin {plugin.__name__} (before_request): {e}")
        return None

    def hook_on_startup(self, core_context: dict):
        for plugin in self._plugins:
            if hasattr(plugin, 'on_startup'):
                try:
                    plugin.on_startup(core_context)
                except Exception as e:
                    app_logger.error(f"Error in plugin {plugin.__name__} (on_startup): {e}")

    def hook_on_expert_failure(self, label: str, reason: str):
        for plugin in self._plugins:
            if hasattr(plugin, 'on_expert_failure'):
                try:
                    plugin.on_expert_failure(label, reason)
                except Exception as e:
                    app_logger.error(f"Error in plugin {plugin.__name__} (on_expert_failure): {e}")

    def hook_on_router_low_confidence(self, messages: list, current_label: str, current_score: float) -> str | None:
        for plugin in self._plugins:
            if hasattr(plugin, 'on_router_low_confidence'):
                try:
                    target = plugin.on_router_low_confidence(messages, current_label, current_score)
                    if target:
                        return target
                except Exception as e:
                    app_logger.error(f"Error in plugin {plugin.__name__} (on_router_low_confidence): {e}")
        return None

    def reload_plugins(self):
        """
        Re-scans the plugins directory and loads any new .py files not already registered.
        Already-loaded plugins are not reloaded to avoid state duplication.
        """
        plugin_dir = os.path.join(os.getcwd(), 'plugins')
        if not os.path.exists(plugin_dir) or not os.path.isdir(plugin_dir):
            app_logger.warning("reload_plugins: plugins directory not found.")
            return

        loaded_count = 0
        for filename in sorted(os.listdir(plugin_dir)):
            if not filename.endswith(".py") or filename.startswith("__"):
                continue

            plugin_name = filename[:-3]
            if not _PLUGIN_NAME_RE.match(plugin_name):
                continue

            namespace_key = f"l3mcore_plugin.{plugin_name}"
            if namespace_key in sys.modules:
                continue  # already loaded

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

                    app_logger.info(f"reload_plugins: loaded new plugin '{plugin_name}'.")
                    loaded_count += 1
            except Exception as e:
                sys.modules.pop(namespace_key, None)
                app_logger.error(f"reload_plugins: failed to load plugin '{plugin_name}': {e}")

        if loaded_count == 0:
            app_logger.info("reload_plugins: no new plugins found.")
        else:
            app_logger.info(f"reload_plugins: {loaded_count} new plugin(s) loaded.")

    def hook_get_tools(self) -> list:
        """
        Collects tool definitions from all tools that declare get_tools().
        Returns a flat list in OpenAI tool format, ready to include in a chat call.
        Only used when tool_calling.enabled = true in config.json.
        """
        tools = []
        for tool_module in self._tools:
            if not hasattr(tool_module, 'get_tools'):
                continue
            try:
                plugin_tools = tool_module.get_tools()
                if isinstance(plugin_tools, list):
                    tools.extend(plugin_tools)
            except Exception as e:
                app_logger.error(f"Error in tool {tool_module.__name__} (get_tools): {e}")
        return tools

    def hook_execute_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Routes a tool_call from the LLM to the tool module that owns that tool.
        Iterates tool modules in load order; the first one returning a non-None value wins.
        Returns a string result suitable for a role=tool message.
        """
        if not isinstance(tool_name, str) or not tool_name:
            return "Error: tool_name is empty or invalid."
        if not isinstance(arguments, dict):
            arguments = {}

        for tool_module in self._tools:
            if not hasattr(tool_module, 'execute_tool'):
                continue
            try:
                result = tool_module.execute_tool(tool_name, arguments)
                if result is not None:
                    return str(result)
            except Exception as e:
                app_logger.error(
                    f"Error in tool {tool_module.__name__} (execute_tool '{tool_name}'): {e}"
                )
        app_logger.warning(f"hook_execute_tool: no tool handled '{tool_name}'.")
        return f"Tool '{tool_name}' is not available or returned no result."


