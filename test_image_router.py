"""
Test suite for plugins/image_router.py
Run from the project root:
    python test_image_router.py
"""

import sys
import os
import importlib.util
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

class _SilentLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []
    def warning(self, msg):
        self.warnings.append(msg)
    def info(self, msg):
        self.infos.append(msg)
    def debug(self, msg):
        pass
    def error(self, msg):
        pass

_mock_logger = _SilentLogger()

import types
modules_mock = types.ModuleType("modules")
logger_mock = types.ModuleType("modules.logger")
logger_mock.app_logger = _mock_logger
sys.modules["modules"] = modules_mock
sys.modules["modules.logger"] = logger_mock
modules_mock.logger = logger_mock

spec = importlib.util.spec_from_file_location(
    "image_router",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins", "image_router.py")
)
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)

PASS = 0
FAIL = 0

def check(test_name, got, expected):
    global PASS, FAIL
    if got == expected:
        print(f"  PASS  {test_name}")
        PASS += 1
    else:
        print(f"  FAIL  {test_name}")
        print(f"        expected={expected!r}  got={got!r}")
        FAIL += 1

def reset_logger():
    _mock_logger.warnings.clear()
    _mock_logger.infos.clear()


print()
print("=" * 60)
print("image_router — test suite")
print("=" * 60)

print()
print("-- 1. Happy path: valid data-URI image --")
reset_logger()
b64_sample = "iVBORw0KGgoAAAANS" + "A" * 100 + "="
payload_valid = [
    {"role": "user", "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_sample}"}}
    ]}
]
check("returns image-expert", plugin.override_route(payload_valid), "image-expert")
check("logs info about detection", any("forcing route" in m for m in _mock_logger.infos), True)

print()
print("-- 2. Text-only message returns None --")
reset_logger()
payload_text = [{"role": "user", "content": "Hello, how are you?"}]
check("returns None for plain text", plugin.override_route(payload_text), None)

print()
print("-- 3. messages is not a list --")
reset_logger()
check("string input returns None", plugin.override_route("not a list"), None)
check("None input returns None", plugin.override_route(None), None)
check("dict input returns None", plugin.override_route({}), None)

print()
print("-- 4. Non-dict element in messages list --")
reset_logger()
payload_mixed = [
    "raw string element",
    42,
    None,
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_sample}"}}
    ]}
]
check("skips non-dict elements, finds image", plugin.override_route(payload_mixed), "image-expert")

print()
print("-- 5. content is not a list --")
reset_logger()
payload_str_content = [{"role": "user", "content": "plain string content"}]
check("returns None when content is string", plugin.override_route(payload_str_content), None)

payload_none_content = [{"role": "user", "content": None}]
check("returns None when content is None", plugin.override_route(payload_none_content), None)

print()
print("-- 6. image_url field is not a dict --")
reset_logger()
payload_bad_imgurl = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": "not-a-dict"}
    ]}
]
check("returns None", plugin.override_route(payload_bad_imgurl), None)
check("logs warning about image_url field", any("image_url field" in m for m in _mock_logger.warnings), True)

print()
print("-- 7. url field empty or None --")
reset_logger()
payload_empty_url = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": ""}}
    ]}
]
check("empty url returns None", plugin.override_route(payload_empty_url), None)
check("logs warning about empty url", any("empty" in m for m in _mock_logger.warnings), True)

reset_logger()
payload_none_url = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": None}}
    ]}
]
check("None url returns None", plugin.override_route(payload_none_url), None)

print()
print("-- 8. Malformed data-URI --")
reset_logger()
payload_bad_datauri = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:text/html;base64,PHNjcmlwdD4="}}
    ]}
]
check("non-image MIME returns None", plugin.override_route(payload_bad_datauri), None)
check("logs warning about format", any("data-URI" in m for m in _mock_logger.warnings), True)

reset_logger()
payload_bad_datauri2 = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;utf8,notbase64!!!"}}
    ]}
]
check("non-base64 encoding returns None", plugin.override_route(payload_bad_datauri2), None)

print()
print("-- 9. Unsafe schemes rejected --")
reset_logger()
payload_file_scheme = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}
    ]}
]
check("file:// scheme returns None", plugin.override_route(payload_file_scheme), None)
check("logs warning about scheme", any("unrecognized scheme" in m for m in _mock_logger.warnings), True)

reset_logger()
payload_ftp = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "ftp://attacker.com/img.jpg"}}
    ]}
]
check("ftp:// scheme returns None", plugin.override_route(payload_ftp), None)

print()
print("-- 10. External http/https URL accepted with warning --")
reset_logger()
payload_ext_url = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
    ]}
]
check("https URL returns image-expert", plugin.override_route(payload_ext_url), "image-expert")
check("logs warning about external URL", any("external image URL" in m for m in _mock_logger.warnings), True)

reset_logger()
payload_http_url = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "http://192.168.1.10/image.jpg"}}
    ]}
]
check("http URL returns image-expert", plugin.override_route(payload_http_url), "image-expert")

print()
print("-- 11. Inspection window limit (MAX_MESSAGES=10) --")
reset_logger()
old_msgs = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_sample}"}}
    ]}
]
filler = [{"role": "user", "content": "filler message"}] * 10
payload_window = old_msgs + filler
check(
    "image before window boundary not detected",
    plugin.override_route(payload_window),
    None
)

filler9 = [{"role": "user", "content": "filler"}] * 9
payload_in_window = old_msgs + filler9
check(
    "image within window boundary detected",
    plugin.override_route(payload_in_window),
    "image-expert"
)

print()
print("-- 12. Parts per message limit (MAX_PARTS_PER_MSG=20) --")
reset_logger()
filler_parts = [{"type": "text", "text": "filler"}] * 20
image_part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_sample}"}}
payload_parts_over = [{"role": "user", "content": filler_parts + [image_part]}]
check(
    "image beyond part 20 not detected",
    plugin.override_route(payload_parts_over),
    None
)

filler_parts19 = [{"type": "text", "text": "filler"}] * 19
payload_parts_in = [{"role": "user", "content": filler_parts19 + [image_part]}]
check(
    "image at position 20 detected",
    plugin.override_route(payload_parts_in),
    "image-expert"
)

print()
print("-- 13. Large base64 triggers warning but still routes --")
reset_logger()
big_b64 = "A" * (plugin._WARN_B64_BYTES + 1)
payload_big = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{big_b64}"}}
    ]}
]
check("oversized base64 still returns image-expert", plugin.override_route(payload_big), "image-expert")
check("logs warning about size", any("overload" in m for m in _mock_logger.warnings), True)

print()
print("-- 14. Empty messages list --")
reset_logger()
check("empty list returns None", plugin.override_route([]), None)

print()
print("-- 15. Multi-message history, image in second message --")
reset_logger()
payload_multi = [
    {"role": "user", "content": "First message, text only"},
    {"role": "assistant", "content": "Sure, I can help."},
    {"role": "user", "content": [
        {"type": "text", "text": "Now look at this"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_sample}"}}
    ]}
]
check("finds image across multiple messages", plugin.override_route(payload_multi), "image-expert")

print()
print("=" * 60)
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed  |  {FAIL} failed")
print("=" * 60)
print()

sys.exit(0 if FAIL == 0 else 1)
