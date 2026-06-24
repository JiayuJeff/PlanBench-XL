from __future__ import annotations

import json
import re
from typing import Any


ACTION_TAGS = ("retrieve_tools", "tool_call", "final_answer")


def extract_action(raw_text: str) -> tuple[str | None, str | None]:
    matches: list[tuple[str, str]] = []
    for action in ACTION_TAGS:
        pattern = re.compile(
            rf"<{action}>(.*?)</{action}>",
            flags=re.DOTALL | re.IGNORECASE,
        )
        found = pattern.findall(raw_text)
        for content in found:
            matches.append((action, content.strip()))

    if len(matches) != 1:
        return None, None
    return matches[0]


def parse_json_block(content: str) -> tuple[bool, dict[str, Any] | None]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False, None
    if not isinstance(payload, dict):
        return False, None
    return True, payload


def parse_retrieve_action(content: str) -> tuple[bool, dict[str, Any] | None]:
    ok, payload = parse_json_block(content)
    if not ok or payload is None:
        return False, None
    meta_tool = payload.get("meta_tool")
    if not isinstance(meta_tool, str) or not meta_tool.strip():
        return False, None
    return True, payload


def parse_tool_call_action(content: str) -> tuple[bool, dict[str, Any] | None]:
    ok, payload = parse_json_block(content)
    if not ok or payload is None:
        return False, None
    tool_name = payload.get("name")
    if not isinstance(tool_name, str):
        tool_name = payload.get("tool_name")
    tool_id = payload.get("tool_id")
    if not isinstance(tool_name, str) and not isinstance(tool_id, str):
        return False, None
    if not isinstance(payload.get("arguments"), dict):
        return False, None
    return True, payload
