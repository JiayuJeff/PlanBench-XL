from __future__ import annotations

from typing import Any

from env.core.types import RetrievalResult


class HierarchicalRetriever:
    def __init__(
        self,
        tool_registry: dict[str, dict[str, Any]],
        hierarchical_dimensions: list[str] | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.hierarchical_dimensions = hierarchical_dimensions or []

    def validate_keys(self, keys: dict[str, str]) -> tuple[bool, str]:
        if not isinstance(keys, dict):
            return False, "keys must be a JSON object"
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in keys.items()):
            return False, "keys must map strings to strings"
        return True, ""

    def retrieve_tools(
        self,
        keys: dict[str, str],
    ) -> RetrievalResult:
        tools = list(self.tool_registry.values())
        tools.sort(key=lambda tool: tool.get("name") or tool.get("tool_name") or "")
        return RetrievalResult(
            request={"keys": dict(keys)},
            tools=tools,
            matched_information=None,
            internal_retriever_note=None,
            model_retriever_note=None,
        )
