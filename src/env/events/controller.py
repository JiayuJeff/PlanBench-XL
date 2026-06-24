from __future__ import annotations

from typing import Any

from env.core.types import RetrievalResult


def _tool_identifier(tool: dict[str, Any]) -> str:
    return str(tool.get("name") or tool.get("tool_name") or "")


class EventController:
    def __init__(
        self,
        blocker_tools_by_domain: dict[str, list[dict[str, Any]]],
        enable_block: bool,
        blocker_replacements_by_task: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
    ) -> None:
        self.blocker_tools_by_domain = blocker_tools_by_domain
        self.enable_block = enable_block
        self.blocker_replacements_by_task = blocker_replacements_by_task or {}
        self.blocker_tool_lookup: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
        for domain, tools in blocker_tools_by_domain.items():
            domain_lookup: dict[tuple[str, str], dict[str, Any]] = {}
            for tool in tools:
                baseline_tool_name = tool.get("baseline_tool_name")
                noise_type = tool.get("noise_type")
                if not isinstance(baseline_tool_name, str) or not isinstance(noise_type, str):
                    continue
                domain_lookup[(baseline_tool_name, noise_type)] = tool
            self.blocker_tool_lookup[domain] = domain_lookup

    def augment_retrieval_result(
        self,
        query_id: str,
        task_id: str,
        retrieval_result: RetrievalResult,
    ) -> list[dict[str, Any]]:
        _ = query_id
        if not self.enable_block:
            return list(retrieval_result.tools)

        replacements = self.blocker_replacements_by_task.get(task_id, {})
        if not replacements:
            return list(retrieval_result.tools)

        replaced_tools: list[dict[str, Any]] = []
        for tool in retrieval_result.tools:
            replacement_events = replacements.get(_tool_identifier(tool))
            if not replacement_events:
                replaced_tools.append(dict(tool))
                continue
            for replacement_event in replacement_events:
                replaced_tools.append(self._resolve_blocker_tool(tool, replacement_event))
        return replaced_tools

    def _resolve_blocker_tool(
        self,
        baseline_tool: dict[str, Any],
        replacement_event: dict[str, Any],
    ) -> dict[str, Any]:
        domain = baseline_tool["domain"]
        baseline_tool_name = _tool_identifier(baseline_tool)
        noise_type = replacement_event["noise_type"]
        blocker_tool_name = replacement_event.get("blocker_tool_name")

        domain_lookup = self.blocker_tool_lookup.get(domain, {})
        blocker_tool = domain_lookup.get((baseline_tool_name, noise_type))
        if blocker_tool_name:
            candidate = blocker_tool
            if candidate is None or _tool_identifier(candidate) != blocker_tool_name:
                for tool in self.blocker_tools_by_domain.get(domain, []):
                    if _tool_identifier(tool) == blocker_tool_name:
                        candidate = tool
                        break
                blocker_tool = candidate
        if blocker_tool is None:
            raise ValueError(
                f"No blocker tool mapping found for baseline={baseline_tool_name}, noise_type={noise_type}, domain={domain}"
            )
        return dict(blocker_tool)

    def should_block_tool_call(
        self,
        query_id: str,
        task_id: str,
        step_id: int,
        tool_spec: dict[str, Any],
    ) -> bool:
        _ = (query_id, task_id, step_id, tool_spec)
        return False
