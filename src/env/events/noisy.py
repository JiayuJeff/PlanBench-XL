from __future__ import annotations

import hashlib
import random
from typing import Any


def _tool_identifier(tool: dict[str, Any]) -> str:
    return str(tool.get("name") or tool.get("tool_name") or "")


def _noisy_source_identifier(tool: dict[str, Any]) -> str:
    baseline_tool_name = tool.get("baseline_tool_name")
    if isinstance(baseline_tool_name, str) and baseline_tool_name.strip():
        return baseline_tool_name
    return _tool_identifier(tool)


class NoisyToolAugmenter:
    def __init__(
        self,
        noisy_tools_by_domain: dict[str, dict[str, list[dict[str, Any]]]],
        mode: str,
        *,
        max_total_tools: int | None = None,
    ) -> None:
        self.noisy_tools_by_domain = noisy_tools_by_domain
        self.mode = mode
        self.max_total_tools = max_total_tools

    def _copy_unique_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        copied_tools: list[dict[str, Any]] = []
        seen_tool_names: set[str] = set()
        for tool in tools:
            tool_name = _tool_identifier(tool)
            if not tool_name or tool_name in seen_tool_names:
                continue
            copied_tools.append(dict(tool))
            seen_tool_names.add(tool_name)
        return copied_tools

    def _group_primary_tools(
        self,
        primary_tools: list[dict[str, Any]],
        domain_noisy_tools: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        grouped: list[dict[str, Any]] = []
        groups_by_key: dict[str, dict[str, Any]] = {}
        seen_tool_names = {
            tool_name for tool in primary_tools if (tool_name := _tool_identifier(tool))
        }

        for tool in primary_tools:
            group_key = _noisy_source_identifier(tool)
            primary_tool_name = _tool_identifier(tool)
            if not group_key or not primary_tool_name:
                continue
            group = groups_by_key.get(group_key)
            if group is None:
                siblings: list[dict[str, Any]] = []
                for noisy_tool in domain_noisy_tools.get(group_key, []):
                    noisy_tool_name = _tool_identifier(noisy_tool)
                    if not noisy_tool_name or noisy_tool_name in seen_tool_names:
                        continue
                    siblings.append(dict(noisy_tool))
                    seen_tool_names.add(noisy_tool_name)
                group = {
                    "group_key": group_key,
                    "primary_tools": [],
                    "is_blocked": False,
                    "noisy_pool": siblings,
                    "selected_noisy": [],
                }
                groups_by_key[group_key] = group
                grouped.append(group)
            group["primary_tools"].append(tool)
            if tool.get("baseline_tool_name"):
                group["is_blocked"] = True

        return grouped

    def _stable_rng(
        self,
        domain: str,
        group_labels: list[str],
        noisy_budget: int,
    ) -> random.Random:
        seed_material = "||".join([domain, str(noisy_budget), *group_labels])
        seed = int.from_bytes(
            hashlib.blake2b(seed_material.encode("utf-8"), digest_size=16).digest(),
            "big",
        )
        return random.Random(seed)

    def _allocate_round_robin(
        self,
        groups: list[dict[str, Any]],
        *,
        budget: int,
        rng: random.Random,
        max_per_group: int | None = None,
        balance_to_minimum: bool = False,
    ) -> int:
        spent = 0
        while spent < budget:
            eligible_groups = [
                group
                for group in groups
                if group["noisy_pool"]
                and (max_per_group is None or len(group["selected_noisy"]) < max_per_group)
            ]
            if not eligible_groups:
                break

            if balance_to_minimum:
                min_selected = min(len(group["selected_noisy"]) for group in eligible_groups)
                eligible_groups = [
                    group for group in eligible_groups if len(group["selected_noisy"]) == min_selected
                ]

            rng.shuffle(eligible_groups)
            assigned_in_round = False
            for group in eligible_groups:
                if spent >= budget:
                    break
                group["selected_noisy"].append(group["noisy_pool"].pop())
                spent += 1
                assigned_in_round = True
            if not assigned_in_round:
                break
        return spent

    def augment_retrieval_result(
        self,
        retrieval_tools: list[dict[str, Any]],
        *,
        domain: str,
        enable_block: bool,
    ) -> list[dict[str, Any]]:
        primary_tools = self._copy_unique_tools(retrieval_tools)
        if self.mode != "append_noisy_siblings":
            return primary_tools

        domain_noisy_tools = self.noisy_tools_by_domain.get(domain, {})
        groups = self._group_primary_tools(
            primary_tools,
            domain_noisy_tools,
        )
        total_candidates = sum(len(group["noisy_pool"]) for group in groups)
        if total_candidates == 0:
            return primary_tools

        primary_count = sum(len(group["primary_tools"]) for group in groups)
        noisy_budget = total_candidates
        if self.max_total_tools is not None:
            noisy_budget = min(total_candidates, max(0, self.max_total_tools - primary_count))
        if noisy_budget <= 0:
            return primary_tools

        group_labels = [
            f"{group['group_key']}:{'blocked' if group['is_blocked'] else 'baseline'}:{len(group['primary_tools'])}"
            for group in groups
        ]
        rng = self._stable_rng(domain, group_labels, noisy_budget)
        for group in groups:
            rng.shuffle(group["noisy_pool"])

        remaining_budget = noisy_budget
        unblocked_groups = [group for group in groups if not group["is_blocked"]]
        if unblocked_groups:
            stage_one_budget = min(remaining_budget, 2 * len(unblocked_groups))
            spent = self._allocate_round_robin(
                unblocked_groups,
                budget=stage_one_budget,
                rng=rng,
                max_per_group=2,
            )
            remaining_budget -= spent

        if remaining_budget > 0:
            self._allocate_round_robin(
                groups,
                budget=remaining_budget,
                rng=rng,
                balance_to_minimum=True,
            )

        mixed_tools: list[dict[str, Any]] = []
        for group in groups:
            mixed_tools.extend(group["primary_tools"])
            mixed_tools.extend(group["selected_noisy"])
        return mixed_tools
