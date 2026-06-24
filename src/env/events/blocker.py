from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


NOISE_TYPES = ("explicit failures", "implicit failures", "semantic misleading")


@dataclass(frozen=True)
class ToolSpec:
    tool_name: str
    input_datatypes: tuple[str, ...]
    output_datatype: str


@dataclass(frozen=True)
class ComboCandidate:
    edges: tuple[str, ...]
    covered_paths: tuple[int, ...]


def _normalize_runtime_task_entries(raw_tasks: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tasks, list):
        raise ValueError("blocker plan tasks must be a list.")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_tasks):
        if not isinstance(item, dict):
            raise ValueError(f"blocker plan tasks[{index}] must be an object.")
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"blocker plan tasks[{index}].task_id must be a non-empty string.")
        normalized.append(item)
    return normalized


def _normalize_noise_type(noise_type: str) -> str:
    return noise_type


def _task_plan_to_replacements(
    task_entry: dict[str, Any],
    *,
    tool_registry: dict[str, dict[str, Any]] | dict[str, ToolSpec],
) -> dict[str, list[dict[str, Any]]]:
    task_id = task_entry["task_id"]
    if task_entry.get("status") != "success":
        return {}

    selected_edge_set = task_entry.get("selected_edge_set", [])
    events = task_entry.get("events", [])
    if not isinstance(selected_edge_set, list) or not all(isinstance(x, str) and x.strip() for x in selected_edge_set):
        raise ValueError(f"blocker plan task {task_id} selected_edge_set must be list[str].")
    if not isinstance(events, list):
        raise ValueError(f"blocker plan task {task_id} events must be a list.")

    event_map: dict[str, list[dict[str, Any]]] = {}
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"blocker plan task {task_id} events[{event_index}] must be an object.")
        baseline_tool_name = event.get("baseline_tool_name")
        noise_type = event.get("noise_type")
        if not isinstance(baseline_tool_name, str) or not baseline_tool_name.strip():
            raise ValueError(
                f"blocker plan task {task_id} events[{event_index}].baseline_tool_name must be a non-empty string."
            )
        if not isinstance(noise_type, str) or not noise_type.strip():
            raise ValueError(
                f"blocker plan task {task_id} events[{event_index}].noise_type must be a non-empty string."
            )
        if baseline_tool_name not in tool_registry:
            raise ValueError(f"blocker plan task {task_id} references unknown baseline tool: {baseline_tool_name}")

        normalized_event = dict(event)
        normalized_event["noise_type"] = _normalize_noise_type(noise_type)
        event_map.setdefault(baseline_tool_name, []).append(normalized_event)

    if set(selected_edge_set) != set(event_map):
        raise ValueError(
            f"blocker plan task {task_id} selected_edge_set does not match events baseline_tool_name set."
        )

    return event_map


def generate_blocker_replacements_by_task(
    *,
    paths_set_catalog: dict[str, list[dict[str, Any]]],
    baseline_tools_path: Path,
    tasks_path: Path,
    selection_mode: str,
    block_n_per_task: int | None,
    target_remaining_paths: int | None,
    target_remaining_ratio: float | None,
    remaining_tolerance: int,
    min_remaining_paths: int,
    remaining_path_length_objective: str,
    blocking_edge_count_objective: str,
    seed: int,
    noise_mode: str,
    fixed_noise_type: str | None,
    fixed_noise_types: list[str] | None,
    multi_noise_count: int,
    max_combo_candidates: int,
    max_cover_size: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    raw_baseline_tools = load_json(baseline_tools_path)
    raw_tasks = load_json(tasks_path)
    tool_registry = build_tool_registry(raw_baseline_tools)
    task_registry = build_task_registry(raw_tasks)

    replacements_by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task_id, path_entries in paths_set_catalog.items():
        task = task_registry.get(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} appears in paths catalog but is missing in tasks.json.")
        ensure_task_shape(task_id, task)
        if not isinstance(path_entries, list):
            raise ValueError(f"paths_set_catalog[{task_id}] must be a list.")

        normalized_entries: list[dict[str, Any]] = []
        for path_index, entry in enumerate(path_entries):
            if not isinstance(entry, dict):
                raise ValueError(f"paths_set_catalog[{task_id}][{path_index}] must be an object.")
            tool_set = entry.get("tool_set")
            steps = entry.get("steps")
            if not isinstance(tool_set, list) or not all(isinstance(x, str) and x.strip() for x in tool_set):
                raise ValueError(f"paths_set_catalog[{task_id}][{path_index}].tool_set must be list[str].")
            if not isinstance(steps, int) or steps < 0:
                raise ValueError(f"paths_set_catalog[{task_id}][{path_index}].steps must be a non-negative int.")
            for tool_name in tool_set:
                if tool_name not in tool_registry:
                    raise ValueError(f"Task {task_id} references unknown baseline tool in paths_set_catalog: {tool_name}")
            normalized_entries.append({"tools": sorted(set(tool_set)), "steps": steps})

        task_plan = generate_task_plan(
            task_id=task_id,
            task=task,
            path_entries=normalized_entries,
            tool_registry=tool_registry,
            block_n_per_task=block_n_per_task,
            selection_mode=selection_mode,
            target_remaining_paths=target_remaining_paths,
            target_remaining_ratio=target_remaining_ratio,
            remaining_tolerance=remaining_tolerance,
            min_remaining_paths=min_remaining_paths,
            remaining_path_length_objective=remaining_path_length_objective,
            blocking_edge_count_objective=blocking_edge_count_objective,
            seed=seed,
            noise_mode=noise_mode,
            fixed_noise_type=fixed_noise_type,
            fixed_noise_types=fixed_noise_types,
            multi_noise_count=multi_noise_count,
            max_cover_size=max_cover_size,
            max_combo_candidates=max_combo_candidates,
        )
        replacements_by_task[task_id] = _task_plan_to_replacements(task_plan, tool_registry=tool_registry)

    return replacements_by_task


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def stable_int_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_tool_registry(raw_tools: Any) -> dict[str, ToolSpec]:
    if not isinstance(raw_tools, list):
        raise ValueError("baseline_tools.json must be a list.")

    registry: dict[str, ToolSpec] = {}
    for index, item in enumerate(raw_tools):
        if not isinstance(item, dict):
            raise ValueError(f"baseline_tools[{index}] must be an object.")
        tool_name = item.get("name") or item.get("tool_name")
        input_datatypes = item.get("input_datatypes")
        output_datatype = item.get("output_datatype")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError(f"baseline_tools[{index}].name must be a non-empty string.")
        if not isinstance(input_datatypes, list) or not all(isinstance(x, str) and x.strip() for x in input_datatypes):
            raise ValueError(f"baseline_tools[{index}].input_datatypes must be list[str].")
        if not isinstance(output_datatype, str) or not output_datatype.strip():
            raise ValueError(f"baseline_tools[{index}].output_datatype must be a non-empty string.")
        if tool_name in registry:
            raise ValueError(f"Duplicate tool name in baseline_tools.json: {tool_name}")
        registry[tool_name] = ToolSpec(
            tool_name=tool_name,
            input_datatypes=tuple(input_datatypes),
            output_datatype=output_datatype,
        )
    return registry


def build_task_registry(raw_tasks: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_tasks, list):
        raise ValueError("tasks.json must be a list.")

    registry: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_tasks):
        if not isinstance(item, dict):
            raise ValueError(f"tasks[{index}] must be an object.")
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"tasks[{index}].task_id must be a non-empty string.")
        if task_id in registry:
            raise ValueError(f"Duplicate task_id in tasks.json: {task_id}")
        registry[task_id] = item
    return registry


def ensure_task_shape(task_id: str, task: dict[str, Any]) -> None:
    required_fields = ("input_datatypes", "num_inputs", "target_datatype")
    for field in required_fields:
        if field not in task:
            raise ValueError(f"Task {task_id} is missing required field: {field}")

    inputs = task["input_datatypes"]
    num_inputs = task["num_inputs"]
    target = task["target_datatype"]
    if not isinstance(inputs, list) or not all(isinstance(x, str) and x.strip() for x in inputs):
        raise ValueError(f"Task {task_id}: input_datatypes must be list[str].")
    if not isinstance(num_inputs, int) or num_inputs < 0:
        raise ValueError(f"Task {task_id}: num_inputs must be a non-negative int.")
    if num_inputs != len(inputs):
        raise ValueError(f"Task {task_id}: num_inputs={num_inputs} does not match len(input_datatypes)={len(inputs)}.")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"Task {task_id}: target_datatype must be a non-empty string.")


def validate_path_sequence(
    task_id: str,
    path_index: int,
    sequence: list[str],
    task: dict[str, Any],
    tool_registry: dict[str, ToolSpec],
) -> None:
    available = set(task["input_datatypes"])
    for step_index, tool_name in enumerate(sequence):
        tool = tool_registry.get(tool_name)
        if tool is None:
            raise ValueError(
                f"Task {task_id} path[{path_index}] step[{step_index}] unknown tool in baseline_tools: {tool_name}"
            )
        missing = [dtype for dtype in tool.input_datatypes if dtype not in available]
        if missing:
            raise ValueError(
                f"Task {task_id} path[{path_index}] step[{step_index}] tool={tool_name} "
                f"requires missing datatypes={missing}. Available={sorted(available)}"
            )
        available.add(tool.output_datatype)

    target = task["target_datatype"]
    if target not in available:
        raise ValueError(
            f"Task {task_id} path[{path_index}] cannot reach target_datatype={target}. "
            f"Available_end={sorted(available)}"
        )


def build_edge_cover_map(path_entries: list[dict[str, Any]]) -> dict[str, set[int]]:
    edge_cover: dict[str, set[int]] = {}
    for path_index, entry in enumerate(path_entries):
        unique_edges = set(entry["tools"])
        for edge in unique_edges:
            edge_cover.setdefault(edge, set()).add(path_index)
    return edge_cover


def enumerate_candidates(
    edges_sorted: list[str],
    edge_cover: dict[str, set[int]],
    max_cover_size: int,
    max_combo_candidates: int,
) -> tuple[list[ComboCandidate], bool]:
    # Empty combo means "do not block any edge", which should be a valid candidate.
    candidates: list[ComboCandidate] = [ComboCandidate(edges=tuple(), covered_paths=tuple())]
    truncated = False
    cover_upper = min(max_cover_size, len(edges_sorted))

    if len(candidates) >= max_combo_candidates:
        return candidates, True

    for cover_size in range(1, cover_upper + 1):
        for combo in itertools.combinations(edges_sorted, cover_size):
            covered: set[int] = set()
            for edge in combo:
                covered.update(edge_cover[edge])
            candidates.append(
                ComboCandidate(
                    edges=combo,
                    covered_paths=tuple(sorted(covered)),
                )
            )
            if len(candidates) >= max_combo_candidates:
                truncated = True
                break
        if truncated:
            break

    return candidates, truncated


def choose_noise_type(
    rng: random.Random,
    noise_mode: str,
    fixed_noise_type: str | None,
    fixed_noise_types: list[str] | None,
    multi_noise_count: int,
) -> list[str]:
    if noise_mode == "fixed":
        assert fixed_noise_type is not None
        return [fixed_noise_type]
    if noise_mode == "random":
        return [NOISE_TYPES[rng.randrange(len(NOISE_TYPES))]]
    if noise_mode == "fixed_multi":
        assert fixed_noise_types is not None
        normalized = [_normalize_noise_type(item) for item in fixed_noise_types]
        deduped = list(dict.fromkeys(normalized))
        if not deduped:
            raise ValueError("fixed_multi mode requires at least one fixed noise type.")
        return deduped
    if noise_mode == "random_multi":
        count = min(max(1, multi_noise_count), len(NOISE_TYPES))
        return rng.sample(list(NOISE_TYPES), count)
    raise ValueError(f"Unsupported noise_mode: {noise_mode}")


def compute_remaining_path_indices(candidate: ComboCandidate, total_paths: int) -> list[int]:
    blocked = set(candidate.covered_paths)
    return [path_index for path_index in range(total_paths) if path_index not in blocked]


def compute_remaining_path_mean_steps(
    candidate: ComboCandidate,
    path_lengths: list[int],
    total_paths: int,
) -> Fraction:
    remaining_indices = compute_remaining_path_indices(candidate, total_paths)
    if not remaining_indices:
        raise ValueError("Cannot compute remaining path mean steps for candidate with zero remaining paths.")
    total_steps = sum(path_lengths[path_index] for path_index in remaining_indices)
    return Fraction(total_steps, len(remaining_indices))


def _fraction_to_float(value: Fraction) -> float:
    return float(value.numerator) / float(value.denominator)


def choose_path_length_bucket(
    candidates: list[ComboCandidate],
    *,
    path_lengths: list[int],
    total_paths: int,
    remaining_path_length_objective: str,
    rng: random.Random,
) -> tuple[list[ComboCandidate], dict[str, Any]]:
    meta: dict[str, Any] = {"remaining_path_length_objective": remaining_path_length_objective}
    if not candidates:
        return candidates, meta

    metric_by_candidate = {
        candidate: compute_remaining_path_mean_steps(candidate, path_lengths, total_paths) for candidate in candidates
    }
    distinct_metrics = sorted(set(metric_by_candidate.values()))
    meta["remaining_path_mean_steps_min"] = _fraction_to_float(distinct_metrics[0])
    meta["remaining_path_mean_steps_max"] = _fraction_to_float(distinct_metrics[-1])
    meta["remaining_path_mean_steps_distinct_count"] = len(distinct_metrics)

    if remaining_path_length_objective == "none":
        return list(candidates), meta

    if remaining_path_length_objective == "maximize":
        chosen_metric = distinct_metrics[-1]
    elif remaining_path_length_objective == "minimize":
        chosen_metric = distinct_metrics[0]
    elif remaining_path_length_objective in {"random", "random_middle"}:
        choice_pool = distinct_metrics
        pick = rng.randrange(len(choice_pool))
        chosen_metric = choice_pool[pick]
        meta["random_path_length_choice_pool_size"] = len(choice_pool)
        meta["random_path_length_choice_index"] = pick
    else:
        raise ValueError(f"Unsupported remaining_path_length_objective: {remaining_path_length_objective}")

    filtered = [candidate for candidate in candidates if metric_by_candidate[candidate] == chosen_metric]
    meta["selected_remaining_path_mean_steps"] = _fraction_to_float(chosen_metric)
    return filtered, meta


def choose_best_candidate(
    candidates: list[ComboCandidate],
    path_lengths: list[int],
    total_paths: int,
    selection_mode: str,
    block_n_per_task: int | None,
    target_remaining_paths: int | None,
    target_remaining_ratio: float | None,
    remaining_tolerance: int,
    min_remaining_paths: int,
    remaining_path_length_objective: str,
    blocking_edge_count_objective: str,
    rng: random.Random,
) -> tuple[ComboCandidate | None, dict[str, Any]]:
    if not candidates:
        return None, {"reason": "no_combo_candidates_under_limits"}

    meta: dict[str, Any] = {}
    if selection_mode == "exact_blocked_paths":
        assert block_n_per_task is not None
        effective_target = min(block_n_per_task, total_paths)
        meta["target_block_n"] = block_n_per_task
        meta["effective_target_block_n"] = effective_target
        primary_pool: list[ComboCandidate] = []
        for candidate in candidates:
            blocked_count = len(candidate.covered_paths)
            remaining = total_paths - blocked_count
            if blocked_count == effective_target and remaining >= min_remaining_paths:
                primary_pool.append(candidate)
        if not primary_pool:
            return None, {"reason": "no_feasible_exact_blocked_paths_combo", **meta}
        meta["selection_mode"] = selection_mode
        meta["distance_candidate_pool_size"] = len(primary_pool)
    else:
        if selection_mode == "target_remaining_paths":
            assert target_remaining_paths is not None
            target_remaining = max(min_remaining_paths, min(total_paths, target_remaining_paths))
            meta["target_remaining_paths"] = target_remaining_paths
            meta["effective_target_remaining_paths"] = target_remaining
            meta["remaining_tolerance"] = remaining_tolerance
        elif selection_mode == "target_remaining_ratio":
            assert target_remaining_ratio is not None
            derived = int(round(total_paths * target_remaining_ratio))
            target_remaining = max(min_remaining_paths, min(total_paths, derived))
            meta["target_remaining_ratio"] = target_remaining_ratio
            meta["effective_target_remaining_paths"] = target_remaining
            meta["remaining_tolerance"] = remaining_tolerance
        else:
            raise ValueError(f"Unsupported selection_mode: {selection_mode}")

        scored: list[tuple[int, ComboCandidate]] = []
        for candidate in candidates:
            blocked_count = len(candidate.covered_paths)
            remaining = total_paths - blocked_count
            if remaining < min_remaining_paths:
                continue
            distance = abs(remaining - target_remaining)
            if distance > remaining_tolerance:
                continue
            scored.append((distance, candidate))

        if not scored:
            return None, {"reason": "no_feasible_remaining_paths_combo_within_tolerance", **meta}

        best_distance = min(distance for distance, _ in scored)
        primary_pool = [candidate for distance, candidate in scored if distance == best_distance]
        meta["selection_mode"] = selection_mode
        meta["best_distance_to_target_remaining_paths"] = best_distance
        meta["distance_candidate_pool_size"] = len(primary_pool)

    path_length_pool, path_length_meta = choose_path_length_bucket(
        primary_pool,
        path_lengths=path_lengths,
        total_paths=total_paths,
        remaining_path_length_objective=remaining_path_length_objective,
        rng=rng,
    )
    meta.update(path_length_meta)
    meta["path_length_candidate_pool_size"] = len(path_length_pool)
    meta["blocking_edge_count_objective"] = blocking_edge_count_objective

    if blocking_edge_count_objective == "minimize":
        best_edge_count = min(len(candidate.edges) for candidate in path_length_pool)
        selection_pool = [candidate for candidate in path_length_pool if len(candidate.edges) == best_edge_count]
        meta["best_edge_count"] = best_edge_count
    elif blocking_edge_count_objective == "none":
        selection_pool = list(path_length_pool)
    else:
        raise ValueError(f"Unsupported blocking_edge_count_objective: {blocking_edge_count_objective}")

    pick = rng.randrange(len(selection_pool))
    meta["selection_index"] = pick
    meta["candidate_pool_size"] = len(selection_pool)
    return selection_pool[pick], meta


def generate_task_plan(
    task_id: str,
    task: dict[str, Any],
    path_entries: list[dict[str, Any]],
    tool_registry: dict[str, ToolSpec],
    block_n_per_task: int | None,
    selection_mode: str,
    target_remaining_paths: int | None,
    target_remaining_ratio: float | None,
    remaining_tolerance: int,
    min_remaining_paths: int,
    remaining_path_length_objective: str,
    blocking_edge_count_objective: str,
    seed: int,
    noise_mode: str,
    fixed_noise_type: str | None,
    fixed_noise_types: list[str] | None,
    multi_noise_count: int,
    max_cover_size: int,
    max_combo_candidates: int,
) -> dict[str, Any]:
    total_paths = len(path_entries)
    path_lengths = [entry["steps"] for entry in path_entries]

    edge_cover = build_edge_cover_map(path_entries)
    edges_sorted = sorted(edge_cover.keys())

    task_seed = seed + stable_int_from_text(task_id)
    rng = random.Random(task_seed)
    candidates, search_truncated = enumerate_candidates(
        edges_sorted=edges_sorted,
        edge_cover=edge_cover,
        max_cover_size=max_cover_size,
        max_combo_candidates=max_combo_candidates,
    )

    selected, selection_meta = choose_best_candidate(
        candidates=candidates,
        path_lengths=path_lengths,
        total_paths=total_paths,
        selection_mode=selection_mode,
        block_n_per_task=block_n_per_task,
        target_remaining_paths=target_remaining_paths,
        target_remaining_ratio=target_remaining_ratio,
        remaining_tolerance=remaining_tolerance,
        min_remaining_paths=min_remaining_paths,
        remaining_path_length_objective=remaining_path_length_objective,
        blocking_edge_count_objective=blocking_edge_count_objective,
        rng=rng,
    )

    if selected is None:
        return {
            "task_id": task_id,
            "status": "unresolved",
            "actual_blocked_path_count": 0,
            "actual_remaining_path_count": total_paths,
            "selected_edge_set": [],
            "blocked_path_indices": [],
            "events": [],
            "reason": str(selection_meta.get("reason", "no_feasible_combo")),
            "meta": {
                "total_paths": total_paths,
                "task_seed": task_seed,
                "max_cover_size": max_cover_size,
                "max_combo_candidates": max_combo_candidates,
                "search_truncated": search_truncated,
                **selection_meta,
            },
        }

    blocked_path_indices = list(selected.covered_paths)
    selected_edges = list(selected.edges)
    actual_blocked_path_count = len(blocked_path_indices)
    actual_remaining_path_count = total_paths - actual_blocked_path_count
    remaining_path_indices = compute_remaining_path_indices(selected, total_paths)
    remaining_path_steps = [path_lengths[path_index] for path_index in remaining_path_indices]
    actual_remaining_path_mean_steps = sum(remaining_path_steps) / len(remaining_path_steps)

    events: list[dict[str, Any]] = []
    for edge in selected_edges:
        noise_types = choose_noise_type(
            rng,
            noise_mode,
            fixed_noise_type,
            fixed_noise_types,
            multi_noise_count,
        )
        covered_paths_for_edge = sorted(edge_cover[edge])
        for noise_type in noise_types:
            events.append(
                {
                    "baseline_tool_name": edge,
                    "action": "replace",
                    "noise_type": noise_type,
                    "blocker_tool_name": None,
                    "reason": "selected_by_seeded_combo_sampling",
                    "meta": {
                        "task_id": task_id,
                        "edge_cover_count": len(covered_paths_for_edge),
                        "covered_path_indices": covered_paths_for_edge,
                        "rng_draw": {
                            "task_seed": task_seed,
                            "selection_mode": selection_mode,
                        },
                        "search_truncated": search_truncated,
                        "tool_input_datatypes": list(tool_registry[edge].input_datatypes),
                        "tool_output_datatype": tool_registry[edge].output_datatype,
                        "noise_mode": noise_mode,
                        "noise_types_for_edge": noise_types,
                    },
                }
            )

    return {
        "task_id": task_id,
        "status": "success",
        "actual_blocked_path_count": actual_blocked_path_count,
        "actual_remaining_path_count": actual_remaining_path_count,
        "selected_edge_set": selected_edges,
        "blocked_path_indices": blocked_path_indices,
        "events": events,
        "reason": "selected_by_seeded_combo_sampling",
        "meta": {
            "total_paths": total_paths,
            "task_seed": task_seed,
            "candidate_count": len(candidates),
            "max_cover_size": max_cover_size,
            "max_combo_candidates": max_combo_candidates,
            "search_truncated": search_truncated,
            "task_input_datatypes": task["input_datatypes"],
            "task_target_datatype": task["target_datatype"],
            "actual_remaining_path_mean_steps": actual_remaining_path_mean_steps,
            "actual_remaining_path_steps": remaining_path_steps,
            "selected_edge_count": len(selected_edges),
            **selection_meta,
        },
    }


def parse_task_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    picked: set[str] = set()
    for raw in values:
        parts = [part.strip() for part in raw.split(",")]
        for part in parts:
            if part:
                picked.add(part)
    return picked or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate blocker plan JSON from path sequences.")
    parser.add_argument(
        "--paths_set_catalog",
        required=True,
        help="Path to paths_set_catalog.json (deduplicated path definitions; required)",
    )
    parser.add_argument("--baseline_tools", required=True, help="Path to baseline_tools.json")
    parser.add_argument("--tasks", required=True, help="Path to tasks.json")
    parser.add_argument("--output", required=True, help="Output path for generated blocker plan json")
    parser.add_argument(
        "--selection_mode",
        choices=("exact_blocked_paths", "target_remaining_paths", "target_remaining_ratio"),
        default="target_remaining_paths",
        help="Selection strategy for choosing blocked edge set.",
    )
    parser.add_argument("--block_n_per_task", type=int, default=None, help="Required when selection_mode=exact_blocked_paths.")
    parser.add_argument(
        "--target_remaining_paths",
        type=int,
        default=None,
        help="Required when selection_mode=target_remaining_paths.",
    )
    parser.add_argument(
        "--target_remaining_ratio",
        type=float,
        default=None,
        help="Required when selection_mode=target_remaining_ratio. Example: 0.3 means keep ~30% paths.",
    )
    parser.add_argument(
        "--min_remaining_paths",
        type=int,
        default=1,
        help="Hard lower bound on remaining paths to keep task solvable.",
    )
    parser.add_argument(
        "--remaining_tolerance",
        type=int,
        default=1,
        help="Allowed absolute error for remaining-path targets in remaining-path modes.",
    )
    parser.add_argument(
        "--remaining_path_length_objective",
        choices=("none", "maximize", "minimize", "random", "random_middle"),
        default="none",
        help="Secondary selection objective over the remaining paths' average length.",
    )
    parser.add_argument(
        "--blocking_edge_count_objective",
        choices=("none", "minimize"),
        default="none",
        help="Optional tertiary selection objective over the number of blocked edges.",
    )
    parser.add_argument("--seed", required=True, type=int, help="Global seed for deterministic sampling")
    parser.add_argument("--noise_mode", required=True, choices=("random", "fixed", "random_multi", "fixed_multi"))
    parser.add_argument("--fixed_noise_type", choices=NOISE_TYPES, default=None)
    parser.add_argument("--fixed_noise_types", nargs="*", choices=NOISE_TYPES, default=None)
    parser.add_argument("--multi_noise_count", type=int, default=1)
    parser.add_argument("--max_combo_candidates", type=int, default=5000)
    parser.add_argument("--max_cover_size", type=int, default=4)
    parser.add_argument("--task_filter", nargs="*", default=None, help="Optional task ids (space or comma separated)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_remaining_paths < 0:
        raise ValueError("--min_remaining_paths must be >= 0.")
    if args.remaining_tolerance < 0:
        raise ValueError("--remaining_tolerance must be >= 0.")
    if args.remaining_path_length_objective not in {"none", "maximize", "minimize", "random", "random_middle"}:
        raise ValueError(
            "--remaining_path_length_objective must be one of: none, maximize, minimize, random, random_middle."
        )
    if args.blocking_edge_count_objective not in {"none", "minimize"}:
        raise ValueError("--blocking_edge_count_objective must be one of: none, minimize.")
    if args.selection_mode == "exact_blocked_paths":
        if args.block_n_per_task is None:
            raise ValueError("--block_n_per_task is required when --selection_mode=exact_blocked_paths.")
        if args.block_n_per_task < 0:
            raise ValueError("--block_n_per_task must be >= 0.")
    elif args.selection_mode == "target_remaining_paths":
        if args.target_remaining_paths is None:
            raise ValueError("--target_remaining_paths is required when --selection_mode=target_remaining_paths.")
        if args.target_remaining_paths < 0:
            raise ValueError("--target_remaining_paths must be >= 0.")
    elif args.selection_mode == "target_remaining_ratio":
        if args.target_remaining_ratio is None:
            raise ValueError("--target_remaining_ratio is required when --selection_mode=target_remaining_ratio.")
        if not (0.0 <= args.target_remaining_ratio <= 1.0):
            raise ValueError("--target_remaining_ratio must be between 0 and 1.")

    if args.max_combo_candidates <= 0:
        raise ValueError("--max_combo_candidates must be > 0.")
    if args.max_cover_size <= 0:
        raise ValueError("--max_cover_size must be > 0.")
    if args.noise_mode == "fixed" and args.fixed_noise_type is None:
        raise ValueError("--fixed_noise_type is required when --noise_mode=fixed.")
    if args.noise_mode == "random" and args.fixed_noise_type is not None:
        raise ValueError("--fixed_noise_type must not be set when --noise_mode=random.")
    if args.noise_mode != "fixed" and args.fixed_noise_type is not None:
        raise ValueError("--fixed_noise_type is only supported when --noise_mode=fixed.")
    if args.noise_mode == "fixed_multi":
        if not args.fixed_noise_types:
            raise ValueError("--fixed_noise_types is required when --noise_mode=fixed_multi.")
    elif args.fixed_noise_types:
        raise ValueError("--fixed_noise_types is only supported when --noise_mode=fixed_multi.")
    if args.noise_mode == "random_multi":
        if args.multi_noise_count <= 0:
            raise ValueError("--multi_noise_count must be > 0 when --noise_mode=random_multi.")
    elif args.multi_noise_count != 1:
        raise ValueError("--multi_noise_count is only supported when --noise_mode=random_multi.")

    paths_set_catalog_path = Path(args.paths_set_catalog).resolve()
    baseline_tools_path = Path(args.baseline_tools).resolve()
    tasks_path = Path(args.tasks).resolve()
    output_path = Path(args.output).resolve()

    raw_paths_catalog = load_json(paths_set_catalog_path)
    path_catalog_kind = "set"
    raw_baseline_tools = load_json(baseline_tools_path)
    raw_tasks = load_json(tasks_path)

    if not isinstance(raw_paths_catalog, dict):
        raise ValueError("paths_set_catalog.json must be an object: {task_id: [path_entry, ...]}")

    tool_registry = build_tool_registry(raw_baseline_tools)
    task_registry = build_task_registry(raw_tasks)
    task_filter = parse_task_filter(args.task_filter)

    selected_task_ids = []
    for task_id in raw_paths_catalog.keys():
        if task_filter is None or task_id in task_filter:
            selected_task_ids.append(task_id)

    if task_filter:
        unknown = sorted(task_filter.difference(raw_paths_catalog.keys()))
        if unknown:
            raise ValueError(f"task_filter contains unknown task_id(s) not in paths catalog: {unknown}")

    task_results: list[dict[str, Any]] = []
    for task_id in selected_task_ids:
        task = task_registry.get(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} appears in paths catalog but is missing in tasks.json.")
        ensure_task_shape(task_id, task)

        path_entries = raw_paths_catalog[task_id]
        if not isinstance(path_entries, list):
            raise ValueError(f"paths_catalog[{task_id}] must be a list.")

        normalized_entries: list[dict[str, Any]] = []
        for path_index, entry in enumerate(path_entries):
            if not isinstance(entry, dict):
                raise ValueError(f"paths_catalog[{task_id}][{path_index}] must be an object.")
            tool_set = entry.get("tool_set")
            steps = entry.get("steps")
            if not isinstance(tool_set, list) or not all(isinstance(x, str) and x.strip() for x in tool_set):
                raise ValueError(f"paths_set_catalog[{task_id}][{path_index}].tool_set must be list[str].")
            if not isinstance(steps, int) or steps < 0:
                raise ValueError(f"paths_set_catalog[{task_id}][{path_index}].steps must be a non-negative int.")
            for step_index, tool_name in enumerate(tool_set):
                if tool_name not in tool_registry:
                    raise ValueError(
                        f"Task {task_id} path_set[{path_index}] item[{step_index}] unknown tool in baseline_tools: {tool_name}"
                    )
            normalized_entries.append({"tools": sorted(set(tool_set)), "steps": steps})

        task_plan = generate_task_plan(
            task_id=task_id,
            task=task,
            path_entries=normalized_entries,
            tool_registry=tool_registry,
            block_n_per_task=args.block_n_per_task,
            selection_mode=args.selection_mode,
            target_remaining_paths=args.target_remaining_paths,
            target_remaining_ratio=args.target_remaining_ratio,
            remaining_tolerance=args.remaining_tolerance,
            min_remaining_paths=args.min_remaining_paths,
            remaining_path_length_objective=args.remaining_path_length_objective,
            blocking_edge_count_objective=args.blocking_edge_count_objective,
            seed=args.seed,
            noise_mode=args.noise_mode,
            fixed_noise_type=args.fixed_noise_type,
            fixed_noise_types=args.fixed_noise_types,
            multi_noise_count=args.multi_noise_count,
            max_cover_size=args.max_cover_size,
            max_combo_candidates=args.max_combo_candidates,
        )
        task_results.append(task_plan)

    output_payload = {
        "version": 1,
        "seed": args.seed,
        "generator_config": {
            "path_catalog_kind": path_catalog_kind,
            "paths_set_catalog": str(paths_set_catalog_path),
            "baseline_tools": str(baseline_tools_path),
            "tasks": str(tasks_path),
            "selection_mode": args.selection_mode,
            "block_n_per_task": args.block_n_per_task,
            "target_remaining_paths": args.target_remaining_paths,
            "target_remaining_ratio": args.target_remaining_ratio,
            "remaining_tolerance": args.remaining_tolerance,
            "min_remaining_paths": args.min_remaining_paths,
            "remaining_path_length_objective": args.remaining_path_length_objective,
            "blocking_edge_count_objective": args.blocking_edge_count_objective,
            "noise_mode": args.noise_mode,
            "fixed_noise_type": args.fixed_noise_type,
            "fixed_noise_types": args.fixed_noise_types,
            "multi_noise_count": args.multi_noise_count,
            "max_combo_candidates": args.max_combo_candidates,
            "max_cover_size": args.max_cover_size,
            "task_filter": sorted(task_filter) if task_filter else None,
        },
        "tasks": task_results,
    }
    dump_json(output_path, output_payload)


if __name__ == "__main__":
    main()
