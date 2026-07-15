from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _bootstrap_path() -> Path:
    current_file = Path(__file__).resolve()
    src_root = current_file.parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return src_root


_SRC_ROOT = _bootstrap_path()

from env.core.utils import dump_json, load_json  # noqa: E402
from env.core.sampling import sample_sequence  # noqa: E402
from env.core.answer_judges import AnswerJudge, build_answer_judge  # noqa: E402


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    task_id: str
    correct_answer: Any
    input_datatypes: set[str]
    target_datatype: str

def resolve_project_path(project_root: Path, raw_path: str | None, fallback: Path) -> Path:
    if not raw_path:
        return fallback
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    malformed_lines = []
    if not path.exists():
        return rows, malformed_lines
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                malformed_lines.append(
                    {
                        "path": str(path),
                        "error": str(exc),
                        "line": line_no,
                        "column": exc.colno,
                        "position": exc.pos,
                    }
                )
    return rows, malformed_lines


def _is_http_400_error_message(message: Any) -> bool:
    if not isinstance(message, str):
        return False
    normalized = message.lower()
    return (
        "status=400" in normalized
        or "status code 400" in normalized
        or "http 400" in normalized
        or "error code: 400" in normalized
        or ("badrequesterror" in normalized and "code': 400" in normalized)
        or ('badrequesterror' in normalized and 'code": 400' in normalized)
    )


def _final_state_from_latest_state(latest_state: dict[str, Any], has_final_answer: bool) -> dict[str, Any]:
    return {
        "current_datatypes": list(latest_state.get("current_datatypes", [])),
        "available_tool_names": list(latest_state.get("available_tool_names", [])),
        "available_tool_ids": list(latest_state.get("available_tool_ids", [])),
        "discovered_tool_names": list(latest_state.get("discovered_tool_names", [])),
        "tool_name_to_id": dict(latest_state.get("tool_name_to_id", {})),
        "tool_id_to_name": dict(latest_state.get("tool_id_to_name", {})),
        "retrieval_attempt_count": latest_state.get("retrieval_attempt_count", 0),
        "retrieval_exec_count": latest_state.get("retrieval_exec_count", 0),
        "tool_call_attempt_count": latest_state.get("tool_call_attempt_count", 0),
        "tool_call_exec_count": latest_state.get("tool_call_exec_count", 0),
        "incorrect_final_answer_feedback_count": latest_state.get("incorrect_final_answer_feedback_count", 0),
        "trusted_values_by_datatype": dict(latest_state.get("trusted_values_by_datatype", {})),
        "untrusted_values_by_datatype": dict(latest_state.get("untrusted_values_by_datatype", {})),
        "untrusted_value_sources_by_datatype": dict(latest_state.get("untrusted_value_sources_by_datatype", {})),
        "has_final_answer": has_final_answer,
    }


def pending_http_400_result(progress: dict[str, Any]) -> dict[str, Any] | None:
    if progress.get("status") != "pending":
        return None

    query_id = progress.get("query_id")
    if not isinstance(query_id, str):
        return None

    turns = progress.get("turns")
    if not isinstance(turns, list):
        return None

    last_error: dict[str, Any] | None = None
    last_error_turn_id: int | None = None
    for turn in reversed(turns):
        if not isinstance(turn, dict):
            continue
        error = turn.get("error")
        if isinstance(error, dict) and error.get("message"):
            last_error = error
            turn_id = turn.get("turn_id")
            last_error_turn_id = turn_id if isinstance(turn_id, int) else None
            break

    error_message = None if last_error is None else last_error.get("message")
    if not _is_http_400_error_message(error_message):
        return None

    latest_state = progress.get("latest_state") or {}
    trace = latest_state.get("steps_trace")
    if not isinstance(trace, list):
        trace = []
    step_count = latest_state.get("step_count")
    if not isinstance(step_count, int):
        step_count = len(trace)

    return {
        "query_id": query_id,
        "stop_reason": "runtime_error_http_400",
        "status": "failed",
        "final_answer": None,
        "steps": step_count,
        "trace": trace,
        "final_state": _final_state_from_latest_state(latest_state, has_final_answer=False),
        "failure_reason_detail": "pending_runtime_error_http_400",
        "progress_status": progress.get("status"),
        "runtime_error": {
            "type": last_error.get("type"),
            "message": error_message,
            "turn_id": last_error_turn_id,
        },
    }


def load_results(output_dir: Path) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    result_jsonl = output_dir / "result.jsonl"
    rows, malformed_result_files = load_jsonl(result_jsonl)

    query_dir = output_dir / "progress" / "queries"
    if not query_dir.exists():
        if rows:
            return rows, "result_jsonl", malformed_result_files
        return [], "none", malformed_result_files

    progress_rows = []
    progress_rows_by_query_id: dict[str, dict[str, Any]] = {}
    for path in sorted(query_dir.glob("*.json")):
        try:
            progress = load_json(path)
        except ValueError as exc:
            malformed_result_files.append(
                {
                    "path": str(path),
                    "error": str(exc),
                    "line": None,
                    "column": None,
                    "position": None,
                }
            )
            continue
        final_result = progress.get("final_result")
        if isinstance(final_result, dict):
            query_id = final_result.get("query_id")
            if isinstance(query_id, str):
                progress_rows_by_query_id[query_id] = final_result
            else:
                progress_rows.append(final_result)
            continue

        synthetic_result = pending_http_400_result(progress)
        if isinstance(synthetic_result, dict):
            query_id = synthetic_result.get("query_id")
            if isinstance(query_id, str):
                progress_rows_by_query_id[query_id] = synthetic_result
            else:
                progress_rows.append(synthetic_result)

    if rows:
        merged_rows = []
        seen_query_ids: set[str] = set()
        for row in rows:
            merged_rows.append(row)
            query_id = row.get("query_id")
            if isinstance(query_id, str):
                seen_query_ids.add(query_id)
        for query_id, row in progress_rows_by_query_id.items():
            if query_id not in seen_query_ids:
                merged_rows.append(row)
        merged_rows.extend(progress_rows)
        source = "result_jsonl+progress_queries" if progress_rows_by_query_id or progress_rows else "result_jsonl"
        return merged_rows, source, malformed_result_files

    merged_progress_rows = list(progress_rows_by_query_id.values()) + progress_rows
    if merged_progress_rows:
        return merged_progress_rows, "progress_queries", malformed_result_files
    return [], "none", malformed_result_files


def build_tool_registry(baseline_tools_path: Path) -> dict[str, dict[str, Any]]:
    tools = {}
    for tool in load_json(baseline_tools_path):
        name = tool.get("name") or tool.get("tool_name")
        if isinstance(name, str):
            tools[name] = tool
    return tools


def build_noisy_tool_name_set(noisy_tools_path: Path) -> set[str]:
    if not noisy_tools_path.exists():
        return set()

    tool_names: set[str] = set()
    for tool in load_json(noisy_tools_path):
        name = tool.get("name") or tool.get("tool_name")
        if isinstance(name, str):
            tool_names.add(name)
    return tool_names


def load_query_records(query_path: Path) -> list[QueryRecord]:
    records = []
    for item in load_json(query_path):
        records.append(
            QueryRecord(
                query_id=item["query_id"],
                task_id=item["task_id"],
                correct_answer=item["correct_answer"],
                input_datatypes=set(item.get("input_datatypes", [])),
                target_datatype=item["target_datatype"],
            )
        )
    return records


def load_progress_query_ids(output_dir: Path) -> list[str] | None:
    index_path = output_dir / "progress" / "index.json"
    if not index_path.exists():
        return None
    index = load_json(index_path)
    query_ids = []
    for item in index.get("queries", []):
        query_id = item.get("query_id")
        if isinstance(query_id, str):
            query_ids.append(query_id)
    return query_ids


def ground_truth_datatypes_for_task(
    task_id: str,
    paths_set_catalog: dict[str, list[dict[str, Any]]],
    tool_registry: dict[str, dict[str, Any]],
) -> set[str]:
    datatypes: set[str] = set()
    for path_spec in paths_set_catalog.get(task_id, []):
        for tool_name in path_spec.get("tool_set", []):
            tool = tool_registry.get(tool_name)
            if not tool:
                continue
            datatypes.update(tool.get("input_datatypes", []))
            output_datatype = tool.get("output_datatype")
            if isinstance(output_datatype, str):
                datatypes.add(output_datatype)
    return datatypes


def close_seen_tool_outputs(
    known_datatypes: set[str],
    seen_tools: list[dict[str, Any]],
) -> set[str]:
    expanded = set(known_datatypes)
    changed = True
    while changed:
        changed = False
        for tool in seen_tools:
            input_datatypes = set(tool.get("input_datatypes", []))
            output_datatype = tool.get("output_datatype")
            if not isinstance(output_datatype, str):
                continue
            if input_datatypes.issubset(expanded) and output_datatype not in expanded:
                expanded.add(output_datatype)
                changed = True
    return expanded


def trace_metrics(
    result: dict[str, Any],
    query: QueryRecord,
    noisy_tool_names: set[str] | None = None,
) -> dict[str, Any]:
    trace = result.get("trace") or []
    explored_datatypes = set(query.input_datatypes)
    seen_tools: list[dict[str, Any]] = []
    executed_datatypes: set[str] = set()
    search_turns = 0
    call_turns = 0
    invalid_tool_call_count = 0
    untrusted_input_rejection_count = 0
    noisy_tool_call_count = 0
    noisy_tool_names = noisy_tool_names or set()

    for step in trace:
        action = step.get("action")
        if action == "retrieve_tools":
            search_turns += 1
            tools = ((step.get("retrieval_result") or {}).get("tools") or [])
            for tool in tools:
                if isinstance(tool, dict):
                    seen_tools.append(tool)
            explored_datatypes = close_seen_tool_outputs(explored_datatypes, seen_tools)
            continue

        if action == "call_tool":
            call_turns += 1
            parse_ok = step.get("parse_ok")
            blocked = bool(step.get("blocked"))
            internal_error_type = step.get("internal_error_type")
            if parse_ok is False:
                invalid_tool_call_count += 1
            elif parse_ok is True and "tool_result" not in step and not blocked:
                if internal_error_type == "untrusted_input_rejected":
                    untrusted_input_rejection_count += 1
                else:
                    invalid_tool_call_count += 1
            request = step.get("request") or {}
            tool_result = step.get("tool_result") or {}
            request_name = request.get("name")
            tool_type = str(tool_result.get("tool_type") or "").lower()
            if (isinstance(request_name, str) and request_name in noisy_tool_names) or tool_type.startswith("noisy"):
                noisy_tool_call_count += 1
            output_datatype = tool_result.get("output_datatype")
            output_value = tool_result.get("output_value")
            output_provenance = tool_result.get("output_provenance")
            trusted_output_acquired = output_provenance == "trusted"
            if output_provenance is None:
                trusted_output_acquired = isinstance(output_datatype, str) and output_value is not None
            if isinstance(output_datatype, str) and trusted_output_acquired:
                executed_datatypes.add(output_datatype)
                explored_datatypes.add(output_datatype)
                explored_datatypes = close_seen_tool_outputs(explored_datatypes, seen_tools)

    discovered_datatypes = explored_datatypes - query.input_datatypes
    combined_invalid_tool_call_count = invalid_tool_call_count + untrusted_input_rejection_count
    return {
        "search_turns": search_turns,
        "call_turns": call_turns,
        "search_to_call_ratio": safe_ratio(search_turns, call_turns),
        "invalid_tool_call_count": invalid_tool_call_count,
        "invalid_tool_call_rate": safe_ratio(invalid_tool_call_count, call_turns),
        "untrusted_input_rejection_count": untrusted_input_rejection_count,
        "untrusted_input_rejection_rate": safe_ratio(untrusted_input_rejection_count, call_turns),
        "combined_invalid_tool_call_count": combined_invalid_tool_call_count,
        "combined_invalid_tool_call_rate": safe_ratio(combined_invalid_tool_call_count, call_turns),
        "noisy_tool_call_count": noisy_tool_call_count,
        "noisy_tool_call_rate": safe_ratio(noisy_tool_call_count, call_turns),
        "explored_datatype_count": len(discovered_datatypes),
        "explored_datatypes": sorted(discovered_datatypes),
        "executed_datatypes": sorted(executed_datatypes),
    }


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def result_turn_count(result: dict[str, Any]) -> int:
    steps = result.get("steps")
    if isinstance(steps, int):
        return steps
    trace = result.get("trace")
    if isinstance(trace, list):
        return len(trace)
    return 0


def last_final_answer_trace_entry(result: dict[str, Any]) -> dict[str, Any] | None:
    trace = result.get("trace")
    if not isinstance(trace, list):
        return None
    for step in reversed(trace):
        if isinstance(step, dict) and step.get("action") == "final_answer":
            return step
    return None


def datatype_check_passed_for_result(result: dict[str, Any], query: QueryRecord) -> bool:
    final_answer_step = last_final_answer_trace_entry(result)
    if final_answer_step is None:
        return False

    datatype_check_passed = final_answer_step.get("datatype_check_passed")
    if isinstance(datatype_check_passed, bool):
        return datatype_check_passed

    target_datatype_reached = final_answer_step.get("target_datatype_reached")
    matched_untrusted_source_types = final_answer_step.get("matched_untrusted_source_types") or []
    if isinstance(target_datatype_reached, bool):
        return target_datatype_reached and not matched_untrusted_source_types

    final_state = result.get("final_state") or {}
    current_datatypes = final_state.get("current_datatypes") or []
    if isinstance(current_datatypes, list):
        return query.target_datatype in current_datatypes
    return False


def evaluate_output(
    output_dir: Path,
    answer_judge: AnswerJudge,
    query_file: Path | None = None,
    baseline_tools_file: Path | None = None,
    noisy_tools_file: Path | None = None,
    paths_set_catalog_file: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    project_root = _SRC_ROOT.parent
    metadata_path = output_dir / "metadata.json"
    metadata = load_json(metadata_path)
    config = metadata.get("config") or {}
    data_config = config.get("data") or {}

    default_query_path = project_root / "src" / "data" / metadata["domain"] / "queries.json"
    query_path = query_file or resolve_project_path(
        project_root,
        data_config.get("query_file"),
        default_query_path,
    )
    if not query_path.exists() and query_file is None:
        query_path = default_query_path
    baseline_tools_path = baseline_tools_file or resolve_project_path(
        project_root,
        data_config.get("baseline_tools_file"),
        project_root / "src" / "data" / metadata["domain"] / "baseline_tools.json",
    )
    noisy_tools_path = noisy_tools_file or resolve_project_path(
        project_root,
        data_config.get("noisy_tools_file"),
        project_root / "src" / "data" / metadata["domain"] / "noisy_tools.json",
    )
    paths_set_path = paths_set_catalog_file or resolve_project_path(
        project_root,
        data_config.get("paths_set_catalog_file"),
        project_root / "src" / "data" / metadata["domain"] / "paths_set_catalog.json",
    )

    query_records = load_query_records(query_path)
    query_sample = config.get("query_sample") or {}
    query_records = sample_sequence(
        query_records,
        query_sample.get("size"),
        int(query_sample.get("seed", 42)),
    )

    progress_query_ids = load_progress_query_ids(output_dir)
    if progress_query_ids is not None:
        progress_query_id_set = set(progress_query_ids)
        query_records = [query for query in query_records if query.query_id in progress_query_id_set]

    queries = {query.query_id: query for query in query_records}
    tool_registry = build_tool_registry(baseline_tools_path)
    noisy_tool_names = build_noisy_tool_name_set(noisy_tools_path)
    paths_set_catalog = load_json(paths_set_path)
    results, result_source, malformed_result_files = load_results(output_dir)

    per_query = []
    correct_count = 0
    evaluated_count = 0
    total_search_turns = 0
    total_call_turns = 0
    total_invalid_tool_calls = 0
    total_untrusted_input_rejections = 0
    total_combined_invalid_tool_calls = 0
    total_noisy_tool_calls = 0
    turn_counts: list[int] = []
    explored_counts: list[int] = []
    executed_gt_precisions: list[float] = []
    executed_gt_recalls: list[float] = []

    for result in results:
        query_id = result.get("query_id")
        query = queries.get(query_id)
        if query is None:
            continue

        datatype_check_passed = datatype_check_passed_for_result(result, query)
        if datatype_check_passed:
            answer_correct = answer_judge.is_correct(query.correct_answer, result.get("final_answer"))
            evaluation_failure_reason_detail = None if answer_correct else "final_answer_wrong"
        else:
            answer_correct = False
            evaluation_failure_reason_detail = "target_datatype_not_reached"
        trace_stats = trace_metrics(result, query, noisy_tool_names=noisy_tool_names)
        ground_truth_datatypes = ground_truth_datatypes_for_task(
            query.task_id,
            paths_set_catalog,
            tool_registry,
        )
        executed_datatypes = set(trace_stats["executed_datatypes"])
        executed_gt_datatypes = executed_datatypes & ground_truth_datatypes

        executed_gt_precision = safe_ratio(len(executed_gt_datatypes), len(executed_datatypes))
        executed_gt_recall = safe_ratio(len(executed_gt_datatypes), len(ground_truth_datatypes))
        turn_count = result_turn_count(result)

        evaluated_count += 1
        correct_count += int(answer_correct)
        total_search_turns += trace_stats["search_turns"]
        total_call_turns += trace_stats["call_turns"]
        total_invalid_tool_calls += trace_stats["invalid_tool_call_count"]
        total_untrusted_input_rejections += trace_stats["untrusted_input_rejection_count"]
        total_combined_invalid_tool_calls += trace_stats["combined_invalid_tool_call_count"]
        total_noisy_tool_calls += trace_stats["noisy_tool_call_count"]
        turn_counts.append(turn_count)
        explored_counts.append(trace_stats["explored_datatype_count"])
        if executed_gt_precision is not None:
            executed_gt_precisions.append(executed_gt_precision)
        if executed_gt_recall is not None:
            executed_gt_recalls.append(executed_gt_recall)

        per_query.append(
            {
                "query_id": query.query_id,
                "task_id": query.task_id,
                "answer_correct": answer_correct,
                "datatype_check_passed": datatype_check_passed,
                "expected_answer": query.correct_answer,
                "final_answer": result.get("final_answer"),
                "evaluation_failure_reason_detail": evaluation_failure_reason_detail,
                "runtime_status": result.get("status"),
                "runtime_progress_status": result.get("progress_status"),
                "runtime_failure_reason_detail": result.get("failure_reason_detail"),
                "runtime_error_message": ((result.get("runtime_error") or {}).get("message")),
                "steps": result.get("steps"),
                "turn_count": turn_count,
                "search_turns": trace_stats["search_turns"],
                "call_turns": trace_stats["call_turns"],
                "search_to_call_ratio": trace_stats["search_to_call_ratio"],
                "invalid_tool_call_count": trace_stats["invalid_tool_call_count"],
                "invalid_tool_call_rate": trace_stats["invalid_tool_call_rate"],
                "untrusted_input_rejection_count": trace_stats["untrusted_input_rejection_count"],
                "untrusted_input_rejection_rate": trace_stats["untrusted_input_rejection_rate"],
                "combined_invalid_tool_call_count": trace_stats["combined_invalid_tool_call_count"],
                "combined_invalid_tool_call_rate": trace_stats["combined_invalid_tool_call_rate"],
                "noisy_tool_call_count": trace_stats["noisy_tool_call_count"],
                "noisy_tool_call_rate": trace_stats["noisy_tool_call_rate"],
                "used_noisy_tool": trace_stats["noisy_tool_call_count"] > 0,
                "explored_datatype_count": trace_stats["explored_datatype_count"],
                "explored_datatypes": trace_stats["explored_datatypes"],
                "executed_datatypes": sorted(executed_datatypes),
                "ground_truth_datatypes": sorted(ground_truth_datatypes),
                "executed_ground_truth_datatypes": sorted(executed_gt_datatypes),
                "executed_gt_datatype_precision": executed_gt_precision,
                "executed_gt_datatype_recall": executed_gt_recall,
            }
        )

    return {
        "output_dir": str(output_dir),
        "result_source": result_source,
        "malformed_result_files": malformed_result_files,
        "answer_judge": answer_judge.name,
        "query_count": len(queries),
        "evaluated_count": evaluated_count,
        "unevaluated_count": max(len(queries) - evaluated_count, 0),
        "metrics": {
            "accuracy": safe_ratio(correct_count, evaluated_count),
            "correct_count": correct_count,
            "turns": {
                "mean_turn_count": mean(turn_counts),
                "total_turn_count": sum(turn_counts),
            },
            "exploration": {
                "mean_explored_datatype_count": mean(explored_counts),
                "total_explored_datatype_count": sum(explored_counts),
            },
            "explore_exploit_balance": {
                "total_search_turns": total_search_turns,
                "total_call_turns": total_call_turns,
                "search_to_call_ratio": safe_ratio(total_search_turns, total_call_turns),
            },
            "executed_ground_truth_datatype_coverage": {
                "mean_precision": mean(executed_gt_precisions),
                "mean_recall": mean(executed_gt_recalls),
            },
            "invalid_tool_call": {
                "total_invalid_tool_calls": total_invalid_tool_calls,
                "total_tool_call_attempts": total_call_turns,
                "invalid_tool_call_rate": safe_ratio(total_invalid_tool_calls, total_call_turns),
            },
            "untrusted_input_rejection": {
                "total_untrusted_input_rejections": total_untrusted_input_rejections,
                "total_tool_call_attempts": total_call_turns,
                "untrusted_input_rejection_rate": safe_ratio(total_untrusted_input_rejections, total_call_turns),
            },
            "combined_invalid_tool_call": {
                "total_combined_invalid_tool_calls": total_combined_invalid_tool_calls,
                "total_tool_call_attempts": total_call_turns,
                "combined_invalid_tool_call_rate": safe_ratio(total_combined_invalid_tool_calls, total_call_turns),
            },
            "noisy_tool_call": {
                "queries_with_noisy_tool_calls": sum(
                    1 for item in per_query if item["noisy_tool_call_count"] > 0
                ),
                "noisy_tool_query_rate": safe_ratio(
                    sum(1 for item in per_query if item["noisy_tool_call_count"] > 0),
                    len(per_query),
                ),
                "total_noisy_tool_calls": total_noisy_tool_calls,
                "mean_noisy_tool_calls_per_query": safe_ratio(total_noisy_tool_calls, len(per_query)),
                "mean_noisy_tool_calls_given_any": safe_ratio(
                    total_noisy_tool_calls,
                    sum(1 for item in per_query if item["noisy_tool_call_count"] > 0),
                ),
                "noisy_tool_call_rate_among_tool_calls": safe_ratio(total_noisy_tool_calls, total_call_turns),
            },
        },
        "per_query": per_query,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tool-planning run outputs.")
    parser.add_argument("--output_dir", required=True, help="Run output directory containing metadata.json.")
    parser.add_argument("--output", default=None, help="Evaluation JSON path. Defaults to <output_dir>/evaluation.json.")
    parser.add_argument(
        "--answer_judge",
        default="normalized_contains",
        choices=["normalized_contains", "llm_judge"],
        help="Final-answer correctness strategy.",
    )
    parser.add_argument("--query_file", default=None, help="Optional override for query JSON.")
    parser.add_argument("--baseline_tools_file", default=None, help="Optional override for baseline tools JSON.")
    parser.add_argument("--noisy_tools_file", default=None, help="Optional override for noisy tools JSON.")
    parser.add_argument(
        "--paths_set_catalog_file",
        default=None,
        help="Optional override for ground-truth path set catalog JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    evaluation_path = Path(args.output) if args.output else output_dir / "evaluation.json"
    report = evaluate_output(
        output_dir=output_dir,
        answer_judge=build_answer_judge(args.answer_judge),
        query_file=Path(args.query_file).resolve() if args.query_file else None,
        baseline_tools_file=Path(args.baseline_tools_file).resolve() if args.baseline_tools_file else None,
        noisy_tools_file=Path(args.noisy_tools_file).resolve() if args.noisy_tools_file else None,
        paths_set_catalog_file=Path(args.paths_set_catalog_file).resolve()
        if args.paths_set_catalog_file
        else None,
    )
    dump_json(evaluation_path, report)
    metrics = report["metrics"]
    print(json.dumps({
        "evaluation_path": str(evaluation_path),
        "evaluated_count": report["evaluated_count"],
        "malformed_result_file_count": len(report.get("malformed_result_files", [])),
        "accuracy": metrics["accuracy"],
        "correct_count": metrics["correct_count"],
        "mean_turn_count": (metrics.get("turns") or {}).get("mean_turn_count"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
