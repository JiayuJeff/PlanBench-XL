from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Optional

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - optional dependency
    _tqdm = None

from env.core.answer_judges import build_answer_judge, normalize_answer
from env.core.types import AgentState, QuerySpec, RunnerConfig
from env.core.utils import compute_signature, dump_json, load_json, normalize_runtime_value, now_utc_iso
from env.domains.executor import DomainToolExecutor
from env.events.controller import EventController
from env.events.noisy import NoisyToolAugmenter
from env.retriever.schema import ALLOWED_DOMAINS, format_retrieve_request_instructions, validate_retrieve_request
from env.retriever.semantic import SemanticRetriever
from env.runtime.llm import LLMClient
from env.runtime.parsing import extract_action, parse_retrieve_action, parse_tool_call_action
from env.runtime.prompts import PromptManager


class _NullProgressBar:
    def __init__(self, total: int, initial: int, desc: str) -> None:
        self.total = total
        self.n = initial
        self.desc = desc

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_description_str(self, desc: str) -> None:
        self.desc = desc

    def set_postfix_str(self, postfix: str) -> None:
        _ = postfix

    def close(self) -> None:
        return None


def _read_progress_position() -> int:
    raw_position = os.getenv("PWMT_PROGRESS_POSITION")
    if raw_position is None:
        return 0
    try:
        return max(int(raw_position), 0)
    except ValueError:
        return 0


def _build_progress_bar(
    *,
    total: int,
    initial: int,
    desc: str,
    unit: str,
) -> object:
    if _tqdm is None:
        return _NullProgressBar(total, initial, desc)
    return _tqdm(
        total=total,
        initial=initial,
        desc=desc,
        unit=unit,
        position=_read_progress_position(),
        dynamic_ncols=True,
        leave=True,
        disable=not sys.stderr.isatty(),
    )


def build_domain_context(
    prompt_manager: PromptManager,
    domain: str,
    domain_prompt_filename: str | None = None,
) -> str:
    domain_prompt_filename = domain_prompt_filename or f"domain_context_{domain}.txt"
    domain_prompt_path = prompt_manager.prompt_dir / domain_prompt_filename
    if domain_prompt_path.exists():
        return prompt_manager.render(domain_prompt_filename)
    return (
        f"You are working in the {domain} domain. Stay within this domain unless the query "
        "explicitly requires another domain."
    )


def format_max_steps(max_steps: float) -> str:
    if math.isinf(max_steps):
        return "inf"
    if float(max_steps).is_integer():
        return str(int(max_steps))
    return str(max_steps)


def load_queries(file_path: Path) -> list[QuerySpec]:
    data = load_json(file_path)
    return [QuerySpec(**item) for item in data]


def load_paths_set_catalog(file_path: Path) -> dict[str, list[dict[str, Any]]]:
    return load_json(file_path)


def load_all_baseline_tools(data_root: Path) -> dict[str, dict[str, Any]]:
    tool_registry: dict[str, dict[str, Any]] = {}
    for domain_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if domain_dir.name not in ALLOWED_DOMAINS:
            continue
        baseline_path = domain_dir / "baseline_tools.json"
        if not baseline_path.exists():
            continue
        for tool in load_json(baseline_path):
            tool_name = tool.get("name") or tool.get("tool_name")
            input_datatypes = tool.get("input_datatypes")
            output_datatype = tool.get("output_datatype")
            if (
                not isinstance(tool_name, str)
                or not isinstance(input_datatypes, list)
                or not all(isinstance(item, str) for item in input_datatypes)
                or not isinstance(output_datatype, str)
            ):
                continue
            enriched = dict(tool)
            enriched["name"] = tool_name
            enriched["tool_type"] = "baseline"
            enriched["domain"] = domain_dir.name
            enriched["io_type"] = f"{len(input_datatypes)}in1out"
            tool_registry[tool_name] = enriched
    return tool_registry


def load_all_datatypes(data_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    datatype_registry: dict[str, dict[str, dict[str, Any]]] = {}
    for domain_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if domain_dir.name not in ALLOWED_DOMAINS:
            continue
        datatypes_path = domain_dir / "datatypes.json"
        if not datatypes_path.exists():
            continue
        domain_datatypes: dict[str, dict[str, Any]] = {}
        for datatype in load_json(datatypes_path):
            name = datatype.get("name")
            if not isinstance(name, str):
                continue
            domain_datatypes[name] = dict(datatype)
        datatype_registry[domain_dir.name] = domain_datatypes
    return datatype_registry


def load_all_blocker_tools(data_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    tool_registry: dict[str, dict[str, Any]] = {}
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if domain_dir.name not in ALLOWED_DOMAINS:
            continue
        blocker_path = domain_dir / "blocker_tools.json"
        if not blocker_path.exists():
            continue
        domain_tools = []
        for tool in load_json(blocker_path):
            enriched = dict(tool)
            tool_name = enriched.get("name") or enriched.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            enriched["tool_type"] = tool.get("tool_type") or "blocker_misleading"
            enriched["domain"] = domain_dir.name
            input_datatypes = enriched.get("input_datatypes")
            if isinstance(input_datatypes, list) and all(isinstance(item, str) for item in input_datatypes):
                enriched["io_type"] = f"{len(input_datatypes)}in1out"
            enriched["name"] = tool_name
            tool_registry[tool_name] = enriched
            domain_tools.append(enriched)
        by_domain[domain_dir.name] = domain_tools
    return tool_registry, by_domain


def load_all_noisy_tools(
    data_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    tool_registry: dict[str, dict[str, Any]] = {}
    by_domain: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for domain_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if domain_dir.name not in ALLOWED_DOMAINS:
            continue
        noisy_path = domain_dir / "noisy_tools.json"
        if not noisy_path.exists():
            continue
        domain_tools_by_baseline: dict[str, list[dict[str, Any]]] = {}
        for tool in load_json(noisy_path):
            enriched = dict(tool)
            tool_name = enriched.get("name") or enriched.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            enriched["tool_type"] = tool.get("tool_type") or "noisy_misleading"
            enriched["domain"] = domain_dir.name
            input_datatypes = enriched.get("input_datatypes")
            if isinstance(input_datatypes, list) and all(isinstance(item, str) for item in input_datatypes):
                enriched["io_type"] = f"{len(input_datatypes)}in1out"
            enriched["name"] = tool_name
            tool_registry[tool_name] = enriched

            baseline_tool_name = enriched.get("baseline_tool_name")
            if isinstance(baseline_tool_name, str) and baseline_tool_name.strip():
                domain_tools_by_baseline.setdefault(baseline_tool_name, []).append(enriched)

        for siblings in domain_tools_by_baseline.values():
            siblings.sort(key=lambda tool: tool.get("name") or tool.get("tool_name") or "")
        by_domain[domain_dir.name] = domain_tools_by_baseline
    return tool_registry, by_domain


def load_noisy_tools_file(
    file_path: Path,
    *,
    domain: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    tool_registry: dict[str, dict[str, Any]] = {}
    domain_tools_by_baseline: dict[str, list[dict[str, Any]]] = {}
    if not file_path.exists():
        return tool_registry, {domain: domain_tools_by_baseline}

    for tool in load_json(file_path):
        enriched = dict(tool)
        tool_name = enriched.get("name") or enriched.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        enriched["tool_type"] = tool.get("tool_type") or "noisy_misleading"
        enriched["domain"] = domain
        input_datatypes = enriched.get("input_datatypes")
        if isinstance(input_datatypes, list) and all(isinstance(item, str) for item in input_datatypes):
            enriched["io_type"] = f"{len(input_datatypes)}in1out"
        enriched["name"] = tool_name
        tool_registry[tool_name] = enriched

        baseline_tool_name = enriched.get("baseline_tool_name")
        if isinstance(baseline_tool_name, str) and baseline_tool_name.strip():
            domain_tools_by_baseline.setdefault(baseline_tool_name, []).append(enriched)

    for siblings in domain_tools_by_baseline.values():
        siblings.sort(key=lambda tool: tool.get("name") or tool.get("tool_name") or "")
    return tool_registry, {domain: domain_tools_by_baseline}


def _serialize_value_store(values_by_datatype: dict[str, set[str]]) -> dict[str, list[str]]:
    return {
        datatype: sorted(values)
        for datatype, values in values_by_datatype.items()
        if values
    }


def _deserialize_value_store(payload: Any) -> dict[str, set[str]]:
    values_by_datatype: dict[str, set[str]] = {}
    if not isinstance(payload, dict):
        return values_by_datatype
    for datatype, raw_values in payload.items():
        if not isinstance(datatype, str):
            continue
        if not isinstance(raw_values, list):
            continue
        values_by_datatype[datatype] = {str(value) for value in raw_values}
    return values_by_datatype


def _serialize_source_store(
    sources_by_datatype: dict[str, dict[str, set[str]]],
) -> dict[str, dict[str, list[str]]]:
    return {
        datatype: {
            normalized_value: sorted(source_types)
            for normalized_value, source_types in value_sources.items()
            if source_types
        }
        for datatype, value_sources in sources_by_datatype.items()
        if value_sources
    }


def _deserialize_source_store(payload: Any) -> dict[str, dict[str, set[str]]]:
    sources_by_datatype: dict[str, dict[str, set[str]]] = {}
    if not isinstance(payload, dict):
        return sources_by_datatype
    for datatype, raw_value_sources in payload.items():
        if not isinstance(datatype, str) or not isinstance(raw_value_sources, dict):
            continue
        value_sources: dict[str, set[str]] = {}
        for normalized_value, raw_source_types in raw_value_sources.items():
            if not isinstance(normalized_value, str) or not isinstance(raw_source_types, list):
                continue
            value_sources[normalized_value] = {str(source_type) for source_type in raw_source_types}
        sources_by_datatype[datatype] = value_sources
    return sources_by_datatype


def _initial_trusted_values_for_query(query: QuerySpec) -> dict[str, set[str]]:
    trusted_values_by_datatype: dict[str, set[str]] = {}
    if not isinstance(query.input_values, dict):
        return trusted_values_by_datatype
    for datatype, value in query.input_values.items():
        if not isinstance(datatype, str):
            continue
        trusted_values_by_datatype.setdefault(datatype, set()).add(normalize_runtime_value(value))
    return trusted_values_by_datatype


def load_all_databases(data_root: Path) -> dict[str, dict[str, Any]]:
    databases: dict[str, dict[str, Any]] = {}
    for domain_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if domain_dir.name not in ALLOWED_DOMAINS:
            continue
        db_path = domain_dir / "database.json"
        if db_path.exists():
            databases[domain_dir.name] = load_json(db_path)
    return databases


class EnvRunner:
    def __init__(
        self,
        config: RunnerConfig,
        llm_client: LLMClient,
        retriever: SemanticRetriever,
        event_controller: EventController,
        noisy_tool_augmenter: NoisyToolAugmenter | None,
        tool_executor: DomainToolExecutor,
        prompt_manager: PromptManager,
        tool_registry: dict[str, dict[str, Any]],
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.retriever = retriever
        self.event_controller = event_controller
        self.noisy_tool_augmenter = noisy_tool_augmenter
        self.tool_executor = tool_executor
        self.prompt_manager = prompt_manager
        self.tool_registry = tool_registry
        self.logger = logging.getLogger("env_runner")
        self.progress_lock = Lock()
        self.use_tool_ids = bool(self.config.model.capabilities.get("tool_id_calling", False))
        self.answer_judge = build_answer_judge("normalized_contains")
        self.system_prompt = self.prompt_manager.render(
            self.config.prompt.system_runtime_prompt_file,
            retrieve_request_instructions=format_retrieve_request_instructions(),
            domain_context=build_domain_context(
                self.prompt_manager,
                self.config.domain,
                self.config.prompt.domain_context_file,
            ),
            max_steps=format_max_steps(self.config.runtime.max_steps),
            tool_call_format_instructions=self._build_tool_call_format_instructions(),
            tool_call_identifier_rules=self._build_tool_call_identifier_rules(),
        )

    def _progress_label(self) -> str:
        output_dir = self.config.output.output_dir
        model_label = self.config.model.model_name or output_dir.parent.name or self.config.run_id
        if output_dir.parent.name == model_label and output_dir.name:
            setting_label = output_dir.name
        else:
            setting_label = self.config.run_id or output_dir.name
        return f"{model_label} | {setting_label}"

    def _update_query_progress_bar(self, progress_bar: object, completed_queries: int, total_queries: int) -> None:
        remaining_queries = max(total_queries - completed_queries, 0)
        progress_bar.set_description_str(self._progress_label())
        progress_bar.set_postfix_str(f"remaining={remaining_queries}")

    def _build_initial_user_message(self, query: QuerySpec) -> str:
        return query.query_text

    def _build_tool_call_format_instructions(self) -> str:
        if not self.use_tool_ids:
            return (
                '<tool_call>\n'
                '{\n'
                '  "name": "...",\n'
                '  "arguments": { ... }\n'
                '}\n'
                '</tool_call>'
            )
        return (
            '<tool_call>\n'
            '{\n'
            '  "tool_id": "T1",\n'
            '  "arguments": { ... }\n'
            '}\n'
            '</tool_call>\n\n'
            'When a retrieve result includes `tool_id`, copy that exact `tool_id` into `<tool_call>`.\n'
            'Do not invent new tool_ids. Prefer `tool_id` over `name`.'
        )

    def _build_tool_call_identifier_rules(self) -> str:
        if not self.use_tool_ids:
            return "- Call tools by the exact `name` returned in a retrieval result, and place that value in `name` inside `<tool_call>`."
        return (
            "- Call tools by the exact `tool_id` returned in a retrieval result.\n"
            "- `name` is shown only for reading and reasoning; use `tool_id` when you execute."
        )

    def _upgrade_initial_history(
        self,
        history: list[dict[str, str]],
        query: QuerySpec,
    ) -> list[dict[str, str]]:
        if len(history) < 2:
            return history
        if history[0].get("role") != "system" or history[1].get("role") != "user":
            return history

        expected_user_message = self._build_initial_user_message(query)
        current_user_message = history[1].get("content", "")
        if current_user_message == expected_user_message:
            return history
        if not current_user_message.startswith(query.query_text):
            return history

        upgraded_history = list(history)
        upgraded_history[1] = {"role": "user", "content": expected_user_message}
        return upgraded_history

    def run(
        self,
        queries: list[QuerySpec],
        paths_set_catalog: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        _ = paths_set_catalog
        output_dir = self.config.output.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_dir = output_dir / "progress"
        query_progress_dir = progress_dir / "queries"
        query_progress_dir.mkdir(parents=True, exist_ok=True)

        config_signature = compute_signature(self.config.merged_config)
        metadata_path = output_dir / "metadata.json"
        index_path = progress_dir / "index.json"
        strict_resume_signature = bool(
            self.config.merged_config.get("strict_resume_config_signature", False)
        )
        unsafe_ignore_signature_mismatch = bool(
            self.config.merged_config.get("unsafe_ignore_config_signature_mismatch", False)
        )

        queries_by_id = {query.query_id: query for query in queries}
        if metadata_path.exists() or index_path.exists():
            metadata = load_json(metadata_path)
            index = load_json(index_path)
            old_signature = metadata.get("config_signature") or index.get("config_signature")
            if old_signature != config_signature:
                if strict_resume_signature and not unsafe_ignore_signature_mismatch:
                    raise RuntimeError(
                        "config_signature mismatch. Please use a new output_dir or delete existing "
                        "progress/metadata before rerunning."
                    )
                self.logger.warning(
                    "Config signature mismatch for %s; continuing resume because strict signature "
                    "validation is disabled.",
                    output_dir,
                )
            unresolved_exists = self._index_has_unresolved(index, output_dir)
            if unresolved_exists:
                if index["resume_attempts_used"] >= self.config.runtime.max_resume_attempts:
                    self.logger.warning(
                        "Max resume attempts reached for %s, but unresolved progress remains; continuing resume.",
                        output_dir,
                    )
                index["resume_attempts_used"] += 1
                dump_json(index_path, index)
            else:
                results = self._collect_existing_results(index, output_dir)
                self._write_result_jsonl(results, output_dir)
                return results
        else:
            metadata = {
                "domain": self.config.domain,
                "model_name": self.config.model.model_name,
                "config": self.config.merged_config,
                "config_signature": config_signature,
                "created_at": now_utc_iso(),
                "query_count": len(queries),
            }
            dump_json(metadata_path, metadata)
            index = self._initialize_progress_index(queries, config_signature)
            dump_json(index_path, index)
            for query in queries:
                dump_json(
                    query_progress_dir / f"{query.query_id}.json",
                    self._initial_query_progress(query),
                )

        queries_to_run = []
        for item in load_json(index_path)["queries"]:
            if not self._query_has_unresolved(item, output_dir):
                continue
            queries_to_run.append(queries_by_id[item["query_id"]])

        total_queries = len(queries)
        completed_queries = total_queries - len(queries_to_run)
        progress_bar = _build_progress_bar(
            total=total_queries,
            initial=completed_queries,
            desc=self._progress_label(),
            unit="query",
        )
        self._update_query_progress_bar(progress_bar, completed_queries, total_queries)

        try:
            with ThreadPoolExecutor(max_workers=self.config.runtime.max_concurrency) as executor:
                futures = {
                    executor.submit(
                        self.run_single_query,
                        query,
                        paths_set_catalog.get(query.task_id, []),
                    ): query.query_id
                    for query in queries_to_run
                }
                for future in as_completed(futures):
                    query_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:  # pragma: no cover - defensive logging path
                        self.logger.exception("Query %s encountered an unexpected error: %s", query_id, exc)
                    finally:
                        completed_queries += 1
                        progress_bar.update(1)
                        self._update_query_progress_bar(progress_bar, completed_queries, total_queries)
        finally:
            progress_bar.close()

        index = load_json(index_path)
        results = self._collect_existing_results(index, output_dir)
        if not self._index_has_unresolved(index, output_dir):
            self._write_result_jsonl(results, output_dir)
        return results

    def run_single_query(
        self,
        query: QuerySpec,
        paths_set_catalog: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        _ = paths_set_catalog
        progress_path = self.config.output.output_dir / "progress" / "queries" / f"{query.query_id}.json"
        progress = load_json(progress_path)

        history = self._upgrade_initial_history(progress["resume_checkpoint"]["effective_history"], query)
        if history != progress["resume_checkpoint"]["effective_history"]:
            progress["resume_checkpoint"] = {
                "effective_history": history,
                "raw_history_tail": history[-4:],
            }
            dump_json(progress_path, progress)
        state = self._load_agent_state(progress, query)
        turns = progress["turns"]
        next_turn_id = self._next_turn_id(progress)

        while state.total_step_count < self.config.runtime.max_steps:
            self._mark_turn_pending(query.query_id, next_turn_id, history, state)
            try:
                raw_response = self.llm_client.generate(history)
            except Exception as exc:
                self._mark_turn_error(query.query_id, next_turn_id, str(exc))
                return None

            state.total_step_count += 1
            action, content = extract_action(raw_response)

            if action is None:
                state.label_error_cnt += 1
                feedback = self.prompt_manager.render(
                    "error_label_format.txt",
                    current_attempt=state.label_error_cnt,
                    max_label_errors=self.config.runtime.max_label_errors,
                )
                state.steps_trace.append(
                    {"step_id": next_turn_id, "action": "others", "raw_output": raw_response}
                )
                history = self._append_turn_messages(history, raw_response, feedback)
                self._mark_turn_completed(
                    query.query_id,
                    next_turn_id,
                    raw_response,
                    {"action": "others", "parse_ok": False},
                    feedback,
                    state,
                    history,
                )
                if state.label_error_cnt >= self.config.runtime.max_label_errors:
                    return self._finalize_query(query, state, None, "exceeded_max_label_errors")
                next_turn_id += 1
                continue

            state.label_error_cnt = 0

            if action == "retrieve_tools":
                final = self._handle_retrieve(query, raw_response, content or "", next_turn_id, state, history)
                history = final["history"]
                if final["done"]:
                    return final["result"]
            elif action == "tool_call":
                final = self._handle_tool_call(query, raw_response, content or "", next_turn_id, state, history)
                history = final["history"]
                if final["done"]:
                    return final["result"]
            else:
                final = self._handle_final_answer(query, raw_response, content or "", next_turn_id, state, history)
                history = final["history"]
                self._mark_turn_completed(
                    query.query_id,
                    next_turn_id,
                    raw_response,
                    {"action": "final_answer", "parse_ok": True},
                    final["feedback"],
                    state,
                    history,
                )
                if final["done"]:
                    return final["result"]

            next_turn_id += 1

        return self._finalize_query(query, state, None, "exceeded_max_steps")

    def _handle_retrieve(
        self,
        query: QuerySpec,
        raw_response: str,
        content: str,
        step_id: int,
        state: AgentState,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        state.retrieval_attempt_count += 1
        parse_ok, request = parse_retrieve_action(content)
        if not parse_ok or request is None:
            state.retrieval_error_cnt += 1
            feedback = self.prompt_manager.render(
                "error_retrieve_json_format.txt",
                current_attempt=state.retrieval_error_cnt,
                max_retrieval_errors=self.config.runtime.max_retrieval_errors,
            )
            state.steps_trace.append(
                {"step_id": step_id, "action": "retrieve_tools", "parse_ok": False, "raw_output": raw_response}
            )
            history = self._append_turn_messages(history, raw_response, feedback)
            self._mark_turn_completed(
                query.query_id,
                step_id,
                raw_response,
                {"action": "retrieve_tools", "parse_ok": False},
                feedback,
                state,
                history,
            )
            if state.retrieval_error_cnt >= self.config.runtime.max_retrieval_errors:
                return {"done": True, "result": self._finalize_query(query, state, None, "exceeded_max_retrieval_errors"), "history": history}
            return {"done": False, "result": None, "history": history}

        request_ok, _, normalized_request = validate_retrieve_request(request, default_domain=self.config.domain)
        if not request_ok or normalized_request is None:
            state.retrieval_error_cnt += 1
            feedback = self.prompt_manager.render(
                "error_retrieve_semantic.txt",
                retrieve_request_instructions=format_retrieve_request_instructions(),
            )
            state.steps_trace.append(
                {
                    "step_id": step_id,
                    "action": "retrieve_tools",
                    "parse_ok": True,
                    "request": request,
                }
            )
            history = self._append_turn_messages(history, raw_response, feedback)
            self._mark_turn_completed(
                query.query_id,
                step_id,
                raw_response,
                {"action": "retrieve_tools", "parse_ok": True, "request": request},
                feedback,
                state,
                history,
            )
            if state.retrieval_error_cnt >= self.config.runtime.max_retrieval_errors:
                return {"done": True, "result": self._finalize_query(query, state, None, "exceeded_max_retrieval_errors"), "history": history}
            return {"done": False, "result": None, "history": history}

        state.retrieval_exec_count += 1
        state.retrieval_error_cnt = 0
        retrieval_result = self.retriever.retrieve_tools(normalized_request)
        if self.config.blocker.enable_block:
            primary_tools = self.event_controller.augment_retrieval_result(
                query.query_id,
                query.task_id,
                retrieval_result,
            )
        else:
            primary_tools = list(retrieval_result.tools)
        if self.noisy_tool_augmenter is not None:
            mixed_tools = self.noisy_tool_augmenter.augment_retrieval_result(
                primary_tools,
                domain=str(normalized_request.get("domain") or self.config.domain),
                enable_block=self.config.blocker.enable_block,
            )
        else:
            mixed_tools = primary_tools
        mixed_tools = self._shuffle_retrieved_tools(query, normalized_request, mixed_tools)
        mixed_tools = self._attach_tool_ids(mixed_tools, state)
        state.available_tool_names = [tool["name"] for tool in mixed_tools]
        state.available_tool_ids = [tool["tool_id"] for tool in mixed_tools if isinstance(tool.get("tool_id"), str)]
        state.discovered_tool_names.update(state.available_tool_names)

        trace_tools = [self._tool_trace_payload(tool) for tool in mixed_tools]
        feedback = self._format_retrieve_feedback(retrieval_result, mixed_tools)
        state.steps_trace.append(
            {
                "step_id": step_id,
                "action": "retrieve_tools",
                "parse_ok": True,
                "request": dict(normalized_request),
                "matched_information": retrieval_result.matched_information,
                "internal_retriever_note": retrieval_result.internal_retriever_note,
                "model_retriever_note": retrieval_result.model_retriever_note,
                "retrieval_result": {"tool_count": len(trace_tools), "tools": trace_tools},
            }
        )
        history = self._append_turn_messages(history, raw_response, feedback)
        self._mark_turn_completed(
            query.query_id,
            step_id,
            raw_response,
            {"action": "retrieve_tools", "parse_ok": True, "request": dict(normalized_request)},
            feedback,
            state,
            history,
        )
        return {"done": False, "result": None, "history": history}

    def _handle_tool_call(
        self,
        query: QuerySpec,
        raw_response: str,
        content: str,
        step_id: int,
        state: AgentState,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        state.tool_call_attempt_count += 1
        parse_ok, payload = parse_tool_call_action(content)
        if not parse_ok or payload is None:
            state.call_error_cnt += 1
            feedback = self.prompt_manager.render(
                "error_call_json_format.txt",
                current_attempt=state.call_error_cnt,
                max_call_errors=self.config.runtime.max_call_errors,
            )
            state.steps_trace.append(
                {"step_id": step_id, "action": "call_tool", "parse_ok": False, "raw_output": raw_response}
            )
            history = self._append_turn_messages(history, raw_response, feedback)
            self._mark_turn_completed(
                query.query_id,
                step_id,
                raw_response,
                {"action": "call_tool", "parse_ok": False},
                feedback,
                state,
                history,
            )
            if state.call_error_cnt >= self.config.runtime.max_call_errors:
                return {"done": True, "result": self._finalize_query(query, state, None, "exceeded_max_tool_call_errors"), "history": history}
            return {"done": False, "result": None, "history": history}

        requested_tool_id = payload.get("tool_id")
        tool_name = payload.get("name")
        if not isinstance(tool_name, str):
            tool_name = payload.get("tool_name")
        if isinstance(requested_tool_id, str):
            resolved_tool_name = state.tool_id_to_name.get(requested_tool_id)
            if resolved_tool_name is not None:
                tool_name = resolved_tool_name
        arguments = payload["arguments"]
        tool_spec = self.tool_registry.get(tool_name) if isinstance(tool_name, str) else None

        if tool_name not in state.discovered_tool_names or tool_spec is None:
            feedback = self.prompt_manager.render("error_call_tool_not_available.txt")
            state.call_error_cnt += 1
            return self._tool_call_error(
                query,
                step_id,
                raw_response,
                self._normalize_tool_call_payload(payload, tool_name, state),
                feedback,
                state,
                history,
            )

        surface_argument_to_datatype = tool_spec.get("surface_argument_to_datatype") or {}
        expected_argument_keys = list(surface_argument_to_datatype) if surface_argument_to_datatype else list(
            ((tool_spec.get("parameters") or {}).get("properties") or {}).keys()
        )
        if not expected_argument_keys:
            expected_argument_keys = list(tool_spec["input_datatypes"])

        expected_inputs = tool_spec["input_datatypes"]
        if set(arguments) != set(expected_argument_keys):
            state.call_error_cnt += 1
            feedback = self.prompt_manager.render(
                "error_call_wrong_inputs.txt",
                tool_name=tool_name,
                provided_arguments=list(arguments),
            )
            return self._tool_call_error(
                query,
                step_id,
                raw_response,
                self._normalize_tool_call_payload(payload, tool_name, state),
                feedback,
                state,
                history,
            )

        canonical_arguments = {
            surface_argument_to_datatype.get(argument_name, argument_name): argument_value
            for argument_name, argument_value in arguments.items()
        }
        untrusted_input_datatypes = [
            datatype
            for datatype in expected_inputs
            if self._is_untrusted_input_value(state, datatype, canonical_arguments.get(datatype))
        ]
        missing_inputs = [datatype for datatype in expected_inputs if datatype not in state.current_datatypes]
        if untrusted_input_datatypes or missing_inputs:
            state.call_error_cnt += 1
            feedback = self.prompt_manager.render("error_call_missing_inputs.txt")
            return self._tool_call_error(
                query,
                step_id,
                raw_response,
                self._normalize_tool_call_payload(payload, tool_name, state),
                feedback,
                state,
                history,
                trace_extra={
                    "internal_error_type": "untrusted_input_rejected"
                    if untrusted_input_datatypes
                    else "missing_required_input",
                    "untrusted_input_datatypes": untrusted_input_datatypes,
                    "missing_input_datatypes": missing_inputs,
                },
            )

        if self.event_controller.should_block_tool_call(query.query_id, query.task_id, step_id, tool_spec):
            state.call_error_cnt += 1
            feedback = self.prompt_manager.render("error_call_blocked.txt")
            return self._tool_call_error(
                query,
                step_id,
                raw_response,
                self._normalize_tool_call_payload(payload, tool_name, state),
                feedback,
                state,
                history,
                trace_extra={"blocked": True, "blocked_tool_name": tool_name},
            )

        execution_result = self.tool_executor.execute_tool(tool_spec, canonical_arguments)
        state.tool_call_exec_count += 1
        state.call_error_cnt = 0
        self._record_execution_output(state, execution_result)

        normalized_payload = self._normalize_tool_call_payload(payload, tool_name, state)
        feedback = self._format_tool_feedback(normalized_payload, execution_result)
        state.steps_trace.append(
            {
                "step_id": step_id,
                "action": "call_tool",
                "parse_ok": True,
                "request": normalized_payload,
                "tool_result": {
                    "success": execution_result.success,
                    "output_datatype": execution_result.output_datatype,
                    "output_value": execution_result.output_value,
                    "tool_type": execution_result.tool_type,
                    "output_provenance": execution_result.output_provenance,
                    "untrusted_source_type": execution_result.untrusted_source_type,
                },
            }
        )
        history = self._append_turn_messages(history, raw_response, feedback)
        self._mark_turn_completed(
            query.query_id,
            step_id,
            raw_response,
            {"action": "call_tool", "parse_ok": True, "request": normalized_payload},
            feedback,
            state,
            history,
        )
        return {"done": False, "result": None, "history": history}

    def _tool_call_error(
        self,
        query: QuerySpec,
        step_id: int,
        raw_response: str,
        payload: dict[str, Any],
        feedback: str,
        state: AgentState,
        history: list[dict[str, str]],
        trace_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        trace_entry = {
            "step_id": step_id,
            "action": "call_tool",
            "parse_ok": True,
            "request": payload,
        }
        if trace_extra:
            trace_entry.update(trace_extra)
        state.steps_trace.append(trace_entry)
        history = self._append_turn_messages(history, raw_response, feedback)
        self._mark_turn_completed(
            query.query_id,
            step_id,
            raw_response,
            {"action": "call_tool", "parse_ok": True, "request": payload},
            feedback,
            state,
            history,
        )
        if state.call_error_cnt >= self.config.runtime.max_call_errors:
            return {"done": True, "result": self._finalize_query(query, state, None, "exceeded_max_tool_call_errors"), "history": history}
        return {"done": False, "result": None, "history": history}

    def _handle_final_answer(
        self,
        query: QuerySpec,
        raw_response: str,
        content: str,
        step_id: int,
        state: AgentState,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        final_answer = content.strip()
        answer_correct = self.answer_judge.is_correct(query.correct_answer, final_answer)
        target_datatype_reached = query.target_datatype in state.current_datatypes
        matched_untrusted_source_types = self._matched_untrusted_sources_for_final_answer(
            state,
            query.target_datatype,
            final_answer,
        )
        datatype_check_passed = target_datatype_reached and not matched_untrusted_source_types
        trace_entry = {
            "step_id": step_id,
            "action": "final_answer",
            "parse_ok": True,
            "answer": final_answer,
            "answer_correct": answer_correct,
            "target_datatype_reached": target_datatype_reached,
            "matched_untrusted_source_types": sorted(matched_untrusted_source_types),
            "datatype_check_passed": datatype_check_passed,
        }

        if (
            self.config.runtime.enable_incorrect_final_answer_feedback
            and (not datatype_check_passed or not answer_correct)
            and state.incorrect_final_answer_feedback_count < self.config.runtime.incorrect_final_answer_feedback_budget
            and state.total_step_count < self.config.runtime.max_steps
        ):
            state.incorrect_final_answer_feedback_count += 1
            trace_entry["incorrect_final_answer_feedback_sent"] = True
            trace_entry["incorrect_final_answer_feedback_count"] = state.incorrect_final_answer_feedback_count
            state.steps_trace.append(trace_entry)
            feedback = self._format_incorrect_final_answer_feedback()
            history = self._append_turn_messages(history, raw_response, feedback)
            return {"done": False, "result": None, "history": history, "feedback": feedback}

        state.steps_trace.append(trace_entry)
        if datatype_check_passed and answer_correct:
            result = self._finalize_query(query, state, final_answer, None)
        else:
            if not datatype_check_passed:
                result = self._finalize_query(query, state, final_answer, "target_datatype_not_reached")
            else:
                result = self._finalize_query(query, state, final_answer, "final_answer_wrong")
        history = history + [{"role": "assistant", "content": raw_response}]
        return {"done": True, "result": result, "history": history, "feedback": None}

    def _format_incorrect_final_answer_feedback(self) -> str:
        return (
            "The final answer you just gave does not appear to be correct, and you have not obtained "
            "the target information yet. Please continue exploring if needed before giving another "
            "final answer."
        )

    def _format_retrieve_feedback(self, retrieval_result: Any, tools: list[dict[str, Any]]) -> str:
        payload = {
            "feedback_type": "retrieve_tools_result",
            "request": dict(getattr(retrieval_result, "request", {})),
            "retriever_note": getattr(retrieval_result, "model_retriever_note", None),
            "retrieval_result": {
                "tool_count": len(tools),
                "tools": [
                    {
                        **({"tool_id": tool["tool_id"]} if self.use_tool_ids and isinstance(tool.get("tool_id"), str) else {}),
                        "type": tool.get("type", "function"),
                        "name": tool["name"],
                        "description": tool["description"],
                        "strict": tool.get("strict", True),
                        "parameters": tool.get("parameters"),
                    }
                    for tool in tools
                ],
            },
        }
        return (
            "Retrieved tools for your request. Any tool retrieved earlier in this query remains callable.\n"
            + ("Use the returned `tool_id` values when you send `<tool_call>`.\n" if self.use_tool_ids else "")
            + "```json\n"
            + json.dumps(
            payload, ensure_ascii=False, indent=2
            )
            + "\n```"
        )

    def _format_tool_feedback(self, request_payload: dict[str, Any], execution_result: Any) -> str:
        tool_name = request_payload.get("name")
        tool_spec = self.tool_registry.get(tool_name) if isinstance(tool_name, str) else None
        if execution_result.tool_type == "baseline" and execution_result.output_value is None:
            return self._format_missing_output_feedback(request_payload, execution_result)

        payload = {
            "feedback_type": "tool_call_result",
            "request": request_payload,
            "tool_result": {
                "success": execution_result.success,
                "output_value": execution_result.output_value,
            },
        }
        return "Tool call completed.\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"

    def _format_missing_output_feedback(
        self,
        request_payload: dict[str, Any],
        execution_result: Any,
    ) -> str:
        arguments = request_payload.get("arguments", {})
        input_pairs = ", ".join(
            f"{name}={json.dumps(value, ensure_ascii=False)}" for name, value in arguments.items()
        )
        error_message = f"Error: no result exists for the provided input(s): {input_pairs}."
        payload = {
            "feedback_type": "tool_call_error",
            "request": request_payload,
            "error": error_message,
            "tool_result": {
                "success": False,
                "output_value": None,
            },
        }
        return error_message + "\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"

    def _tool_trace_payload(self, tool: dict[str, Any]) -> dict[str, Any]:
        return {
            **({"tool_id": tool["tool_id"]} if isinstance(tool.get("tool_id"), str) else {}),
            "name": tool["name"],
            "description": tool["description"],
            "input_datatypes": tool["input_datatypes"],
            "output_datatype": tool["output_datatype"],
            "tool_type": tool.get("tool_type", "baseline"),
        }

    def _is_untrusted_input_value(
        self,
        state: AgentState,
        datatype: str,
        value: Any,
    ) -> bool:
        normalized_value = normalize_runtime_value(value)
        untrusted_values = state.untrusted_values_by_datatype.get(datatype, set())
        trusted_values = state.trusted_values_by_datatype.get(datatype, set())
        return normalized_value in untrusted_values and normalized_value not in trusted_values

    def _record_execution_output(
        self,
        state: AgentState,
        execution_result: Any,
    ) -> None:
        if execution_result.output_value is None:
            return

        datatype = execution_result.output_datatype
        runtime_normalized_value = normalize_runtime_value(execution_result.output_value)
        answer_normalized_value = normalize_answer(execution_result.output_value)

        if execution_result.output_provenance == "trusted":
            state.current_datatypes.add(datatype)
            state.trusted_values_by_datatype.setdefault(datatype, set()).add(runtime_normalized_value)
            return

        if execution_result.output_provenance != "untrusted":
            return

        state.untrusted_values_by_datatype.setdefault(datatype, set()).add(runtime_normalized_value)
        source_type = execution_result.untrusted_source_type
        if source_type is not None and answer_normalized_value:
            state.untrusted_value_sources_by_datatype.setdefault(datatype, {}).setdefault(
                answer_normalized_value,
                set(),
            ).add(source_type)

    def _matched_untrusted_sources_for_final_answer(
        self,
        state: AgentState,
        target_datatype: str,
        final_answer: Any,
    ) -> set[str]:
        normalized_final_answer = normalize_answer(final_answer)
        if not normalized_final_answer:
            return set()

        matched_sources: set[str] = set()
        for normalized_value, source_types in state.untrusted_value_sources_by_datatype.get(target_datatype, {}).items():
            if normalized_value and normalized_value in normalized_final_answer:
                matched_sources.update(source_types)
        return matched_sources

    def _append_turn_messages(
        self,
        history: list[dict[str, str]],
        raw_response: str,
        feedback: str,
    ) -> list[dict[str, str]]:
        model_name = (self.config.model.model_name or "").lower()
        # feedback_role = "user" if "gemini" in model_name else "system"
        feedback_role = "user"
        feedback_prefix = "Environment feedback:\n" if feedback_role == "user" else "SYSTEM FEEDBACK:\n"
        return history + [
            {"role": "assistant", "content": raw_response},
            {"role": feedback_role, "content": feedback_prefix + feedback},
        ]

    def _finalize_query(
        self,
        query: QuerySpec,
        state: AgentState,
        final_answer: Any,
        failure_reason_detail: Optional[str],
    ) -> dict[str, Any]:
        result = {
            "query_id": query.query_id,
            "stop_reason": "normal",
            "status": "success" if failure_reason_detail is None else "failed",
            "final_answer": final_answer,
            "steps": state.total_step_count,
            "trace": state.steps_trace,
            "final_state": {
                "current_datatypes": sorted(state.current_datatypes),
                "available_tool_names": list(state.available_tool_names),
                "available_tool_ids": list(state.available_tool_ids),
                "discovered_tool_names": sorted(state.discovered_tool_names),
                "tool_name_to_id": dict(state.tool_name_to_id),
                "tool_id_to_name": dict(state.tool_id_to_name),
                "retrieval_attempt_count": state.retrieval_attempt_count,
                "retrieval_exec_count": state.retrieval_exec_count,
                "tool_call_attempt_count": state.tool_call_attempt_count,
                "tool_call_exec_count": state.tool_call_exec_count,
                "incorrect_final_answer_feedback_count": state.incorrect_final_answer_feedback_count,
                "trusted_values_by_datatype": _serialize_value_store(state.trusted_values_by_datatype),
                "untrusted_values_by_datatype": _serialize_value_store(state.untrusted_values_by_datatype),
                "untrusted_value_sources_by_datatype": _serialize_source_store(
                    state.untrusted_value_sources_by_datatype
                ),
                "has_final_answer": final_answer is not None,
            },
        }
        if failure_reason_detail is not None:
            result["failure_reason_detail"] = failure_reason_detail

        progress_path = self.config.output.output_dir / "progress" / "queries" / f"{query.query_id}.json"
        progress = load_json(progress_path)
        progress["status"] = "completed"
        progress["latest_state"] = self._agent_state_snapshot(state)
        progress["final_result"] = result
        dump_json(progress_path, progress)

        with self.progress_lock:
            index_path = self.config.output.output_dir / "progress" / "index.json"
            index = load_json(index_path)
            for item in index["queries"]:
                if item["query_id"] == query.query_id:
                    item["status"] = "completed"
                    item["final_result_ready"] = True
                    item["unresolved_turn_id"] = None
                    break
            dump_json(index_path, index)
        return result

    def _initial_query_progress(self, query: QuerySpec) -> dict[str, Any]:
        initial_user_message = self._build_initial_user_message(query)
        history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": initial_user_message},
        ]
        return {
            "query_id": query.query_id,
            "status": "pending",
            "truncate_history": self.config.runtime.truncate_history,
            "latest_state": {
                "current_datatypes": query.input_datatypes,
                "trusted_values_by_datatype": _serialize_value_store(_initial_trusted_values_for_query(query)),
                "untrusted_values_by_datatype": {},
                "untrusted_value_sources_by_datatype": {},
                "available_tool_names": [],
                "available_tool_ids": [],
                "discovered_tool_names": [],
                "tool_name_to_id": {},
                "tool_id_to_name": {},
                "next_tool_id_num": 1,
                "step_count": 0,
                "retrieval_attempt_count": 0,
                "retrieval_exec_count": 0,
                "tool_call_attempt_count": 0,
                "tool_call_exec_count": 0,
                "incorrect_final_answer_feedback_count": 0,
            },
            "resume_checkpoint": {
                "effective_history": history,
                "raw_history_tail": history[-4:],
            },
            "turns": [],
            "final_result": None,
        }

    def _load_agent_state(self, progress: dict[str, Any], query: QuerySpec) -> AgentState:
        latest = progress.get("latest_state") or {}
        final_result = progress.get("final_result") or {}
        final_state = final_result.get("final_state") or {}
        steps_trace = list(latest.get("steps_trace", final_result.get("trace", [])))
        discovered_tool_names = latest.get("discovered_tool_names")
        tool_name_to_id = dict(latest.get("tool_name_to_id", {}))
        tool_id_to_name = dict(latest.get("tool_id_to_name", {}))
        if discovered_tool_names is None:
            discovered_tool_names = list(latest.get("available_tool_names", []))
            for step in steps_trace:
                if step.get("action") != "retrieve_tools":
                    continue
                tools = ((step.get("retrieval_result") or {}).get("tools") or [])
                discovered_tool_names.extend(
                    str(tool.get("name") or tool.get("tool_name"))
                    for tool in tools
                    if isinstance(tool.get("name") or tool.get("tool_name"), str)
                )
        if not tool_name_to_id or not tool_id_to_name:
            for step in steps_trace:
                if step.get("action") != "retrieve_tools":
                    continue
                tools = ((step.get("retrieval_result") or {}).get("tools") or [])
                for tool in tools:
                    tool_name = tool.get("name") or tool.get("tool_name")
                    tool_id = tool.get("tool_id")
                    if isinstance(tool_name, str) and isinstance(tool_id, str):
                        tool_name_to_id.setdefault(tool_name, tool_id)
                        tool_id_to_name.setdefault(tool_id, tool_name)
        trusted_values_by_datatype = _deserialize_value_store(
            latest.get("trusted_values_by_datatype", final_state.get("trusted_values_by_datatype"))
        )
        if not trusted_values_by_datatype:
            trusted_values_by_datatype = _initial_trusted_values_for_query(query)
        return AgentState(
            current_datatypes=set(latest.get("current_datatypes", query.input_datatypes)),
            trusted_values_by_datatype=trusted_values_by_datatype,
            untrusted_values_by_datatype=_deserialize_value_store(
                latest.get("untrusted_values_by_datatype", final_state.get("untrusted_values_by_datatype"))
            ),
            untrusted_value_sources_by_datatype=_deserialize_source_store(
                latest.get(
                    "untrusted_value_sources_by_datatype",
                    final_state.get("untrusted_value_sources_by_datatype"),
                )
            ),
            available_tool_names=list(latest.get("available_tool_names", [])),
            available_tool_ids=list(latest.get("available_tool_ids", [])),
            discovered_tool_names=set(discovered_tool_names),
            tool_name_to_id=tool_name_to_id,
            tool_id_to_name=tool_id_to_name,
            next_tool_id_num=int(latest.get("next_tool_id_num", 1)),
            total_step_count=latest.get("step_count", 0),
            retrieval_attempt_count=latest.get("retrieval_attempt_count", 0),
            retrieval_exec_count=latest.get("retrieval_exec_count", 0),
            tool_call_attempt_count=latest.get("tool_call_attempt_count", 0),
            tool_call_exec_count=latest.get("tool_call_exec_count", 0),
            label_error_cnt=latest.get("label_error_cnt", 0),
            retrieval_error_cnt=latest.get("retrieval_error_cnt", 0),
            call_error_cnt=latest.get("call_error_cnt", 0),
            incorrect_final_answer_feedback_count=latest.get("incorrect_final_answer_feedback_count", 0),
            steps_trace=steps_trace,
        )

    def _agent_state_snapshot(self, state: AgentState) -> dict[str, Any]:
        return {
            "current_datatypes": sorted(state.current_datatypes),
            "trusted_values_by_datatype": _serialize_value_store(state.trusted_values_by_datatype),
            "untrusted_values_by_datatype": _serialize_value_store(state.untrusted_values_by_datatype),
            "untrusted_value_sources_by_datatype": _serialize_source_store(
                state.untrusted_value_sources_by_datatype
            ),
            "available_tool_names": list(state.available_tool_names),
            "available_tool_ids": list(state.available_tool_ids),
            "discovered_tool_names": sorted(state.discovered_tool_names),
            "tool_name_to_id": dict(state.tool_name_to_id),
            "tool_id_to_name": dict(state.tool_id_to_name),
            "next_tool_id_num": state.next_tool_id_num,
            "step_count": state.total_step_count,
            "retrieval_attempt_count": state.retrieval_attempt_count,
            "retrieval_exec_count": state.retrieval_exec_count,
            "tool_call_attempt_count": state.tool_call_attempt_count,
            "tool_call_exec_count": state.tool_call_exec_count,
            "label_error_cnt": state.label_error_cnt,
            "retrieval_error_cnt": state.retrieval_error_cnt,
            "call_error_cnt": state.call_error_cnt,
            "incorrect_final_answer_feedback_count": state.incorrect_final_answer_feedback_count,
            "steps_trace": state.steps_trace,
        }

    def _attach_tool_ids(self, tools: list[dict[str, Any]], state: AgentState) -> list[dict[str, Any]]:
        if not self.use_tool_ids:
            return tools

        enriched_tools: list[dict[str, Any]] = []
        for tool in tools:
            tool_name = tool["name"]
            tool_id = state.tool_name_to_id.get(tool_name)
            if tool_id is None:
                tool_id = f"T{state.next_tool_id_num}"
                state.next_tool_id_num += 1
                state.tool_name_to_id[tool_name] = tool_id
                state.tool_id_to_name[tool_id] = tool_name
            enriched = dict(tool)
            enriched["tool_id"] = tool_id
            enriched_tools.append(enriched)
        return enriched_tools

    def _shuffle_retrieved_tools(
        self,
        query: QuerySpec,
        normalized_request: dict[str, Any],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if len(tools) <= 1:
            return tools

        tool_fingerprint = [
            {
                "name": tool.get("name") or tool.get("tool_name"),
                "tool_type": tool.get("tool_type", "baseline"),
            }
            for tool in tools
        ]
        seed_payload = {
            "global_seed": self.config.blocker.seed,
            "task_id": query.task_id,
            "query_id": query.query_id,
            "request": normalized_request,
            "tools": tool_fingerprint,
        }
        seed_material = json.dumps(seed_payload, ensure_ascii=False, sort_keys=True, default=str)
        seed = int.from_bytes(
            hashlib.blake2b(seed_material.encode("utf-8"), digest_size=16).digest(),
            "big",
        )
        shuffled = list(tools)
        random.Random(seed).shuffle(shuffled)
        return shuffled

    def _normalize_tool_call_payload(
        self,
        payload: dict[str, Any],
        resolved_tool_name: str | None,
        state: AgentState,
    ) -> dict[str, Any]:
        normalized = dict(payload)
        normalized.pop("tool_name", None)
        if resolved_tool_name is not None:
            normalized["name"] = resolved_tool_name
            tool_id = state.tool_name_to_id.get(resolved_tool_name)
            if tool_id is not None:
                normalized["tool_id"] = tool_id
        return normalized

    def _next_turn_id(self, progress: dict[str, Any]) -> int:
        for turn in progress.get("turns", []):
            if turn["status"] in {"pending", "error"}:
                return turn["turn_id"]
        return len(progress.get("turns", [])) + 1

    def _mark_turn_pending(
        self,
        query_id: str,
        turn_id: int,
        history: list[dict[str, str]],
        state: AgentState,
    ) -> None:
        progress_path = self.config.output.output_dir / "progress" / "queries" / f"{query_id}.json"
        progress = load_json(progress_path)
        turn_entry = {
            "turn_id": turn_id,
            "status": "pending",
            "llm_raw_response": None,
            "parsed_action": None,
            "feedback_to_model": None,
            "state_after_turn": None,
            "error": None,
        }
        turns = [turn for turn in progress["turns"] if turn["turn_id"] != turn_id]
        turns.append(turn_entry)
        turns.sort(key=lambda item: item["turn_id"])
        progress["turns"] = turns
        progress["resume_checkpoint"] = {
            "effective_history": history,
            "raw_history_tail": history[-4:],
        }
        progress["latest_state"] = self._agent_state_snapshot(state)
        dump_json(progress_path, progress)

        with self.progress_lock:
            index_path = self.config.output.output_dir / "progress" / "index.json"
            index = load_json(index_path)
            for item in index["queries"]:
                if item["query_id"] == query_id:
                    item["status"] = "pending"
                    item["last_turn_id"] = max(item.get("last_turn_id", 0), turn_id)
                    item["unresolved_turn_id"] = turn_id
                    item["final_result_ready"] = False
                    break
            dump_json(index_path, index)

    def _mark_turn_completed(
        self,
        query_id: str,
        turn_id: int,
        raw_response: str,
        parsed_action: dict[str, Any],
        feedback_to_model: Optional[str],
        state: AgentState,
        history: list[dict[str, str]],
    ) -> None:
        progress_path = self.config.output.output_dir / "progress" / "queries" / f"{query_id}.json"
        progress = load_json(progress_path)
        for turn in progress["turns"]:
            if turn["turn_id"] == turn_id:
                turn["status"] = "completed"
                turn["llm_raw_response"] = raw_response if self.config.output.save_raw_llm_response else None
                turn["parsed_action"] = parsed_action
                turn["feedback_to_model"] = feedback_to_model
                turn["state_after_turn"] = self._agent_state_snapshot(state)
                turn["error"] = None
                break
        progress["latest_state"] = self._agent_state_snapshot(state)
        progress["resume_checkpoint"] = {
            "effective_history": history,
            "raw_history_tail": history[-4:],
        }
        dump_json(progress_path, progress)

        with self.progress_lock:
            index_path = self.config.output.output_dir / "progress" / "index.json"
            index = load_json(index_path)
            for item in index["queries"]:
                if item["query_id"] == query_id:
                    item["last_turn_id"] = turn_id
                    item["unresolved_turn_id"] = None
                    break
            dump_json(index_path, index)

    def _mark_turn_error(self, query_id: str, turn_id: int, error_message: str) -> None:
        progress_path = self.config.output.output_dir / "progress" / "queries" / f"{query_id}.json"
        progress = load_json(progress_path)
        for turn in progress["turns"]:
            if turn["turn_id"] == turn_id:
                turn["status"] = "error"
                turn["error"] = {"type": "runtime_error", "message": error_message}
                break
        dump_json(progress_path, progress)

        with self.progress_lock:
            index_path = self.config.output.output_dir / "progress" / "index.json"
            index = load_json(index_path)
            for item in index["queries"]:
                if item["query_id"] == query_id:
                    item["status"] = "pending"
                    item["unresolved_turn_id"] = turn_id
                    break
            dump_json(index_path, index)

    def _initialize_progress_index(self, queries: list[QuerySpec], config_signature: str) -> dict[str, Any]:
        return {
            "config_signature": config_signature,
            "resume_attempts_used": 0,
            "max_resume_attempts": self.config.runtime.max_resume_attempts,
            "queries": [
                {
                    "query_id": query.query_id,
                    "status": "pending",
                    "last_turn_id": 0,
                    "unresolved_turn_id": None,
                    "query_progress_path": f"progress/queries/{query.query_id}.json",
                    "final_result_ready": False,
                }
                for query in queries
            ],
        }

    def _query_has_unresolved(self, item: dict[str, Any], output_dir: Path) -> bool:
        if item.get("status") != "completed":
            return True
        if not item.get("final_result_ready", False):
            return True

        progress_rel = item.get("query_progress_path")
        if not progress_rel:
            return True

        progress_path = output_dir / progress_rel
        if not progress_path.exists():
            return True

        progress = load_json(progress_path)
        if progress.get("status") != "completed":
            return True
        if progress.get("final_result") is None:
            return True
        if any(turn.get("status") in {"pending", "error"} for turn in progress.get("turns", [])):
            return True

        return False

    def _index_has_unresolved(self, index: dict[str, Any], output_dir: Path) -> bool:
        return any(self._query_has_unresolved(item, output_dir) for item in index.get("queries", []))

    def _collect_existing_results(self, index: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in index["queries"]:
            progress_path = output_dir / item["query_progress_path"]
            progress = load_json(progress_path)
            final_result = progress.get("final_result")
            if final_result is not None:
                results.append(final_result)
        return results

    def _write_result_jsonl(self, results: list[dict[str, Any]], output_dir: Path) -> None:
        result_path = output_dir / "result.jsonl"
        with result_path.open("w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False))
                f.write("\n")
