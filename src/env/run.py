from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import yaml


def _bootstrap_path() -> Path:
    current_file = Path(__file__).resolve()
    src_root = current_file.parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return src_root


_SRC_ROOT = _bootstrap_path()

from env.core.config import load_config
from env.core.sampling import sample_sequence
from env.domains.executor import DomainToolExecutor
from env.events.blocker import generate_blocker_replacements_by_task
from env.events.controller import EventController
from env.events.noisy import NoisyToolAugmenter
from env.retriever.semantic import SemanticRetriever
from env.runtime.llm import LLMClient
from env.runtime.prompts import PromptManager
from env.runtime.runner import (
    EnvRunner,
    load_all_baseline_tools,
    load_all_blocker_tools,
    load_all_databases,
    load_all_datatypes,
    load_noisy_tools_file,
    load_paths_set_catalog,
    load_queries,
)


def _parse_cli_override(raw_entry: str) -> tuple[str, object]:
    if "=" not in raw_entry:
        raise ValueError(
            f"Unrecognized argument '{raw_entry}'. CLI overrides must use KEY=VALUE syntax, "
            "for example runtime.max_steps=120."
        )
    key, raw_value = raw_entry.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid CLI override '{raw_entry}': missing key before '='.")
    return key, yaml.safe_load(raw_value)


def _collect_cli_overrides(raw_entries: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for raw_entry in raw_entries:
        key, value = _parse_cli_override(raw_entry)
        overrides[key] = value
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tool-planning runtime.")
    parser.add_argument("--run_config", required=True, help="Path to run yaml")
    parser.add_argument(
        "--resume-validation",
        choices=("strict", "warn-signature", "ignore-signature"),
        default="warn-signature",
        help=(
            "Control how strictly existing progress/metadata are validated before resuming. "
            "'warn-signature' continues resume and logs a warning when the saved config signature "
            "no longer matches; 'strict' blocks resume on mismatch. 'ignore-signature' is kept "
            "as a backward-compatible alias of 'warn-signature'."
        ),
    )
    parser.add_argument(
        "--set",
        dest="override_entries",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override any merged config field from the command line, for example "
            "--set runtime.max_steps=120 or --set output.output_dir=new_data/tmp/run."
        ),
    )

    args, unknown = parser.parse_known_args()
    raw_override_entries = list(args.override_entries)
    raw_override_entries.extend(unknown)
    try:
        args.cli_overrides = _collect_cli_overrides(raw_override_entries)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    run_config_path = Path(args.run_config).resolve()
    config = load_config(run_config_path, cli_overrides=args.cli_overrides)
    if args.resume_validation == "strict":
        config.merged_config["strict_resume_config_signature"] = True

    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_root = _SRC_ROOT / "data"
    baseline_registry = load_all_baseline_tools(data_root)
    datatype_registry = load_all_datatypes(data_root)
    noisy_registry, noisy_tools_by_domain = load_noisy_tools_file(
        config.data.noisy_tools_file,
        domain=config.domain,
    )
    blocker_registry, blocker_by_domain = load_all_blocker_tools(data_root)
    tool_registry = {**baseline_registry, **noisy_registry, **blocker_registry}
    databases = load_all_databases(data_root)

    all_queries = load_queries(config.data.query_file)
    queries = sample_sequence(all_queries, config.query_sample.size, config.query_sample.seed)
    paths_set_catalog = load_paths_set_catalog(config.data.paths_set_catalog_file)
    blocker_replacements_by_task: dict[str, dict[str, list[dict[str, Any]]]] | None = None
    if config.blocker.enable_block:
        blocker_replacements_by_task = generate_blocker_replacements_by_task(
            paths_set_catalog=paths_set_catalog,
            baseline_tools_path=config.data.baseline_tools_file,
            tasks_path=config.data.task_file,
            selection_mode=config.blocker.selection_mode,
            block_n_per_task=config.blocker.block_n_per_task,
            target_remaining_paths=config.blocker.target_remaining_paths,
            target_remaining_ratio=config.blocker.target_remaining_ratio,
            remaining_tolerance=config.blocker.remaining_tolerance,
            min_remaining_paths=config.blocker.min_remaining_paths,
            remaining_path_length_objective=config.blocker.remaining_path_length_objective,
            blocking_edge_count_objective=config.blocker.blocking_edge_count_objective,
            seed=config.blocker.seed,
            noise_mode=config.blocker.noise_mode,
            fixed_noise_type=config.blocker.fixed_noise_type,
            fixed_noise_types=config.blocker.fixed_noise_types,
            multi_noise_count=config.blocker.multi_noise_count,
            max_combo_candidates=config.blocker.max_combo_candidates,
            max_cover_size=config.blocker.max_cover_size,
        )

    llm_client = LLMClient(config.model)
    retriever = SemanticRetriever(baseline_registry, datatype_registry, config.retriever.embedding_model)
    event_controller = EventController(blocker_by_domain, config.blocker.enable_block, blocker_replacements_by_task)
    noisy_tool_augmenter = NoisyToolAugmenter(
        noisy_tools_by_domain,
        config.noise.mode,
        max_total_tools=config.noise.max_total_tools,
    )
    tool_executor = DomainToolExecutor(databases)
    prompt_manager = PromptManager(config.prompt.prompt_dir)

    runner = EnvRunner(
        config=config,
        llm_client=llm_client,
        retriever=retriever,
        event_controller=event_controller,
        noisy_tool_augmenter=noisy_tool_augmenter,
        tool_executor=tool_executor,
        prompt_manager=prompt_manager,
        tool_registry=tool_registry,
    )
    runner.run(queries, paths_set_catalog)


if __name__ == "__main__":
    main()
