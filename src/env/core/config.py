from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from env.core.types import (
    AuthConfig,
    BlockerConfig,
    DataConfig,
    LoggingConfig,
    ModelProfile,
    NoiseConfig,
    OutputConfig,
    PromptConfig,
    QuerySampleConfig,
    RequestConfig,
    RetrieverConfig,
    RunnerConfig,
    RuntimeConfig,
)
from env.core.utils import deep_merge, load_yaml


def _resolve_relative(project_root: Path, raw_path: str) -> Path:
    return (project_root / raw_path).resolve()


def _filter_dataclass_kwargs(cls: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    return {key: value for key, value in payload.items() if key in allowed}


def _discover_project_root(run_config_path: Path) -> Path:
    candidates = [run_config_path.parent, *run_config_path.parents]
    for candidate in candidates:
        if (candidate / "src" / "env" / "config" / "model_registry.yaml").exists():
            return candidate
    raise ValueError(
        f"Unable to locate project root from run_config path: {run_config_path}. "
        "Expected to find src/env/config/model_registry.yaml in one of its parent directories."
    )


def _materialize_cli_overrides(cli_overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not cli_overrides:
        return {}

    nested: dict[str, Any] = {}
    for raw_path, value in cli_overrides.items():
        path = raw_path.strip()
        if not path:
            raise ValueError("CLI override path cannot be empty")

        current = nested
        parts = path.split(".")
        for raw_part in parts[:-1]:
            part = raw_part.strip()
            if not part:
                raise ValueError(f"Invalid CLI override path: {raw_path}")
            existing = current.get(part)
            if existing is None:
                child: dict[str, Any] = {}
                current[part] = child
                current = child
                continue
            if not isinstance(existing, dict):
                raise ValueError(f"CLI override path conflict at '{part}' in {raw_path}")
            current = existing

        leaf = parts[-1].strip()
        if not leaf:
            raise ValueError(f"Invalid CLI override path: {raw_path}")
        current[leaf] = value

    return nested


def load_config(
    run_config_path: Path,
    model_registry_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> RunnerConfig:
    run_config_path = run_config_path.resolve()
    project_root = _discover_project_root(run_config_path)
    env_root = project_root / "src" / "env"
    cli_override_tree = _materialize_cli_overrides(cli_overrides)

    if model_registry_path is None:
        model_registry_path = env_root / "config" / "model_registry.yaml"
    model_registry_path = model_registry_path.resolve()

    run_yaml = load_yaml(run_config_path)
    if cli_override_tree:
        run_yaml = deep_merge(run_yaml, cli_override_tree)
    frozen_merged = run_yaml.get("frozen_merged_config")
    if frozen_merged is not None:
        if not isinstance(frozen_merged, dict):
            raise ValueError("run yaml 中的 frozen_merged_config 必须是 mapping")
        merged = frozen_merged
        if "base_config" not in merged:
            raise ValueError("frozen_merged_config 缺少 base_config")
        if not isinstance(merged.get("model"), dict):
            raise ValueError("frozen_merged_config 缺少完整 model 配置")
        base_config_path = _resolve_relative(project_root, merged["base_config"])
        model_yaml = merged["model"]
    else:
        base_config_path = _resolve_relative(project_root, run_yaml["base_config"])
        base_yaml = load_yaml(base_config_path)

        registry_yaml = load_yaml(model_registry_path)
        model_rel_path = registry_yaml[run_yaml["model_ref"]]
        model_yaml_path = (model_registry_path.parent / model_rel_path).resolve()
        model_yaml = load_yaml(model_yaml_path)

        merged = deep_merge(base_yaml, {"model": model_yaml})
        merged = deep_merge(merged, run_yaml)
        merged = deep_merge(merged, run_yaml.get("overrides", {}))

    if cli_override_tree:
        merged = deep_merge(merged, cli_override_tree)

    raw_output_dir = Path(merged["output"]["output_dir"])
    if raw_output_dir.is_absolute():
        raise ValueError("run yaml 中的 output.output_dir 必须是相对路径")

    output_root = Path(merged["output"]["root_dir"])
    final_output_dir = (project_root / output_root / raw_output_dir).resolve()
    merged.setdefault("data", {})
    merged["data"].setdefault("noisy_tools_file", f"src/data/{merged['domain']}/noisy_tools.json")

    data = DataConfig(
        task_file=_resolve_relative(project_root, merged["data"]["task_file"]),
        query_file=_resolve_relative(project_root, merged["data"]["query_file"]),
        baseline_tools_file=_resolve_relative(project_root, merged["data"]["baseline_tools_file"]),
        noisy_tools_file=_resolve_relative(project_root, merged["data"]["noisy_tools_file"]),
        blocker_tools_file=_resolve_relative(project_root, merged["data"]["blocker_tools_file"]),
        datatypes_file=_resolve_relative(project_root, merged["data"]["datatypes_file"]),
        database_file=_resolve_relative(project_root, merged["data"]["database_file"]),
        paths_set_catalog_file=_resolve_relative(
            project_root, merged["data"]["paths_set_catalog_file"]
        ),
    )

    model_payload = merged.get("model", model_yaml)
    if not isinstance(model_payload, dict):
        raise ValueError("merged model config 必须是 mapping")
    model = ModelProfile(
        model_id=model_payload["model_id"],
        provider=model_payload["provider"],
        api_style=model_payload["api_style"],
        model_name=model_payload["model_name"],
        auth=AuthConfig(**_filter_dataclass_kwargs(AuthConfig, model_payload.get("auth", {}))),
        request=RequestConfig(**_filter_dataclass_kwargs(RequestConfig, model_payload.get("request", {}))),
        limits=model_payload.get("limits", {}),
        capabilities=model_payload.get("capabilities", {}),
    )

    blocker_yaml = dict(merged.get("blocker", {}))
    if blocker_yaml.get("enable_block"):
        selection_mode = blocker_yaml.get("selection_mode", "target_remaining_ratio")
        if selection_mode not in {"exact_blocked_paths", "target_remaining_paths", "target_remaining_ratio"}:
            raise ValueError(f"Unsupported blocker.selection_mode: {selection_mode}")
        if selection_mode == "exact_blocked_paths" and blocker_yaml.get("block_n_per_task") is None:
            raise ValueError("blocker.block_n_per_task is required when blocker.selection_mode=exact_blocked_paths")
        if selection_mode == "target_remaining_paths" and blocker_yaml.get("target_remaining_paths") is None:
            raise ValueError("blocker.target_remaining_paths is required when blocker.selection_mode=target_remaining_paths")
        if selection_mode == "target_remaining_ratio":
            ratio = blocker_yaml.get("target_remaining_ratio")
            if ratio is None:
                raise ValueError("blocker.target_remaining_ratio is required when blocker.selection_mode=target_remaining_ratio")
            if not (0.0 <= float(ratio) <= 1.0):
                raise ValueError("blocker.target_remaining_ratio must be between 0 and 1")
        remaining_path_length_objective = blocker_yaml.get("remaining_path_length_objective", "none")
        if remaining_path_length_objective not in {"none", "maximize", "minimize", "random", "random_middle"}:
            raise ValueError(
                "blocker.remaining_path_length_objective must be one of: none, maximize, minimize, random, random_middle"
            )
        blocking_edge_count_objective = blocker_yaml.get("blocking_edge_count_objective", "none")
        if blocking_edge_count_objective not in {"none", "minimize"}:
            raise ValueError("blocker.blocking_edge_count_objective must be one of: none, minimize")
        noise_mode = blocker_yaml.get("noise_mode")
        if noise_mode not in {"random", "fixed", "random_multi", "fixed_multi"}:
            raise ValueError("blocker.noise_mode must be one of: random, fixed, random_multi, fixed_multi")
        if noise_mode == "fixed" and blocker_yaml.get("fixed_noise_type") is None:
            raise ValueError("blocker.fixed_noise_type is required when blocker.noise_mode=fixed")
        if noise_mode == "fixed_multi":
            fixed_noise_types = blocker_yaml.get("fixed_noise_types")
            if not isinstance(fixed_noise_types, list) or not fixed_noise_types:
                raise ValueError("blocker.fixed_noise_types must be a non-empty list when blocker.noise_mode=fixed_multi")
        if noise_mode == "random_multi":
            multi_noise_count = blocker_yaml.get("multi_noise_count")
            if multi_noise_count is None:
                raise ValueError("blocker.multi_noise_count is required when blocker.noise_mode=random_multi")
            if int(multi_noise_count) <= 0:
                raise ValueError("blocker.multi_noise_count must be a positive integer")
        if noise_mode in {"random", "fixed"} and blocker_yaml.get("multi_noise_count") not in (None, 1):
            raise ValueError("blocker.multi_noise_count is only supported for blocker.noise_mode=random_multi")
        if noise_mode != "fixed_multi" and blocker_yaml.get("fixed_noise_types") is not None:
            raise ValueError("blocker.fixed_noise_types is only supported for blocker.noise_mode=fixed_multi")

    query_sample_yaml = dict(merged.get("query_sample", {}))
    query_sample_size = query_sample_yaml.get("size")
    if query_sample_size is not None and int(query_sample_size) < 0:
        raise ValueError("query_sample.size must be null or non-negative")

    noise_yaml = dict(merged.get("noise", {}))
    noise_max_total_tools = noise_yaml.get("max_total_tools")
    if noise_max_total_tools is not None and int(noise_max_total_tools) < 0:
        raise ValueError("noise.max_total_tools must be null or non-negative")

    prompt_yaml = dict(merged.get("prompt", {}))
    raw_prompt_dir = prompt_yaml.get("prompt_dir", "src/env/prompt")
    prompt_dir = _resolve_relative(project_root, raw_prompt_dir)
    domain_context_file = prompt_yaml.get("domain_context_file")
    if domain_context_file is not None and not isinstance(domain_context_file, str):
        raise ValueError("prompt.domain_context_file must be a string when provided")

    return RunnerConfig(
        run_id=merged["run_id"],
        domain=merged["domain"],
        base_config_path=base_config_path,
        run_config_path=run_config_path,
        data=data,
        model=model,
        retriever=RetrieverConfig(**_filter_dataclass_kwargs(RetrieverConfig, merged.get("retriever", {}))),
        noise=NoiseConfig(**_filter_dataclass_kwargs(NoiseConfig, noise_yaml)),
        query_sample=QuerySampleConfig(**_filter_dataclass_kwargs(QuerySampleConfig, query_sample_yaml)),
        blocker=BlockerConfig(**_filter_dataclass_kwargs(BlockerConfig, blocker_yaml)),
        runtime=RuntimeConfig(**_filter_dataclass_kwargs(RuntimeConfig, merged.get("runtime", {}))),
        prompt=PromptConfig(
            prompt_dir=prompt_dir,
            system_runtime_prompt_file=prompt_yaml.get(
                "system_runtime_prompt_file",
                "system_runtime_prompt.txt",
            ),
            domain_context_file=domain_context_file,
        ),
        output=OutputConfig(
            root_dir=(project_root / output_root).resolve(),
            output_dir=final_output_dir,
            save_progress_every_turn=merged["output"].get("save_progress_every_turn", True),
            save_raw_llm_response=merged["output"].get("save_raw_llm_response", True),
        ),
        logging=LoggingConfig(**_filter_dataclass_kwargs(LoggingConfig, merged.get("logging", {}))),
        merged_config=merged,
    )
