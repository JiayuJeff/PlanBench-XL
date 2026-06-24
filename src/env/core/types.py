from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class AuthConfig:
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None
    base_url_env: Optional[str] = None
    base_url: Optional[str] = None
    organization_env: Optional[str] = None
    organization: Optional[str] = None


@dataclass(slots=True)
class RequestConfig:
    temperature: Optional[float] = None
    max_tokens: int = 2000
    timeout_seconds: int = 120
    max_retries: int = 6
    reasoning_effort: Optional[str] = None


@dataclass(slots=True)
class ModelProfile:
    model_id: str
    provider: str
    api_style: str
    model_name: str
    auth: AuthConfig
    request: RequestConfig
    limits: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrieverConfig:
    embedding_model: Optional[str] = None


@dataclass(slots=True)
class NoiseConfig:
    mode: str = "none"
    ratio: float = 0.0
    max_total_tools: Optional[int] = None


@dataclass(slots=True)
class QuerySampleConfig:
    size: Optional[int] = None
    seed: int = 42


@dataclass(slots=True)
class BlockerConfig:
    enable_block: bool = False
    selection_mode: str = "target_remaining_ratio"
    block_n_per_task: Optional[int] = None
    target_remaining_paths: Optional[int] = None
    target_remaining_ratio: Optional[float] = 0.6
    remaining_tolerance: int = 1
    min_remaining_paths: int = 1
    remaining_path_length_objective: str = "none"
    blocking_edge_count_objective: str = "none"
    noise_mode: str = "random"
    fixed_noise_type: Optional[str] = None
    fixed_noise_types: Optional[list[str]] = None
    multi_noise_count: int = 1
    seed: int = 42
    max_combo_candidates: int = 5000
    max_cover_size: int = 4


@dataclass(slots=True)
class RuntimeConfig:
    max_steps: float = 20
    max_label_errors: int = 3
    max_retrieval_errors: int = 3
    max_call_errors: int = 3
    max_concurrency: int = 4
    llm_max_concurrent_requests: Optional[int] = None
    max_resume_attempts: int = 3
    truncate_history: bool = False
    enable_incorrect_final_answer_feedback: bool = False
    incorrect_final_answer_feedback_budget: int = 5


@dataclass(slots=True)
class PromptConfig:
    prompt_dir: Path
    system_runtime_prompt_file: str = "system_runtime_prompt.txt"
    domain_context_file: Optional[str] = None


@dataclass(slots=True)
class OutputConfig:
    root_dir: Path
    output_dir: Path
    save_progress_every_turn: bool = True
    save_raw_llm_response: bool = True


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    console: bool = True


@dataclass(slots=True)
class DataConfig:
    task_file: Path
    query_file: Path
    baseline_tools_file: Path
    noisy_tools_file: Path
    blocker_tools_file: Path
    datatypes_file: Path
    database_file: Path
    paths_set_catalog_file: Path


@dataclass(slots=True)
class RunnerConfig:
    run_id: str
    domain: str
    base_config_path: Path
    run_config_path: Path
    data: DataConfig
    model: ModelProfile
    retriever: RetrieverConfig
    noise: NoiseConfig
    query_sample: QuerySampleConfig
    blocker: BlockerConfig
    runtime: RuntimeConfig
    prompt: PromptConfig
    output: OutputConfig
    logging: LoggingConfig
    merged_config: dict[str, Any]


@dataclass(slots=True)
class QuerySpec:
    query_id: str
    task_id: str
    query_text: str
    correct_answer: Any
    input_datatypes: list[str]
    target_datatype: str
    input_type: Optional[str] = None
    input_values: dict[str, Any] | None = None


@dataclass(slots=True)
class AgentState:
    current_datatypes: set[str]
    trusted_values_by_datatype: dict[str, set[str]] = field(default_factory=dict)
    untrusted_values_by_datatype: dict[str, set[str]] = field(default_factory=dict)
    untrusted_value_sources_by_datatype: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    available_tool_names: list[str] = field(default_factory=list)
    available_tool_ids: list[str] = field(default_factory=list)
    discovered_tool_names: set[str] = field(default_factory=set)
    tool_name_to_id: dict[str, str] = field(default_factory=dict)
    tool_id_to_name: dict[str, str] = field(default_factory=dict)
    next_tool_id_num: int = 1
    total_step_count: int = 0
    retrieval_attempt_count: int = 0
    retrieval_exec_count: int = 0
    tool_call_attempt_count: int = 0
    tool_call_exec_count: int = 0
    label_error_cnt: int = 0
    retrieval_error_cnt: int = 0
    call_error_cnt: int = 0
    incorrect_final_answer_feedback_count: int = 0
    steps_trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalResult:
    request: dict[str, Any]
    tools: list[dict[str, Any]]
    matched_information: dict[str, Any] | None = None
    internal_retriever_note: str | None = None
    model_retriever_note: str | None = None


@dataclass(slots=True)
class ToolExecutionResult:
    tool_name: str
    success: bool
    output_datatype: str
    output_value: Any
    tool_type: str
    output_provenance: str
    untrusted_source_type: Optional[str]
