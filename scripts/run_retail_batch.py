#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - optional dependency
    _tqdm = None


LOCAL_MODEL_SUFFIX = "-local"
BASE_URL_PATTERN = re.compile(r"^\s*base_url:\s*(?P<value>\S+)\s*$")


class _NullProgressBar:
    def __init__(self, total: int, desc: str) -> None:
        self.total = total
        self.desc = desc
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_description_str(self, desc: str) -> None:
        self.desc = desc

    def set_postfix_str(self, postfix: str) -> None:
        _ = postfix

    def write(self, message: str) -> None:
        print(message)

    def close(self) -> None:
        return None


def _build_progress_bar(*, total: int, desc: str, unit: str, position: int = 0) -> object:
    if _tqdm is None:
        return _NullProgressBar(total, desc)
    return _tqdm(
        total=total,
        desc=desc,
        unit=unit,
        position=position,
        dynamic_ncols=True,
        leave=True,
        disable=not sys.stderr.isatty(),
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _retail_config_root(project_root: Path) -> Path:
    return project_root / "src" / "env" / "config" / "runs" / "retail"


def _model_config_root(project_root: Path) -> Path:
    return project_root / "src" / "env" / "config" / "models" / "openai"


def _discover_models(retail_root: Path) -> list[str]:
    return sorted(path.name for path in retail_root.iterdir() if path.is_dir())


def _is_local_model(model_name: str) -> bool:
    return model_name.endswith(LOCAL_MODEL_SUFFIX)


def _normalize_models(raw_models: list[str] | None) -> list[str]:
    if not raw_models:
        return []

    normalized: list[str] = []
    for raw_model in raw_models:
        for piece in raw_model.split(","):
            model = piece.strip()
            if model:
                normalized.append(model)
    return normalized


def _normalize_config_tokens(raw_configs: list[str] | None) -> list[str]:
    if not raw_configs:
        return []

    normalized: list[str] = []
    for raw_config in raw_configs:
        for piece in raw_config.split(","):
            config = piece.strip()
            if not config:
                continue
            if config.endswith(".yaml"):
                config = config[: -len(".yaml")]
            normalized.append(config)
    return normalized


def _select_models(
    retail_root: Path,
    requested_models: list[str],
    *,
    include_local_models: bool,
) -> list[Path]:
    available = _discover_models(retail_root)
    if not requested_models:
        selected_models = available
        if not include_local_models:
            selected_models = [model for model in selected_models if not _is_local_model(model)]
        return [retail_root / model for model in selected_models]

    unknown = sorted({model for model in requested_models if model not in available})
    if unknown:
        available_hint = ", ".join(available)
        missing = ", ".join(unknown)
        raise SystemExit(f"Unknown model(s): {missing}\nAvailable models: {available_hint}")

    deduped_models: list[str] = []
    seen: set[str] = set()
    for model in requested_models:
        if model in seen:
            continue
        seen.add(model)
        deduped_models.append(model)
    return [retail_root / model for model in deduped_models]


def _discover_run_configs(model_dirs: list[Path]) -> list[Path]:
    run_configs: list[Path] = []
    for model_dir in model_dirs:
        run_configs.extend(sorted(model_dir.glob("*.yaml")))
    return run_configs


def _config_matches(run_config: Path, requested_configs: list[str]) -> bool:
    if not requested_configs:
        return True

    stem = run_config.stem
    filename = run_config.name
    for token in requested_configs:
        if stem == token or filename == token:
            return True
        if stem.endswith(f"_{token}") or filename.endswith(f"_{token}.yaml"):
            return True
    return False


def _filter_run_configs(run_configs: list[Path], requested_configs: list[str]) -> list[Path]:
    return [run_config for run_config in run_configs if _config_matches(run_config, requested_configs)]


def _format_run_config(project_root: Path, run_config: Path) -> str:
    try:
        return str(run_config.relative_to(project_root))
    except ValueError:
        return str(run_config)


def _read_local_model_ports(project_root: Path) -> dict[str, int | None]:
    ports: dict[str, int | None] = {}
    for model_path in sorted(_model_config_root(project_root).glob(f"*{LOCAL_MODEL_SUFFIX}.yaml")):
        port: int | None = None
        for line in model_path.read_text(encoding="utf-8").splitlines():
            match = BASE_URL_PATTERN.match(line)
            if not match:
                continue
            parsed = urlparse(match.group("value").strip("\"'"))
            port = parsed.port
            break
        ports[model_path.stem] = port
    return ports


def _format_local_model_ports(project_root: Path) -> str:
    ports = _read_local_model_ports(project_root)
    if not ports:
        return "No local-model YAMLs found under src/env/config/models/openai."

    lines = ["Current local-model OpenAI-compatible ports from model YAMLs:"]
    for model_name, port in ports.items():
        port_text = "unknown" if port is None else str(port)
        lines.append(f"  - {model_name}: {port_text}")
    return "\n".join(lines)


def _format_progress_label(run_config: Path) -> str:
    return f"{run_config.parent.name} | {run_config.stem}"


def _extract_output_dir_override(forwarded_args: list[str]) -> str | None:
    output_dir_override: str | None = None
    index = 0
    while index < len(forwarded_args):
        token = forwarded_args[index]
        candidate = token
        if token == "--set" and index + 1 < len(forwarded_args):
            candidate = forwarded_args[index + 1]
            index += 2
        else:
            index += 1

        if not isinstance(candidate, str) or "=" not in candidate:
            continue
        key, value = candidate.split("=", 1)
        if key == "output.output_dir":
            output_dir_override = value
    return output_dir_override


def _materialize_forwarded_args(
    forwarded_args: list[str],
    *,
    model_name: str,
    config_name: str,
) -> list[str]:
    materialized: list[str] = []
    index = 0
    while index < len(forwarded_args):
        token = forwarded_args[index]
        if token == "--set" and index + 1 < len(forwarded_args):
            entry = forwarded_args[index + 1]
            materialized.extend(["--set", _materialize_override_entry(entry, model_name, config_name)])
            index += 2
            continue

        materialized.append(_materialize_override_entry(token, model_name, config_name))
        index += 1
    return materialized


def _materialize_override_entry(entry: str, model_name: str, config_name: str) -> str:
    if not isinstance(entry, str) or "=" not in entry:
        return entry
    key, value = entry.split("=", 1)
    if key != "output.output_dir":
        return entry
    rendered_value = value.format(model=model_name, config=config_name)
    return f"{key}={rendered_value}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run retail experiment YAMLs in batch. By default this runs every non-local YAML "
            "under src/env/config/runs/retail. Use --model and --config to narrow the batch."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Run only the specified model directory. Repeat the flag or pass a comma-separated "
            "list to select multiple models. Example: --model gpt-5.4"
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Run only matching config names. Repeat the flag or pass a comma-separated list. "
            "Examples: --config default, --config blocker, --config retail_gpt5.4_default."
        ),
    )
    parser.add_argument(
        "--resume-validation",
        choices=("strict", "warn-signature", "ignore-signature"),
        default="warn-signature",
        help="Forwarded to src/env/run.py for every selected YAML.",
    )
    parser.add_argument(
        "--set",
        dest="override_entries",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override any config field for every selected YAML. You can also pass bare KEY=VALUE "
            "arguments without --set. If you override output.output_dir while running multiple "
            "YAMLs, use placeholders like {model} and {config} to keep outputs distinct."
        ),
    )
    parser.add_argument(
        "--include-local-models",
        action="store_true",
        help="Include *-local model directories in the default full-batch selection.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the YAMLs that would be run and exit.",
    )
    parser.add_argument(
        "--list-local-model-ports",
        action="store_true",
        help="Print local-model ports parsed from src/env/config/models/openai/*-local.yaml and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed without running them.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if any run exits with a non-zero status.",
    )

    args, unknown = parser.parse_known_args()
    args.requested_models = _normalize_models(args.model)
    args.requested_configs = _normalize_config_tokens(args.config)

    forwarded_args: list[str] = []
    for entry in args.override_entries:
        forwarded_args.extend(["--set", entry])
    forwarded_args.extend(unknown)
    args.forwarded_args = forwarded_args
    return args


def main() -> int:
    args = parse_args()
    project_root = _project_root()

    if args.list_local_model_ports:
        print(_format_local_model_ports(project_root))
        return 0

    retail_root = _retail_config_root(project_root)
    model_dirs = _select_models(
        retail_root,
        args.requested_models,
        include_local_models=args.include_local_models,
    )
    run_configs = _filter_run_configs(_discover_run_configs(model_dirs), args.requested_configs)

    if not run_configs:
        raise SystemExit(
            "No retail YAMLs matched the current filters. "
            f"models={args.requested_models or ['ALL_NON_LOCAL']}, configs={args.requested_configs or ['ALL']}"
        )

    output_dir_override = _extract_output_dir_override(args.forwarded_args)
    if (
        output_dir_override is not None
        and len(run_configs) > 1
        and "{model}" not in output_dir_override
        and "{config}" not in output_dir_override
    ):
        raise SystemExit(
            "You selected multiple retail YAMLs but overrode output.output_dir with one fixed path. "
            "That would make different runs share the same output directory and interfere with resume. "
            "Use placeholders such as output.output_dir=retail/{model}/{config}, or run "
            "src/env/run.py directly for a single YAML."
        )

    if args.list:
        for run_config in run_configs:
            print(_format_run_config(project_root, run_config))
        return 0

    failures: list[tuple[Path, int]] = []
    total = len(run_configs)
    batch_bar = _build_progress_bar(total=total, desc="retail batch", unit="config", position=0)

    try:
        for index, run_config in enumerate(run_configs, start=1):
            display_path = _format_run_config(project_root, run_config)
            forwarded_args = _materialize_forwarded_args(
                args.forwarded_args,
                model_name=run_config.parent.name,
                config_name=run_config.stem,
            )
            cmd = [
                sys.executable,
                "src/env/run.py",
                "--run_config",
                display_path,
                "--resume-validation",
                args.resume_validation,
                *forwarded_args,
            ]

            batch_bar.set_description_str(_format_progress_label(run_config))
            batch_bar.set_postfix_str(f"config {index}/{total}")

            if args.dry_run:
                batch_bar.write(f"[dry-run {index}/{total}] {display_path}")
                batch_bar.write(shlex.join(cmd))
                batch_bar.update(1)
                continue

            batch_bar.write(f"[run {index}/{total}] {display_path}")
            env = os.environ.copy()
            env.setdefault("PWMT_PROGRESS_POSITION", "1")
            completed = subprocess.run(cmd, cwd=project_root, env=env)
            batch_bar.update(1)

            if completed.returncode == 0:
                continue

            failures.append((run_config, completed.returncode))
            batch_bar.write(f"failed: {display_path} (exit code {completed.returncode})")
            if args.fail_fast:
                break
    finally:
        batch_bar.close()

    if failures:
        print("\nFailed runs:", file=sys.stderr)
        for run_config, returncode in failures:
            print(
                f"- {_format_run_config(project_root, run_config)} (exit code {returncode})",
                file=sys.stderr,
            )
        return 1

    print(f"\nCompleted {total} retail run config(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
