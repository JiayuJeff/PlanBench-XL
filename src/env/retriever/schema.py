from __future__ import annotations

from typing import Any


ALLOWED_DOMAINS = ["finance", "retail", "healthcare", "aviation", "education"]

ALLOWED_META_TOOLS = ["by_output_info", "by_inputs", "by_io_info"]

RETRIEVE_REQUEST_EXAMPLES = [
    {"meta_tool": "by_output_info", "output_information": "audit status"},
    {"meta_tool": "by_inputs", "input_information": ["draft item"]},
    {"meta_tool": "by_inputs", "input_information": ["pricing snapshot", "variant"]},
    {
        "meta_tool": "by_io_info",
        "output_information": "authorization code",
        "input_information": ["pricing snapshot", "variant"],
    },
]


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_str_list(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        collapsed = _normalize_text(item)
        if collapsed:
            normalized.append(collapsed)
    return normalized


def normalize_retrieve_request(request: dict[str, Any], default_domain: str) -> dict[str, Any]:
    meta_tool = _normalize_text(str(request["meta_tool"])).strip()
    normalized: dict[str, Any] = {"domain": default_domain, "meta_tool": meta_tool}

    if meta_tool == "by_output_info":
        normalized["output_information"] = _normalize_text(str(request["output_information"]))
        return normalized

    if meta_tool == "by_inputs":
        normalized["input_information"] = _normalize_str_list(request["input_information"])
        return normalized

    # by_io_info
    normalized["output_information"] = _normalize_text(str(request["output_information"]))
    normalized["input_information"] = _normalize_str_list(request["input_information"])
    return normalized


def validate_retrieve_request(
    request: dict[str, Any], default_domain: str
) -> tuple[bool, str, dict[str, Any] | None]:
    if not isinstance(request, dict):
        return False, "invalid_json_root", None

    meta_tool = request.get("meta_tool")
    if not isinstance(meta_tool, str) or not meta_tool.strip():
        return False, "missing_or_invalid_meta_tool", None

    meta_tool = _normalize_text(meta_tool)
    if meta_tool not in ALLOWED_META_TOOLS:
        return False, "invalid_meta_tool", None

    if meta_tool == "by_output_info":
        output_information = request.get("output_information")
        if not isinstance(output_information, str) or not output_information.strip():
            return False, "missing_or_invalid_output_information", None
        normalized = normalize_retrieve_request(
            {"meta_tool": meta_tool, "output_information": output_information},
            default_domain,
        )
        return True, "", normalized

    if meta_tool == "by_inputs":
        input_information = request.get("input_information")
        if not isinstance(input_information, list):
            return False, "missing_or_invalid_input_information", None
        normalized_inputs = _normalize_str_list(input_information)
        if len(normalized_inputs) not in (1, 2):
            return False, "invalid_input_count", None
        normalized = normalize_retrieve_request(
            {"meta_tool": meta_tool, "input_information": normalized_inputs},
            default_domain,
        )
        return True, "", normalized

    # by_io_info
    output_information = request.get("output_information")
    if not isinstance(output_information, str) or not output_information.strip():
        return False, "missing_or_invalid_output_information", None
    input_information = request.get("input_information")
    if not isinstance(input_information, list):
        return False, "missing_or_invalid_input_information", None
    normalized_inputs = _normalize_str_list(input_information)
    if len(normalized_inputs) not in (1, 2):
        return False, "invalid_input_count", None
    normalized = normalize_retrieve_request(
        {
            "meta_tool": meta_tool,
            "output_information": output_information,
            "input_information": normalized_inputs,
        },
        default_domain,
    )
    if not normalized.get("output_information"):
        return False, "missing_or_invalid_output_information", None
    return True, "", normalized


def format_retrieve_request_instructions() -> str:
    lines = [
        "When you retrieve tools, choose exactly one meta_tool and fill only the required fields.",
        f"Allowed meta_tool values: {', '.join(ALLOWED_META_TOOLS)}",
        "Schema:",
        '- by_output_info: {"meta_tool":"by_output_info","output_information":"..."}',
        '- by_inputs: {"meta_tool":"by_inputs","input_information":["..."]} (len=1) or {"input_information":["...","..."]} (len=2)',
        '- by_io_info: {"meta_tool":"by_io_info","output_information":"...","input_information":["..."]} (len=1) or (len=2)',
        "Write short information phrases only. Do not add explanation or reasoning inside JSON.",
        "Examples:",
    ]
    for example in RETRIEVE_REQUEST_EXAMPLES:
        lines.append(f"- {example}")
    return "\n".join(lines)
