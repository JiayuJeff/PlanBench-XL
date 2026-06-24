from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Optional

from env.core.types import RetrievalResult


class _HashingTextEncoder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, text: str) -> dict[int, float]:
        normalized = text.strip().lower().replace("_", " ")
        tokens = re.findall(r"[a-z0-9]+", normalized)
        features: list[str] = []
        features.extend(tokens)
        features.extend(
            f"{tokens[index]}__{tokens[index + 1]}"
            for index in range(len(tokens) - 1)
        )

        collapsed = re.sub(r"\s+", " ", normalized)
        padded = f"  {collapsed}  "
        features.extend(
            f"char::{padded[index:index + 3]}"
            for index in range(max(0, len(padded) - 2))
        )

        vector: dict[int, float] = {}
        for feature in features:
            index = int(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).hexdigest(), 16) % self.dim
            vector[index] = vector.get(index, 0.0) + 1.0

        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {index: value / norm for index, value in vector.items()}

    def similarity(self, left: dict[int, float], right: dict[int, float]) -> float:
        if not left or not right:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(index, 0.0) for index, value in left.items())


class SemanticRetriever:
    def __init__(
        self,
        tool_registry: dict[str, dict[str, Any]],
        datatype_registry: dict[str, dict[str, dict[str, Any]]],
        embedding_model: str | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.datatype_registry = datatype_registry
        self.embedding_model = embedding_model or "hashing_v1"
        self.encoder = _HashingTextEncoder()
        self._domain_datatypes = {
            domain: {
                datatype_name: self._build_datatype_aliases(datatype_name, spec)
                for datatype_name, spec in datatypes.items()
            }
            for domain, datatypes in datatype_registry.items()
        }
        self._alias_embedding_cache: dict[str, dict[int, float]] = {}

    def _encode_alias(self, text: str) -> dict[int, float]:
        cached = self._alias_embedding_cache.get(text)
        if cached is not None:
            return cached
        embedding = self.encoder.encode(text)
        self._alias_embedding_cache[text] = embedding
        return embedding

    def _normalize_aliases(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for value in values:
            collapsed = " ".join(value.strip().split())
            if not collapsed:
                continue
            key = collapsed.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(collapsed)
        return normalized

    def _build_datatype_aliases(self, datatype_name: str, spec: dict[str, Any]) -> list[str]:
        aliases = [datatype_name, datatype_name.replace("_", " ")]
        raw_aliases = spec.get("aliases")
        if isinstance(raw_aliases, list):
            aliases.extend(alias for alias in raw_aliases if isinstance(alias, str))
        description = spec.get("description")
        if isinstance(description, str):
            aliases.append(description)
        return self._normalize_aliases(aliases)

    def _query_aliases(self, domain: str, value: str) -> list[str]:
        domain_registry = self._domain_datatypes.get(domain, {})
        datatype_aliases = domain_registry.get(value)
        if datatype_aliases:
            return datatype_aliases
        return self._normalize_aliases([value])

    def _tool_aliases(self, domain: str, datatype_name: str) -> list[str]:
        domain_registry = self._domain_datatypes.get(domain, {})
        datatype_aliases = domain_registry.get(datatype_name)
        if datatype_aliases:
            return datatype_aliases
        return self._normalize_aliases([datatype_name, datatype_name.replace("_", " ")])

    def _sim_datatype(self, domain: str, left_value: str, right_value: str) -> float:
        left_aliases = self._query_aliases(domain, left_value)
        right_aliases = self._tool_aliases(domain, right_value)
        best_score = 0.0
        for left_alias in left_aliases:
            left_embedding = self._encode_alias(left_alias)
            for right_alias in right_aliases:
                score = self.encoder.similarity(left_embedding, self._encode_alias(right_alias))
                if score > best_score:
                    best_score = score
        return best_score

    def _match_top1_datatype(self, domain: str, query_value: str) -> str | None:
        domain_registry = self._domain_datatypes.get(domain)
        if not domain_registry:
            return None
        best_name: str | None = None
        best_score = -1.0
        for datatype_name in domain_registry:
            score = self._sim_datatype(domain, query_value, datatype_name)
            if score > best_score:
                best_score = score
                best_name = datatype_name
        return best_name

    def retrieve_tools(
        self,
        request: dict[str, Any],
    ) -> RetrievalResult:
        requested_domain = str(request.get("domain") or "").strip()
        meta_tool = str(request.get("meta_tool") or "").strip()

        matched_information: dict[str, Any] = {}
        internal_retriever_note: str | None = None
        model_retriever_note: str | None = None
        matched_output: str | None = None
        matched_inputs: list[str] = []

        if meta_tool in ("by_output_info", "by_io_info"):
            output_query = str(request.get("output_information") or "").strip()
            matched_output = self._match_top1_datatype(requested_domain, output_query) if output_query else None
            matched_information["output_information"] = matched_output

        if meta_tool in ("by_inputs", "by_io_info"):
            raw_inputs = request.get("input_information")
            if isinstance(raw_inputs, list):
                for item in raw_inputs[:2]:
                    if not isinstance(item, str):
                        continue
                    matched = self._match_top1_datatype(requested_domain, item.strip())
                    if matched:
                        matched_inputs.append(matched)
            matched_information["input_information"] = matched_inputs

        tools: list[dict[str, Any]] = []
        if meta_tool == "by_output_info":
            if matched_output:
                tools = [
                    tool
                    for tool in self.tool_registry.values()
                    if tool.get("domain") == requested_domain and tool.get("output_datatype") == matched_output
                ]
        elif meta_tool == "by_inputs":
            if len(matched_inputs) == 1:
                tools = [
                    tool
                    for tool in self.tool_registry.values()
                    if tool.get("domain") == requested_domain and list(tool.get("input_datatypes") or []) == [matched_inputs[0]]
                ]
            elif len(matched_inputs) == 2:
                desired = set(matched_inputs)
                tools = [
                    tool
                    for tool in self.tool_registry.values()
                    if tool.get("domain") == requested_domain
                    and isinstance(tool.get("input_datatypes"), list)
                    and len(tool["input_datatypes"]) == 2
                    and set(tool["input_datatypes"]) == desired
                ]
                if not tools:
                    internal_retriever_note = (
                        f"No 2-input baseline tool matches that requested input combination in domain '{requested_domain}'."
                    )
                    model_retriever_note = (
                        "No direct tool matches that requested two-input combination. "
                        "Try a broader or different request."
                    )
        elif meta_tool == "by_io_info":
            if matched_output and len(matched_inputs) == 1:
                tools = [
                    tool
                    for tool in self.tool_registry.values()
                    if tool.get("domain") == requested_domain
                    and tool.get("output_datatype") == matched_output
                    and list(tool.get("input_datatypes") or []) == [matched_inputs[0]]
                ]
            elif matched_output and len(matched_inputs) == 2:
                desired = set(matched_inputs)
                tools = [
                    tool
                    for tool in self.tool_registry.values()
                    if tool.get("domain") == requested_domain
                    and tool.get("output_datatype") == matched_output
                    and isinstance(tool.get("input_datatypes"), list)
                    and len(tool["input_datatypes"]) == 2
                    and set(tool["input_datatypes"]) == desired
                ]
            if not tools:
                internal_retriever_note = (
                    "No direct single-tool match exists for the matched canonical input/output combination in this domain. "
                    f"matched_output={matched_output}, matched_inputs={matched_inputs}"
                )
                model_retriever_note = (
                    "No direct single-tool match exists for that requested input/output combination. "
                    "A multi-step path may still exist through intermediate information."
                )

        tools.sort(key=lambda tool: tool.get("name") or tool.get("tool_name") or "")

        return RetrievalResult(
            request=dict(request),
            tools=tools,
            matched_information=matched_information,
            internal_retriever_note=internal_retriever_note,
            model_retriever_note=model_retriever_note,
        )
