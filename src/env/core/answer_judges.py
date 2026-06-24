from __future__ import annotations

import json
import re
from typing import Any, Protocol


class AnswerJudge(Protocol):
    name: str

    def is_correct(self, expected: Any, actual: Any) -> bool:
        ...


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value).lower()
    text = re.sub(r"[*_`\"']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class NormalizedContainsJudge:
    name = "normalized_contains"

    def is_correct(self, expected: Any, actual: Any) -> bool:
        expected_text = normalize_answer(expected)
        actual_text = normalize_answer(actual)
        return bool(expected_text) and expected_text in actual_text


class LLMJudgePlaceholder:
    name = "llm_judge"

    def is_correct(self, expected: Any, actual: Any) -> bool:
        raise NotImplementedError(
            "LLM-as-a-judge is not configured yet. Add a concrete AnswerJudge implementation "
            "and select it from build_answer_judge()."
        )


def build_answer_judge(name: str) -> AnswerJudge:
    if name == "normalized_contains":
        return NormalizedContainsJudge()
    if name == "llm_judge":
        return LLMJudgePlaceholder()
    raise ValueError(f"Unsupported answer judge: {name}")
