from __future__ import annotations

from pathlib import Path


class PromptManager:
    def __init__(self, prompt_dir: Path) -> None:
        self.prompt_dir = prompt_dir
        self._cache: dict[str, str] = {}

    def _load(self, filename: str) -> str:
        if filename not in self._cache:
            self._cache[filename] = (self.prompt_dir / filename).read_text(encoding="utf-8")
        return self._cache[filename]

    def render(self, filename: str, **kwargs: object) -> str:
        text = self._load(filename)
        for key, value in kwargs.items():
            text = text.replace("{" + key + "}", str(value))
        return text
