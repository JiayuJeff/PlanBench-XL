from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def sample_sequence(items: Sequence[T], size: int | None, seed: int) -> list[T]:
    if size is None:
        return list(items)
    if size < 0:
        raise ValueError("query_sample.size must be null or non-negative")
    if size > len(items):
        raise ValueError(f"query_sample.size={size} exceeds available query count={len(items)}")

    rng = random.Random(seed)
    selected_indices = set(rng.sample(range(len(items)), size))
    return [item for index, item in enumerate(items) if index in selected_indices]
