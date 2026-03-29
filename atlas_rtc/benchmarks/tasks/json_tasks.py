from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class JSONTask:
    prompt: str
    required_keys: List[str]


def sample_tasks() -> list[JSONTask]:
    return [
        JSONTask(
            prompt='Return a JSON object with keys "name" and "age".',
            required_keys=["name", "age"],
        ),
        JSONTask(
            prompt='Return a JSON object with keys "city" and "country".',
            required_keys=["city", "country"],
        ),
    ]
