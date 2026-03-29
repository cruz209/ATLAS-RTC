from __future__ import annotations

import json
from pathlib import Path
from typing import List

from atlas_rtc.telemetry.events import TelemetryEvent


class EventLogger:
    def __init__(self) -> None:
        self.events: List[TelemetryEvent] = []

    def log(self, name: str, **payload) -> None:
        self.events.append(TelemetryEvent(name=name, payload=payload))

    def dump_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event.to_dict()) + "\n")
