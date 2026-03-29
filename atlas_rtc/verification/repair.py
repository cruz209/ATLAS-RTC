from __future__ import annotations


class SimpleRepair:
    def repair_json_prefix(self, text: str) -> str:
        if text and not text.lstrip().startswith("{"):
            return "{" + text
        return text
