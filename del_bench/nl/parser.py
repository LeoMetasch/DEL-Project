"""Parse LLM raw response into a chosen choice label, or a failure status."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class ParseStatus(str, Enum):
    OK = "OK"
    MALFORMED_JSON = "MALFORMED_JSON"
    UNKNOWN_LABEL = "UNKNOWN_LABEL"
    EMPTY = "EMPTY"


@dataclass(frozen=True)
class ParseResult:
    status: ParseStatus
    chosen_label: Optional[str]
    reasoning: Optional[str]
    raw: str


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _try_parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip code fences if present.
    stripped = _FENCE_RE.sub("", text).strip()
    if stripped != text:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    # Greedy extract first {...} block.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def parse_response(raw: str, choice_labels: Tuple[str, ...]) -> ParseResult:
    if not raw or not raw.strip():
        return ParseResult(ParseStatus.EMPTY, None, None, raw)

    obj = _try_parse_json(raw)
    if obj is None:
        return ParseResult(ParseStatus.MALFORMED_JSON, None, None, raw)

    answer = obj.get("answer") if isinstance(obj, dict) else None
    if not isinstance(answer, str):
        return ParseResult(ParseStatus.MALFORMED_JSON, None, None, raw)

    reasoning = obj.get("reasoning") if isinstance(obj, dict) else None
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)

    if answer in choice_labels:
        return ParseResult(ParseStatus.OK, answer, reasoning, raw)

    # Tolerant match: case-insensitive, whitespace-collapsed.
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    nans = norm(answer)
    for lbl in choice_labels:
        if norm(lbl) == nans:
            return ParseResult(ParseStatus.OK, lbl, reasoning, raw)

    return ParseResult(ParseStatus.UNKNOWN_LABEL, None, reasoning, raw)
