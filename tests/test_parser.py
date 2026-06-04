"""Parser robustness tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from del_bench.nl.parser import ParseStatus, parse_response

CHOICES = ("Heads", "Tails", "No belief either way")


def test_clean_json():
    r = parse_response('{"answer": "Heads"}', CHOICES)
    assert r.status == ParseStatus.OK
    assert r.chosen_label == "Heads"


def test_json_with_reasoning():
    r = parse_response('{"reasoning": "because", "answer": "Tails"}', CHOICES)
    assert r.status == ParseStatus.OK
    assert r.chosen_label == "Tails"
    assert r.reasoning == "because"


def test_code_fence_stripping():
    raw = '```json\n{"answer": "Heads"}\n```'
    r = parse_response(raw, CHOICES)
    assert r.status == ParseStatus.OK
    assert r.chosen_label == "Heads"


def test_prose_with_embedded_json():
    raw = 'Sure thing! Here is my answer: {"answer": "Tails"}. Hope that helps.'
    r = parse_response(raw, CHOICES)
    assert r.status == ParseStatus.OK
    assert r.chosen_label == "Tails"


def test_unknown_label():
    r = parse_response('{"answer": "Maybe"}', CHOICES)
    assert r.status == ParseStatus.UNKNOWN_LABEL
    assert r.chosen_label is None


def test_case_insensitive_match():
    r = parse_response('{"answer": "heads"}', CHOICES)
    assert r.status == ParseStatus.OK
    assert r.chosen_label == "Heads"


def test_empty():
    r = parse_response("", CHOICES)
    assert r.status == ParseStatus.EMPTY


def test_malformed():
    r = parse_response("not json at all really", CHOICES)
    assert r.status == ParseStatus.MALFORMED_JSON


if __name__ == "__main__":
    import traceback

    failed = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            try:
                v()
                print(f"PASS {k}")
            except Exception as e:
                failed += 1
                print(f"FAIL {k}: {e}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
