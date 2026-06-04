"""Renderer tests for narrative paraphrase selection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from del_bench.nl.renderer import render
from del_bench.nl.strategies import get as get_strategy
from del_bench.scenarios.registry import SCENARIOS
from del_bench.scenarios.schema import gold_choice


def test_seed_selects_narrative():
    scenario = SCENARIOS["coin_public_peek_v1"]
    probe = scenario.probes[0]
    strategy = get_strategy("direct")

    rendered = [render(scenario, probe, strategy, seed=i) for i in range(3)]

    assert [r.paraphrase_id for r in rendered] == [0, 1, 2]
    assert len({r.user for r in rendered}) == 3
    assert len({tuple(r.choice_labels) for r in rendered}) == 1
    assert len({str(r.response_format) for r in rendered}) == 1
    assert all(probe.natural_language in r.user for r in rendered)

    gold = gold_choice(probe, scenario.formal_model).label
    assert gold_choice(probe, scenario.formal_model).label == gold


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
