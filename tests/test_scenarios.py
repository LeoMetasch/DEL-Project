"""Scenario integrity tests.

For every scenario:
  - every probe has a unique gold;
  - factual probe (q1) gold matches Atom("H") truth at the actual world.

Plus pair invariants: paired scenarios have identical golds on all shared
(non-diagnostic) probes and *differ* on their declared diagnostic probes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from del_bench.kernel import And, Atom, B, ConditionalB, K, Not, holds
from del_bench.scenarios.registry import SCENARIOS
from del_bench.scenarios.schema import gold_choice


def test_every_probe_has_unique_gold():
    for sid, scen in SCENARIOS.items():
        for p in scen.probes:
            g = gold_choice(p, scen.formal_model)
            assert g.label, f"{sid}/{p.id} no gold"


def test_q1_factual_matches_actual_world():
    for sid, scen in SCENARIOS.items():
        q1 = next((p for p in scen.probes if p.id == "q1_factual"), None)
        if q1 is None:
            continue
        gold = gold_choice(q1, scen.formal_model).label
        if holds(scen.formal_model, Atom("H")):
            assert gold == "Heads", f"{sid}/q1: actual is H but gold is {gold!r}"
        else:
            assert gold == "Tails", f"{sid}/q1: actual is T but gold is {gold!r}"


def test_paired_scenarios_share_non_diagnostic_golds_and_differ_on_diagnostics():
    for sid, scen in SCENARIOS.items():
        if not scen.pair_with:
            continue
        other = SCENARIOS[scen.pair_with]
        diag = set(scen.pair_diagnostic_probes)
        # Shared probes (not diagnostic, not syntactic control): golds must match.
        for pa in scen.probes:
            if pa.id in diag or pa.syntactic_control:
                continue
            pb = next((p for p in other.probes if p.id == pa.id), None)
            if pb is None:
                continue
            ga = gold_choice(pa, scen.formal_model).label
            gb = gold_choice(pb, other.formal_model).label
            assert (
                ga == gb
            ), f"pair {sid}/{scen.pair_with} differs on shared probe {pa.id}: {ga!r} vs {gb!r}"
        # Diagnostic probes: must differ.
        for did in scen.pair_diagnostic_probes:
            pa = next(p for p in scen.probes if p.id == did)
            pb = next(p for p in other.probes if p.id == did)
            ga = gold_choice(pa, scen.formal_model).label
            gb = gold_choice(pb, other.formal_model).label
            assert ga != gb, (
                f"diagnostic probe {did} should differ across pair "
                f"{sid}/{scen.pair_with} but both gold = {ga!r}"
            )


def test_each_scenario_has_a_syntactic_control_probe():
    for sid, scen in SCENARIOS.items():
        has = any(p.syntactic_control for p in scen.probes)
        assert has, f"{sid}: no syntactic-control probe"


def test_each_scenario_exposes_three_narrative_variants():
    for sid, scen in SCENARIOS.items():
        variants = scen.narrative_options
        assert (
            len(scen.narrative_variants) == 2
        ), f"{sid}: expected 2 hand-authored variants"
        assert len(variants) == 3, f"{sid}: expected 3 narrative variants"
        assert (
            variants[0] == scen.narrative
        ), f"{sid}: variant 0 must be canonical narrative"
        assert len(set(variants)) == 3, f"{sid}: narrative variants must be distinct"


def test_narrative_text_is_clean():
    for sid, scen in SCENARIOS.items():
        for idx, narrative in enumerate(scen.narrative_options):
            assert (
                narrative == narrative.strip()
            ), f"{sid}/variant{idx}: leading or trailing whitespace"
            assert (
                "   " not in narrative
            ), f"{sid}/variant{idx}: accidental triple space"
            assert (
                "\n\n\n" not in narrative
            ), f"{sid}/variant{idx}: too many blank lines"


def _modal_paths(formula, prefix=()):
    if isinstance(formula, Atom):
        return {prefix}
    if isinstance(formula, Not):
        return _modal_paths(formula.inner, prefix)
    if isinstance(formula, And):
        return _modal_paths(formula.left, prefix) | _modal_paths(formula.right, prefix)
    if isinstance(formula, K):
        return _modal_paths(formula.inner, prefix + (f"K_{formula.agent}",))
    if isinstance(formula, B):
        return _modal_paths(formula.inner, prefix + (f"B_{formula.agent}",))
    if isinstance(formula, ConditionalB):
        return _modal_paths(formula.condition, prefix) | _modal_paths(
            formula.inner, prefix + (f"BCond_{formula.agent}",)
        )
    raise TypeError(type(formula))


def test_formal_probe_depth_and_nesting_metadata_match_formulas():
    for sid, scen in SCENARIOS.items():
        for probe in scen.probes:
            if probe.syntactic_control:
                continue
            paths = set()
            for choice in probe.choices:
                if choice.formula is not None:
                    paths |= _modal_paths(choice.formula)
            max_depth = max((len(path) for path in paths), default=0)
            maximal_paths = {path for path in paths if len(path) == max_depth}
            assert (
                probe.depth == max_depth
            ), f"{sid}/{probe.id}: declared depth {probe.depth}, formula depth {max_depth}"
            assert probe.nesting in maximal_paths, (
                f"{sid}/{probe.id}: nesting {probe.nesting!r} not in maximal "
                f"formula paths {sorted(maximal_paths)!r}"
            )


def test_registered_benchmark_size_is_stable():
    assert len(SCENARIOS) == 18
    assert sum(len(s.probes) for s in SCENARIOS.values()) == 146
    assert (
        sum(1 for s in SCENARIOS.values() for p in s.probes if not p.syntactic_control)
        == 128
    )
    assert (
        sum(1 for s in SCENARIOS.values() for p in s.probes if p.syntactic_control)
        == 18
    )


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
