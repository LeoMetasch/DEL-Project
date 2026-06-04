"""Explicit semantic contracts for every registered benchmark scenario."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from del_bench.kernel import And, Atom, B, ConditionalB, K, Not, Or, holds
from del_bench.scenarios.registry import SCENARIOS

H = Atom("H")
T = Atom("T")


def _knows_side(agent: str):
    return Or(K(agent, H), K(agent, T))


MOORE = And(H, Not(B("Bob", H)))


CONTRACTS = {
    "coin_public_peek_v1": {
        "true": (H, K("Bob", H), B("Alice", H), B("Alice", _knows_side("Bob"))),
        "false": (K("Alice", H), B("Alice", Not(_knows_side("Bob")))),
    },
    "coin_private_peek_v1": {
        "true": (
            H,
            K("Bob", H),
            B("Alice", H),
            B("Alice", Not(_knows_side("Bob"))),
            B("Bob", B("Alice", H)),
        ),
        "false": (K("Alice", H), B("Alice", _knows_side("Bob"))),
    },
    "coin_successful_lie_v1": {
        "true": (
            H,
            K("Bob", H),
            B("Alice", T),
            B("Alice", K("Bob", T)),
            B("Bob", B("Alice", T)),
            B("Bob", B("Alice", K("Bob", T))),
        ),
        "false": (B("Alice", H), K("Alice", T)),
    },
    "coin_moore_sentence_v1": {
        "true": (H, K("Bob", H), B("Bob", H), B("Bob", Not(MOORE))),
        "false": (MOORE, B("Bob", MOORE), B("Bob", Not(B("Bob", H)))),
    },
    "coin_truthful_announcement_v1": {
        "true": (
            H,
            K("Bob", H),
            B("Alice", H),
            B("Alice", K("Bob", H)),
            B("Bob", B("Alice", H)),
        ),
        "false": (B("Alice", T),),
    },
    "coin_peek_then_announce_v1": {
        "true": (H, K("Bob", H), B("Alice", H), B("Alice", _knows_side("Bob"))),
        "false": (K("Alice", H), B("Alice", Not(_knows_side("Bob")))),
    },
    "coin_soft_announcement_v1": {
        "true": (H, B("Alice", T), B("Bob", T)),
        "false": (K("Alice", T), B("Alice", H)),
    },
    "coin_unreliable_announcement_v1": {
        "true": (H, B("Alice", H), B("Bob", H)),
        "false": (B("Alice", T), B("Bob", T)),
    },
    "coin_trusted_lie_v1": {
        "true": (
            H,
            K("Bob", H),
            B("Alice", T),
            B("Alice", K("Bob", T)),
            B("Bob", B("Alice", K("Bob", T))),
        ),
        "false": (B("Alice", H), K("Alice", T)),
    },
    "coin_public_peek_tails_v1": {
        "true": (T, K("Bob", T), B("Alice", H), B("Alice", _knows_side("Bob"))),
        "false": (H, K("Alice", T), B("Alice", T)),
    },
    "coin_peek_then_soft_v1": {
        "true": (
            H,
            K("Bob", H),
            B("Alice", T),
            B("Bob", H),
            B("Alice", B("Bob", T)),
            B("Bob", B("Alice", B("Bob", T))),
        ),
        "false": (
            B("Alice", H),
            B("Bob", T),
            B("Alice", B("Bob", H)),
        ),
    },
    "coin_private_peek_3agent_v1": {
        "true": (
            H,
            K("Bob", H),
            B("Alice", H),
            B("Charles", H),
            B("Alice", Not(_knows_side("Bob"))),
            B("Charles", Not(_knows_side("Bob"))),
        ),
        "false": (K("Alice", H), K("Charles", H)),
    },
    "coin_successful_lie_3agent_v1": {
        "true": (
            H,
            K("Bob", H),
            B("Alice", T),
            B("Charles", T),
            B("Alice", K("Bob", T)),
            B("Charles", K("Bob", T)),
        ),
        "false": (B("Alice", H), B("Charles", H)),
    },
    "coin_double_peek_v1": {
        "true": (
            H,
            K("Bob", H),
            K("Alice", H),
            B("Alice", Not(_knows_side("Bob"))),
            B("Bob", Not(_knows_side("Alice"))),
        ),
        "false": (
            B("Alice", _knows_side("Bob")),
            B("Bob", _knows_side("Alice")),
        ),
    },
    "coin_lie_then_peek_v1": {
        "true": (
            H,
            K("Bob", H),
            K("Alice", H),
            B("Alice", H),
            B("Bob", B("Alice", H)),
            B("Bob", _knows_side("Alice")),
        ),
        "false": (B("Alice", T),),
    },
    "coin_conflicting_claims_v1": {
        "true": (
            H,
            K("Bob", H),
            K("Charles", H),
            B("Alice", H),
            B("Alice", K("Charles", H)),
            B("Alice", B("Bob", H)),
        ),
        "false": (K("Alice", H), B("Alice", T)),
    },
    "coin_witness_deception_v1": {
        "true": (
            H,
            K("Bob", H),
            K("Charles", H),
            B("Alice", T),
            B("Alice", K("Bob", T)),
            B("Charles", H),
        ),
        "false": (B("Alice", H), B("Charles", T)),
    },
    "coin_conditional_belief_v1": {
        "true": (
            H,
            B("Alice", H),
            ConditionalB("Alice", Atom("marked"), T),
            ConditionalB("Alice", Not(Atom("marked")), H),
            B("Alice", ConditionalB("Bob", Atom("marked"), T)),
        ),
        "false": (
            B("Alice", T),
            ConditionalB("Alice", Atom("marked"), H),
        ),
    },
}


def test_every_registered_scenario_has_a_semantic_contract():
    assert set(CONTRACTS) == set(SCENARIOS)


def test_registered_scenario_semantic_contracts():
    for scenario_id, contract in CONTRACTS.items():
        model = SCENARIOS[scenario_id].formal_model
        for formula in contract["true"]:
            assert holds(model, formula), f"{scenario_id}: expected {formula} to hold"
        for formula in contract["false"]:
            assert not holds(
                model, formula
            ), f"{scenario_id}: expected {formula} not to hold"


if __name__ == "__main__":
    import traceback

    failed = 0
    for key, value in list(globals().items()):
        if key.startswith("test_"):
            try:
                value()
                print(f"PASS {key}")
            except Exception as exc:
                failed += 1
                print(f"FAIL {key}: {exc}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
