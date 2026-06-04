"""Scenario registry. To add a scenario: import its module and add its
SCENARIO instance to the dictionary below."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from del_bench.scenarios.paraphrases import PARAPHRASES
from del_bench.scenarios.schema import Scenario
from del_bench.scenarios import (
    coin_public_peek,
    coin_private_peek,
    coin_successful_lie,
    coin_moore_sentence,
    coin_truthful_announcement,
    coin_peek_then_announce,
    coin_soft_announcement,
    coin_unreliable_announcement,
    coin_trusted_lie,
    coin_public_peek_tails,
    coin_peek_then_soft,
    coin_private_peek_3agent,
    coin_successful_lie_3agent,
    coin_double_peek,
    coin_lie_then_peek,
    coin_conflicting_claims,
    coin_witness_deception,
    coin_conditional_belief,
)


def _with_paraphrases(scenario: Scenario) -> Scenario:
    variants = PARAPHRASES.get(scenario.id)
    if variants is None:
        raise KeyError(f"missing hand-authored paraphrases for {scenario.id}")
    if len(variants) != 2:
        raise ValueError(f"{scenario.id}: expected exactly 2 added paraphrases")
    return replace(scenario, narrative_variants=variants)


SCENARIOS: Dict[str, Scenario] = {
    coin_public_peek.SCENARIO.id: _with_paraphrases(coin_public_peek.SCENARIO),
    coin_private_peek.SCENARIO.id: _with_paraphrases(coin_private_peek.SCENARIO),
    coin_successful_lie.SCENARIO.id: _with_paraphrases(coin_successful_lie.SCENARIO),
    coin_moore_sentence.SCENARIO.id: _with_paraphrases(coin_moore_sentence.SCENARIO),
    coin_truthful_announcement.SCENARIO.id: _with_paraphrases(
        coin_truthful_announcement.SCENARIO
    ),
    coin_peek_then_announce.SCENARIO.id: _with_paraphrases(
        coin_peek_then_announce.SCENARIO
    ),
    coin_soft_announcement.SCENARIO.id: _with_paraphrases(
        coin_soft_announcement.SCENARIO
    ),
    coin_unreliable_announcement.SCENARIO.id: _with_paraphrases(
        coin_unreliable_announcement.SCENARIO
    ),
    coin_trusted_lie.SCENARIO.id: _with_paraphrases(coin_trusted_lie.SCENARIO),
    coin_public_peek_tails.SCENARIO.id: _with_paraphrases(
        coin_public_peek_tails.SCENARIO
    ),
    coin_peek_then_soft.SCENARIO.id: _with_paraphrases(coin_peek_then_soft.SCENARIO),
    coin_private_peek_3agent.SCENARIO.id: _with_paraphrases(
        coin_private_peek_3agent.SCENARIO
    ),
    coin_successful_lie_3agent.SCENARIO.id: _with_paraphrases(
        coin_successful_lie_3agent.SCENARIO
    ),
    coin_double_peek.SCENARIO.id: _with_paraphrases(coin_double_peek.SCENARIO),
    coin_lie_then_peek.SCENARIO.id: _with_paraphrases(coin_lie_then_peek.SCENARIO),
    coin_conflicting_claims.SCENARIO.id: _with_paraphrases(
        coin_conflicting_claims.SCENARIO
    ),
    coin_witness_deception.SCENARIO.id: _with_paraphrases(
        coin_witness_deception.SCENARIO
    ),
    coin_conditional_belief.SCENARIO.id: _with_paraphrases(
        coin_conditional_belief.SCENARIO
    ),
}

extra_paraphrases = set(PARAPHRASES) - set(SCENARIOS)
if extra_paraphrases:
    raise KeyError(
        f"paraphrases provided for unknown scenarios: {sorted(extra_paraphrases)}"
    )


def get(scenario_id: str) -> Scenario:
    if scenario_id not in SCENARIOS:
        raise KeyError(
            f"unknown scenario {scenario_id!r}. available: {sorted(SCENARIOS)}"
        )
    return SCENARIOS[scenario_id]


def all_ids() -> tuple[str, ...]:
    return tuple(sorted(SCENARIOS))
