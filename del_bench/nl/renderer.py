"""Render scenario + probe to a (system, user, response_format) triple.

One LLM call per probe — keeps each probe isolated and avoids ordering
effects. The seed argument selects the scenario narrative paraphrase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from del_bench.nl.strategies import PromptStrategy
from del_bench.scenarios.schema import ProbeQuestion, Scenario

SYSTEM_BASE = (
    "You are answering questions about a short scenario involving multiple "
    "people and their beliefs. Read the scenario carefully and answer the "
    "question by selecting exactly one option from the choices given.\n\n"
    'When the scenario describes that someone "believes" something, this '
    "means they consider it the most likely possibility based on what they "
    "currently know — they may be mistaken if they have been deceived or if "
    "their information is incomplete. When the scenario describes that "
    'someone "knows" something, this means they have direct evidence and '
    "could not be mistaken. These distinctions matter for some of the "
    "questions.\n\n"
    "If a question asks what someone believes and the scenario gives no "
    "basis for them to have a settled belief one way or the other, choose "
    '"No belief either way".'
)


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str
    response_format: dict
    choice_labels: Tuple[str, ...]
    paraphrase_id: int


def render(
    scenario: Scenario,
    probe: ProbeQuestion,
    strategy: PromptStrategy,
    seed: int = 0,
) -> RenderedPrompt:
    narratives = scenario.narrative_options
    paraphrase_id = seed % len(narratives)
    narrative = narratives[paraphrase_id]
    labels = tuple(c.label for c in probe.choices)
    choices_text = "\n".join(f"  - {lbl}" for lbl in labels)

    system = f"{SYSTEM_BASE}\n\n{strategy.system_suffix}"
    user = (
        f"Scenario:\n\n{narrative}\n\n"
        f"Question: {probe.natural_language}\n\n"
        f"Choices:\n{choices_text}"
    )
    response_format = strategy.schema_for_choices(labels)

    return RenderedPrompt(
        system=system,
        user=user,
        response_format=response_format,
        choice_labels=labels,
        paraphrase_id=paraphrase_id,
    )
