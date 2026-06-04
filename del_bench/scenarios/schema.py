"""Scenario, ProbeQuestion, AnswerChoice dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Optional, Tuple

from del_bench.kernel import EPM, Formula


def clean_text(text: str) -> str:
    """Normalize a multiline narrative while preserving paragraphs."""
    paragraphs = []
    for block in dedent(text).strip().split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        paragraphs.append(" ".join(lines))
    return "\n\n".join(paragraphs)


@dataclass(frozen=True)
class AnswerChoice:
    """A possible answer presented to the LLM.

    The label is what appears in the prompt and the JSON enum. The formula
    is what the kernel evaluates to determine gold. A None formula means
    "no belief either way" — gold iff every other choice's formula is
    false at the actual world.
    """

    label: str
    formula: Optional[Formula]


@dataclass(frozen=True)
class ProbeQuestion:
    id: str
    depth: int  # logical nesting depth, 0 = factual
    nesting: Tuple[str, ...]  # e.g. ("B_Bob", "B_Alice")
    natural_language: str
    choices: Tuple[AnswerChoice, ...]
    syntactic_control: bool = (
        False  # True for linguistically-deep but logically-shallow probes
    )
    # If set, gold is determined by label (used for syntactic-control probes
    # whose answer comes from narrative inspection, not formal evaluation).
    fixed_gold_label: Optional[str] = None


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    template_family: str
    formal_model: EPM
    narrative: str
    probes: Tuple[ProbeQuestion, ...]
    pair_with: Optional[str] = None
    pair_diagnostic_probes: Tuple[str, ...] = ()
    # Additional narrative variants. The canonical ``narrative`` field is
    # always variant 0; this tuple supplies variants 1 and 2 when authored.
    narrative_variants: Tuple[str, ...] = ()

    @property
    def narrative_options(self) -> Tuple[str, ...]:
        """All narrative variants exposed to the renderer.

        Registered benchmark scenarios provide exactly three variants: seed 0
        uses the original narrative, and seeds 1/2 use paraphrase variants.
        """
        if self.narrative_variants:
            options = (self.narrative,) + self.narrative_variants
            if len(options) != 3:
                raise ValueError(
                    f"scenario {self.id}: expected exactly 3 narrative variants, got {len(options)}"
                )
            return options
        return (self.narrative,)


def gold_choice(probe: ProbeQuestion, model: EPM) -> AnswerChoice:
    """Return the unique choice that is gold for this probe.

    If ``fixed_gold_label`` is set on the probe, the choice with that label
    is returned (used for syntactic-control probes). Otherwise gold is
    determined by formal evaluation: the unique choice whose formula is
    true at the actual world. If no formula-bearing choice is true, the
    None-formula 'no belief' choice (if present) is gold.
    """
    if probe.fixed_gold_label is not None:
        for c in probe.choices:
            if c.label == probe.fixed_gold_label:
                return c
        raise ValueError(
            f"probe {probe.id}: fixed_gold_label {probe.fixed_gold_label!r} not found in choices"
        )

    from del_bench.kernel import holds

    truths = []
    fallback = None
    for c in probe.choices:
        if c.formula is None:
            fallback = c
            continue
        if holds(model, c.formula):
            truths.append(c)
    if len(truths) == 1:
        return truths[0]
    if len(truths) == 0:
        if fallback is None:
            raise ValueError(
                f"probe {probe.id}: no choice is gold and no 'no belief' fallback"
            )
        return fallback
    raise ValueError(
        f"probe {probe.id}: multiple choices evaluate true: {[c.label for c in truths]}"
    )
