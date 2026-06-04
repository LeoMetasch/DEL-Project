"""Standalone static conditional-belief scenario.

Alice and Bob both believe the hidden coin is Heads. The referee's known
habits also encode a simple revision policy: among the setups with a marked
card, Tails is more plausible than Heads. So, conditional on learning that
the hidden card is marked, each agent would revise to a Tails belief.
"""

from __future__ import annotations

from del_bench.kernel import (
    Atom,
    B,
    ConditionalB,
    EPM,
    Not,
    World,
    holds,
    total_partition,
    total_preorder_pairs,
)
from del_bench.scenarios.schema import (
    AnswerChoice,
    ProbeQuestion,
    Scenario,
    clean_text,
    gold_choice,
)

AGENTS = ("Alice", "Bob")
H = Atom("H")
T = Atom("T")
MARKED = Atom("marked")


def _build_model() -> EPM:
    heads_plain = World("heads_plain", frozenset({"H"}))
    tails_plain = World("tails_plain", frozenset({"T"}))
    heads_marked = World("heads_marked", frozenset({"H", "marked"}))
    tails_marked = World("tails_marked", frozenset({"T", "marked"}))
    worlds = (heads_plain, tails_plain, heads_marked, tails_marked)
    indist = {agent: total_partition(worlds) for agent in AGENTS}
    plaus = {
        agent: total_preorder_pairs(
            {
                heads_plain: 0,
                tails_marked: 1,
                heads_marked: 2,
                tails_plain: 3,
            }
        )
        for agent in AGENTS
    }
    return EPM(
        worlds=worlds,
        agents=AGENTS,
        indist=indist,
        plaus=plaus,
        actual=heads_plain,
    )


def build() -> Scenario:
    model = _build_model()

    assert holds(model, B("Alice", H))
    assert holds(model, ConditionalB("Alice", MARKED, T))
    assert not holds(model, ConditionalB("Alice", MARKED, H))
    assert holds(model, ConditionalB("Alice", Not(MARKED), H))
    assert holds(model, B("Alice", ConditionalB("Bob", MARKED, T)))

    no_belief = AnswerChoice(label="No belief either way", formula=None)

    probes = (
        ProbeQuestion(
            id="q1_factual",
            depth=0,
            nesting=(),
            natural_language="Is the coin currently showing heads or tails?",
            choices=(
                AnswerChoice("Heads", H),
                AnswerChoice("Tails", T),
            ),
        ),
        ProbeQuestion(
            id="q2_alice_belief",
            depth=1,
            nesting=("B_Alice",),
            natural_language=(
                "Before learning anything more, does Alice believe the coin is "
                "heads, believe it is tails, or have no belief either way?"
            ),
            choices=(
                AnswerChoice("Heads", B("Alice", H)),
                AnswerChoice("Tails", B("Alice", T)),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q3_alice_given_marked",
            depth=1,
            nesting=("BCond_Alice",),
            natural_language=(
                "If Alice were to learn that the hidden card is marked, would "
                "she believe the coin is heads, believe it is tails, or have "
                "no belief either way?"
            ),
            choices=(
                AnswerChoice("Heads", ConditionalB("Alice", MARKED, H)),
                AnswerChoice("Tails", ConditionalB("Alice", MARKED, T)),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q4_alice_given_plain",
            depth=1,
            nesting=("BCond_Alice",),
            natural_language=(
                "If Alice were to learn that the hidden card is plain, would "
                "she believe the coin is heads, believe it is tails, or have "
                "no belief either way?"
            ),
            choices=(
                AnswerChoice("Heads", ConditionalB("Alice", Not(MARKED), H)),
                AnswerChoice("Tails", ConditionalB("Alice", Not(MARKED), T)),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q5_alice_about_bob_given_marked",
            depth=2,
            nesting=("B_Alice", "BCond_Bob"),
            natural_language=(
                "Does Alice believe that, if Bob were to learn that the hidden "
                "card is marked, Bob would believe the coin is heads, believe "
                "it is tails, or have no belief either way?"
            ),
            choices=(
                AnswerChoice(
                    "Alice believes Bob would believe heads",
                    B("Alice", ConditionalB("Bob", MARKED, H)),
                ),
                AnswerChoice(
                    "Alice believes Bob would believe tails",
                    B("Alice", ConditionalB("Bob", MARKED, T)),
                ),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q6_syntactic_control",
            depth=2,
            nesting=(),
            natural_language=(
                "According to the scenario, are both the coin and the card "
                "hidden from Alice and Bob?"
            ),
            choices=(
                AnswerChoice("Yes, both the coin and the card are hidden", None),
                AnswerChoice("No, Alice and Bob can see the card", None),
            ),
            syntactic_control=True,
            fixed_gold_label="Yes, both the coin and the card are hidden",
        ),
    )

    narrative = clean_text("""
        Alice and Bob are playing a coin game with a referee.
        The referee places a coin under an opaque cup and a card inside a sealed envelope.
        The card is either plain or marked.
        Neither Alice nor Bob has seen the coin or the card.

        Alice and Bob both know the referee's habits.
        The most likely overall setup is a heads-up coin with a plain card.
        Among setups with a marked card, a tails-up coin is more likely than a heads-up coin.
        Among setups with a plain card, a heads-up coin is more likely than a tails-up coin.
        Alice and Bob both use these same habits when forming their beliefs.

        In fact, the coin is heads-up and the hidden card is plain.
        """)

    scenario = Scenario(
        id="coin_conditional_belief_v1",
        description="Static revision: a marked hidden card would shift the coin belief from heads to tails.",
        template_family="conditional_belief",
        formal_model=model,
        narrative=narrative,
        probes=probes,
    )

    for probe in probes:
        gold_choice(probe, model)

    return scenario


SCENARIO = build()
