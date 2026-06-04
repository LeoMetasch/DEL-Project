"""Scenario J: trusted lie after a fair-game observation.

Minimal pair with coin_soft_announcement_v1. Same prior, same announced
content (T), same narrative frame (someone claims tails), same probe
IDs and natural language. Only change: the source is a trusted speaker
who knows the truth (Bob lies about T) vs. a partially trusted bystander
(soft upgrade).

Since Bob is the speaker and knows the truth (H), his belief stays H.
Alice trusts Bob and believes T (same as in soft). The key divergence
is Bob's belief: Bob believes H here (he's the liar), T in soft
(he's just an audience member who updated).

Diverging golds vs. soft: q3_bob_belief (H here, T in soft).
Shared golds: q1 (factual = H), q2 (Alice believes T), q4, q5, q6.
"""

from __future__ import annotations

from del_bench.kernel import (
    Atom,
    B,
    EPM,
    K,
    Not,
    Or,
    World,
    canonical_actions,
    holds,
    product,
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


def _build_prior() -> EPM:
    s = World("s", frozenset({"H"}))
    t = World("t", frozenset({"T"}))
    worlds = (s, t)
    indist = {ag: total_partition(worlds) for ag in AGENTS}
    plaus = {ag: total_preorder_pairs({s: 0, t: 1}) for ag in AGENTS}
    return EPM(worlds=worlds, agents=AGENTS, indist=indist, plaus=plaus, actual=s)


def build() -> Scenario:
    prior = _build_prior()
    peek = canonical_actions.fair_game_observation(
        observer="Bob", P=H, agents=AGENTS, actual_truth_of_P=True
    )
    after_peek = product(prior, peek)
    lie = canonical_actions.successful_lie(
        speaker="Bob", lie_content=K("Bob", T), agents=AGENTS, actual_is_lie=True
    )
    model = product(after_peek, lie)

    assert holds(model, B("Alice", T))
    assert holds(model, B("Bob", H))
    assert not holds(model, K("Alice", T))

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
                "After the claim about the coin, does Alice believe the coin "
                "is heads, believe it is tails, or have no belief either way?"
            ),
            choices=(
                AnswerChoice("Heads", B("Alice", H)),
                AnswerChoice("Tails", B("Alice", T)),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q3_bob_belief",
            depth=1,
            nesting=("B_Bob",),
            natural_language=(
                "After the claim about the coin, does Bob believe the coin "
                "is heads, believe it is tails, or have no belief either way?"
            ),
            choices=(
                AnswerChoice("Heads", B("Bob", H)),
                AnswerChoice("Tails", B("Bob", T)),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q4_alice_knows",
            depth=1,
            nesting=("K_Alice",),
            natural_language=(
                "Does Alice know for certain which side the coin is showing, "
                "or does she not know for certain?"
            ),
            choices=(
                AnswerChoice(
                    "Alice knows which side", Or(K("Alice", H), K("Alice", T))
                ),
                AnswerChoice(
                    "Alice does not know for certain",
                    Not(Or(K("Alice", H), K("Alice", T))),
                ),
            ),
        ),
        ProbeQuestion(
            id="q5_alice_about_bob",
            depth=2,
            nesting=("B_Alice", "B_Bob"),
            natural_language=(
                "Does Alice believe that Bob believes the coin is heads, "
                "believe that Bob believes it is tails, or have no belief "
                "either way?"
            ),
            choices=(
                AnswerChoice(
                    "Alice believes Bob believes heads", B("Alice", B("Bob", H))
                ),
                AnswerChoice(
                    "Alice believes Bob believes tails", B("Alice", B("Bob", T))
                ),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q6_bob_about_alice_about_bob",
            depth=3,
            nesting=("B_Bob", "B_Alice", "B_Bob"),
            natural_language=(
                "Does Bob believe that Alice believes that Bob believes the "
                "coin is heads, or that Bob believes it is tails, or does "
                "Bob have no belief either way about this?"
            ),
            choices=(
                AnswerChoice(
                    "Bob believes Alice believes Bob believes heads",
                    B("Bob", B("Alice", B("Bob", H))),
                ),
                AnswerChoice(
                    "Bob believes Alice believes Bob believes tails",
                    B("Bob", B("Alice", B("Bob", T))),
                ),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q7_syntactic_control",
            depth=2,
            nesting=(),
            natural_language=(
                "According to the scenario, did a third party make a claim "
                "about the coin that both Alice and Bob heard?"
            ),
            choices=(
                AnswerChoice("Yes, a third party made a claim about the coin", None),
                AnswerChoice("No, no third party was involved", None),
            ),
            syntactic_control=True,
            fixed_gold_label="No, no third party was involved",
        ),
    )

    narrative = clean_text("""
        Alice and Bob are playing a game with a referee.
        The rules of the game are mutually understood by Alice and Bob, and have been for a long time.
        There is a coin lying face-down under an opaque cup on the table.
        Neither Alice nor Bob has seen which side is showing.
        Both believe, based on the referee's known habits, that the coin is heads-up.
        The coin is in fact heads-up.

        The referee lifts the cup briefly and shows the coin to Bob, in such a way that Alice can see Bob is looking at the coin but cannot see the coin's face herself.
        The referee then replaces the cup.
        Alice and Bob both understand that Bob has seen the coin while Alice has not seen the result.

        Bob then says aloud to Alice: "I am certain the coin is tails-up." Alice has not seen the coin and has no other source of information about it.
        Alice fully trusts Bob and is fully satisfied with what he has said.
        It is mutually understood between them that this exchange has just taken place and that Alice trusts Bob's word.
        """)

    scenario = Scenario(
        id="coin_trusted_lie_v1",
        description="Trusted lie: Bob sees heads, claims certainty about tails, and Alice trusts him.",
        template_family="trusted_lie",
        formal_model=model,
        narrative=narrative,
        probes=probes,
        pair_with="coin_soft_announcement_v1",
        pair_diagnostic_probes=("q3_bob_belief",),
    )

    for p in probes:
        gold_choice(p, model)

    return scenario


SCENARIO = build()
