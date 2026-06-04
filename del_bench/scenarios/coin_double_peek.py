"""Scenario: double private peek (Axis 1 Hard).

Both agents peek at the coin privately, in sequence. Each now knows the
coin is heads, but each believes the other does not know — genuine
mutual ignorance of each other's knowledge.

Chain: Prior → Σ₃(Bob, H) → Σ₃(Alice, H)

Pairs with coin_private_peek_v1 as a degradation pair (same golds on
shared probes, but increased complexity from two private observations).
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
    bob_peek = canonical_actions.fully_private_observation(
        observer="Bob", P=H, agents=AGENTS, actual_truth_of_P=True
    )
    after_bob = product(prior, bob_peek)
    alice_peek = canonical_actions.fully_private_observation(
        observer="Alice", P=H, agents=AGENTS, actual_truth_of_P=True
    )
    model = product(after_bob, alice_peek)

    knows_side = Or(K("Bob", H), K("Bob", T))
    alice_knows_side = Or(K("Alice", H), K("Alice", T))
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
            id="q2_bob_belief",
            depth=1,
            nesting=("B_Bob",),
            natural_language="Does Bob believe the coin is heads, believe it is tails, or have no belief either way?",
            choices=(
                AnswerChoice("Heads", B("Bob", H)),
                AnswerChoice("Tails", B("Bob", T)),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q3_alice_belief",
            depth=1,
            nesting=("B_Alice",),
            natural_language="Does Alice believe the coin is heads, believe it is tails, or have no belief either way?",
            choices=(
                AnswerChoice("Heads", B("Alice", H)),
                AnswerChoice("Tails", B("Alice", T)),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q4_alice_about_bob_knows",
            depth=2,
            nesting=("B_Alice", "K_Bob"),
            natural_language=(
                "Concerning Bob's knowledge of the coin: does Alice believe that "
                "Bob knows which side is showing, believe that Bob does not know, "
                "or have no belief either way?"
            ),
            choices=(
                AnswerChoice(
                    "Alice believes Bob knows which side", B("Alice", knows_side)
                ),
                AnswerChoice(
                    "Alice believes Bob does not know", B("Alice", Not(knows_side))
                ),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q5_bob_about_alice_belief",
            depth=2,
            nesting=("B_Bob", "B_Alice"),
            natural_language=(
                "Concerning Alice's belief about the coin: does Bob believe Alice "
                "believes the coin is heads, believe Alice believes it is tails, "
                "or have no belief either way?"
            ),
            choices=(
                AnswerChoice(
                    "Bob believes Alice believes heads", B("Bob", B("Alice", H))
                ),
                AnswerChoice(
                    "Bob believes Alice believes tails", B("Bob", B("Alice", T))
                ),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q6_bob_about_alice_about_bob_knows",
            depth=3,
            nesting=("B_Bob", "B_Alice", "K_Bob"),
            natural_language=(
                "Concerning Alice's belief about Bob's knowledge: does Bob believe "
                "that Alice believes Bob knows which side is showing, believe "
                "that Alice believes Bob does not know, or have no belief either way?"
            ),
            choices=(
                AnswerChoice(
                    "Bob believes Alice believes Bob knows the side",
                    B("Bob", B("Alice", knows_side)),
                ),
                AnswerChoice(
                    "Bob believes Alice believes Bob does not know",
                    B("Bob", B("Alice", Not(knows_side))),
                ),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q7_syntactic_control",
            depth=3,
            nesting=(),
            natural_language=(
                "According to the scenario as described, did both Alice and Bob "
                "each separately lift the cup and look at the coin?"
            ),
            choices=(
                AnswerChoice(
                    "Yes, both Alice and Bob each looked at the coin separately", None
                ),
                AnswerChoice(
                    "No, only one of them or neither looked at the coin", None
                ),
            ),
            syntactic_control=True,
            fixed_gold_label="Yes, both Alice and Bob each looked at the coin separately",
        ),
        ProbeQuestion(
            id="q8_bob_about_alice_knows",
            depth=2,
            nesting=("B_Bob", "K_Alice"),
            natural_language=(
                "Concerning Alice's knowledge of the coin: does Bob believe that "
                "Alice knows which side is showing, believe that Alice does not "
                "know, or have no belief either way?"
            ),
            choices=(
                AnswerChoice(
                    "Bob believes Alice knows which side", B("Bob", alice_knows_side)
                ),
                AnswerChoice(
                    "Bob believes Alice does not know", B("Bob", Not(alice_knows_side))
                ),
                no_belief,
            ),
        ),
    )

    narrative = clean_text("""
        Alice and Bob are playing a game with a referee.
        The rules of the game are mutually understood by Alice and Bob, and have been for a long time.
        The rules forbid peeking at the coin without permission.
        There is a coin lying face-down under an opaque cup on the table.
        Neither Alice nor Bob has yet seen which side is showing.
        Both believe, based on the referee's known habits, that the coin is heads-up.
        The coin is in fact heads-up.

        Alice steps out of the room briefly.
        While she is gone, Bob lifts the cup, looks at the coin, sees that it is heads-up, and replaces the cup.
        Alice does not see this and has no way of knowing it happened.
        The rules of the game are such that Alice has no reason whatsoever to suspect Bob would do this — Bob is known to play strictly by the rules — and it is mutually understood between them that Alice fully trusts Bob to do so.
        Bob says nothing about what he has done.

        Alice returns to the room.
        A short while later, Bob steps out briefly.
        While he is gone, Alice lifts the cup, looks at the coin, sees that it is heads-up, and replaces the cup.
        Bob does not see this and has no way of knowing it happened.
        The rules of the game are such that Bob has no reason whatsoever to suspect Alice would do this — Alice is known to play strictly by the rules — and it is mutually understood between them that Bob fully trusts Alice to do so.
        Alice says nothing about what she has done.

        Bob returns to the room.
        """)

    scenario = Scenario(
        id="coin_double_peek_v1",
        description="Double private peek: both agents secretly view the coin in turn, neither knowing the other peeked.",
        template_family="private_peek",
        formal_model=model,
        narrative=narrative,
        probes=probes,
        pair_with="coin_private_peek_v1",
        pair_diagnostic_probes=(),
    )

    for p in probes:
        gold_choice(p, model)

    return scenario


SCENARIO = build()
