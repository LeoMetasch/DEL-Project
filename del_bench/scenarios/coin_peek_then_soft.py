"""Scenario K: private peek then soft announcement.

Cross-axis pair with coin_soft_announcement_v1. Same prior, same soft
announcement content (T), same probe IDs and natural language. Only
change: Bob privately peeks (sees H) BEFORE the soft announcement.

Since Bob knows H (from the peek), the soft suggestion of T cannot
override his knowledge — Bob's belief stays H. Alice, who did not peek,
updates her belief to T (same as in raw soft).

Diverging golds vs. soft: q3_bob_belief (H here due to knowledge, T in
raw soft where Bob has no prior knowledge).
Shared golds: q1 (factual = H), q2 (Alice believes T), q4 (Alice
doesn't know), q5, q6.
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

    peek = canonical_actions.fully_private_observation(
        observer="Bob", P=H, agents=AGENTS, actual_truth_of_P=True
    )
    after_peek = product(prior, peek)

    soft = canonical_actions.lexicographic_upgrade(
        P=T, agents=AGENTS, actual_truth_of_P=False
    )
    model = product(after_peek, soft)

    assert holds(model, B("Alice", T))
    assert holds(model, B("Bob", H))
    assert not holds(model, K("Alice", T))
    assert not holds(model, K("Alice", H))

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
            fixed_gold_label="Yes, a third party made a claim about the coin",
        ),
    )

    narrative = clean_text("""
        Alice and Bob are playing a game with a referee.
        The rules of the game are mutually understood by Alice and Bob, and have been for a long time.
        The rules forbid peeking at the coin without permission.
        There is a coin lying face-down under an opaque cup on the table.
        Neither Alice nor Bob has seen which side is showing.
        Both believe, based on the referee's known habits, that the coin is heads-up.
        The coin is in fact heads-up.

        Alice steps out of the room briefly.
        While she is gone, Bob lifts the cup, looks at the coin, sees that it is heads-up, and replaces the cup.
        Alice does not see this and has no way of knowing it happened.
        The rules of the game are such that Alice has no reason whatsoever to suspect Bob would do this — Bob is known to play strictly by the rules — and it is mutually understood between them that Alice fully trusts Bob to do so.
        Bob says nothing about what he has done.

        Alice returns to the room.
        A bystander who occasionally watches the game walks by and says aloud: "I think the coin is tails-up this time." Both Alice and Bob hear this clearly.
        The bystander is known to sometimes be right and sometimes wrong — neither Alice nor Bob considers the bystander's opinion authoritative. Alice takes the opinion somewhat seriously because she did not see the result. Bob hears the same comment, but his direct observation of heads remains decisive.
        It is mutually understood by Alice and Bob that they have both heard the bystander's suggestion and that Alice gives it some weight. Alice remains unaware that Bob peeked, so she expects him to give the suggestion the same weight she does. Bob understands this. In fact, his direct observation keeps his knowledge and belief fixed on heads.

        Importantly, Alice cannot rule out the possibility that the coin is actually heads-up — the bystander might be wrong.
        The suggestion has nevertheless shifted Alice's inclination toward tails. Bob, by contrast, can rule out tails because he saw the coin land heads.
        """)

    scenario = Scenario(
        id="coin_peek_then_soft_v1",
        description="Private peek then soft announcement: Bob's knowledge resists the soft suggestion.",
        template_family="soft_announcement",
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
