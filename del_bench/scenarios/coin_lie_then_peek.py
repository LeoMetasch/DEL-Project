"""Scenario: lie then discovery (Axis 2 Hard).

After privately peeking and seeing Heads, Bob lies to Alice that the coin
is tails. Then Alice gets to observe the coin in a fair-game check — she
sees the truth and discovers the lie.

Chain: Prior → Σ₃(Bob peeks H) → Σ₄(Bob lies T) → Σ₂(Alice observes H)

Pairs with coin_successful_lie_v1 (same first two steps, but adds the
discovery). Diagnostic probes: Alice's belief flips back to H, and
higher-order beliefs shift accordingly.
"""

from __future__ import annotations

from del_bench.kernel import (
    Atom,
    B,
    EPM,
    K,
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
    peek = canonical_actions.fully_private_observation(
        observer="Bob", P=H, agents=AGENTS, actual_truth_of_P=True
    )
    after_peek = product(prior, peek)
    lie = canonical_actions.successful_lie(
        speaker="Bob", lie_content=K("Bob", T), agents=AGENTS, actual_is_lie=True
    )
    after_lie = product(after_peek, lie)
    discovery = canonical_actions.fair_game_observation(
        observer="Alice", P=H, agents=AGENTS, actual_truth_of_P=True
    )
    model = product(after_lie, discovery)

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
            id="q4_bob_about_alice",
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
            id="q5_bob_about_alice_about_bob",
            depth=3,
            nesting=("B_Bob", "B_Alice", "B_Bob"),
            natural_language=(
                "Concerning Alice's belief about Bob's belief about the coin: "
                "does Bob believe that Alice believes Bob believes the coin is "
                "heads, believe that Alice believes Bob believes the coin is "
                "tails, or have no belief either way?"
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
            id="q6_depth4",
            depth=4,
            nesting=("B_Bob", "B_Alice", "B_Bob", "B_Alice"),
            natural_language=(
                "Now consider four layers of belief: does Bob believe that "
                "Alice believes that Bob believes that Alice believes the coin "
                "is heads, or does Bob believe that Alice believes that Bob "
                "believes that Alice believes the coin is tails?"
            ),
            choices=(
                AnswerChoice(
                    "Bob believes Alice believes Bob believes Alice believes heads",
                    B("Bob", B("Alice", B("Bob", B("Alice", H)))),
                ),
                AnswerChoice(
                    "Bob believes Alice believes Bob believes Alice believes tails",
                    B("Bob", B("Alice", B("Bob", B("Alice", T)))),
                ),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q7_syntactic_control",
            depth=3,
            nesting=(),
            natural_language=(
                "According to the scenario as described, did Bob, in Alice's "
                "presence, claim aloud that he was certain the coin was tails?"
            ),
            choices=(
                AnswerChoice(
                    "Yes, Bob claimed in Alice's presence that the coin was tails", None
                ),
                AnswerChoice("No, Bob made no such claim in Alice's presence", None),
            ),
            syntactic_control=True,
            fixed_gold_label="Yes, Bob claimed in Alice's presence that the coin was tails",
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

        Alice returns to the room.
        Bob then says aloud to Alice: "I am certain the coin is tails-up." Alice has not seen the coin and has no other source of information about it.
        Alice fully trusts Bob and is fully satisfied with what he has said.
        It is mutually understood between them that this exchange has just taken place and that Alice trusts Bob's word.

        Then the referee announces a coin check.
        In full view of both Alice and Bob, the referee lifts the cup and shows the coin to Alice.
        Alice sees that it is heads-up.
        Bob sees that Alice has looked at the coin, but he cannot see which side is showing — only Alice sees the face.
        The referee replaces the cup.
        It is mutually understood between Alice and Bob that Alice has observed the coin but that Bob has not seen her result.
        """)

    scenario = Scenario(
        id="coin_lie_then_peek_v1",
        description="Lie then discovery: Bob lies, then Alice observes the coin in a fair-game check.",
        template_family="lying",
        formal_model=model,
        narrative=narrative,
        probes=probes,
        pair_with="coin_successful_lie_v1",
        pair_diagnostic_probes=(
            "q3_alice_belief",
            "q4_bob_about_alice",
            "q5_bob_about_alice_about_bob",
            "q6_depth4",
        ),
    )

    for p in probes:
        gold_choice(p, model)

    return scenario


SCENARIO = build()
