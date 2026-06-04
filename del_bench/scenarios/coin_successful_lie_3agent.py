"""Scenario P: three-agent successful lie.

Minimal extension of coin_successful_lie_v1. Same prior, same private
peek (Bob sees H), same lie (Bob claims T). Only change: Charles is
added in Alice's epistemic position (trusts Bob, believes the lie).

Probes q1-q7 are identical to the 2-agent lie (same IDs, formulas,
natural language). Golds should be identical. Probes q8-q10 are
Charles-specific.

Pairs with coin_successful_lie_v1 (degradation pair: adding a passive
third agent). Diagnostic probes are the Charles-specific ones.
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

AGENTS = ("Alice", "Bob", "Charles")
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
    model = product(after_peek, lie)

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
        ProbeQuestion(
            id="q8_charles_belief",
            depth=1,
            nesting=("B_Charles",),
            natural_language="Does Charles believe the coin is heads, believe it is tails, or have no belief either way?",
            choices=(
                AnswerChoice("Heads", B("Charles", H)),
                AnswerChoice("Tails", B("Charles", T)),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q9_bob_about_charles",
            depth=2,
            nesting=("B_Bob", "B_Charles"),
            natural_language=(
                "Concerning Charles's belief about the coin: does Bob believe "
                "Charles believes the coin is heads, believe Charles believes it "
                "is tails, or have no belief either way?"
            ),
            choices=(
                AnswerChoice(
                    "Bob believes Charles believes heads", B("Bob", B("Charles", H))
                ),
                AnswerChoice(
                    "Bob believes Charles believes tails", B("Bob", B("Charles", T))
                ),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q10_charles_about_bob_belief",
            depth=2,
            nesting=("B_Charles", "B_Bob"),
            natural_language=(
                "Concerning Bob's belief about the coin: does Charles believe "
                "that Bob believes the coin is heads, believe that Bob believes "
                "it is tails, or have no belief either way?"
            ),
            choices=(
                AnswerChoice(
                    "Charles believes Bob believes heads", B("Charles", B("Bob", H))
                ),
                AnswerChoice(
                    "Charles believes Bob believes tails", B("Charles", B("Bob", T))
                ),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q11_charles_about_bob_about_charles_about_bob",
            depth=4,
            nesting=("B_Charles", "B_Bob", "B_Charles", "B_Bob"),
            natural_language=(
                "At four layers, does Charles believe that Bob believes that "
                "Charles believes that Bob believes the coin is heads, believe "
                "that Bob believes that Charles believes that Bob believes it "
                "is tails, or have no belief either way?"
            ),
            choices=(
                AnswerChoice(
                    "Charles believes Bob believes Charles believes Bob believes heads",
                    B("Charles", B("Bob", B("Charles", B("Bob", H)))),
                ),
                AnswerChoice(
                    "Charles believes Bob believes Charles believes Bob believes tails",
                    B("Charles", B("Bob", B("Charles", B("Bob", T)))),
                ),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q12_charles_about_alice_about_bob_about_alice",
            depth=4,
            nesting=("B_Charles", "B_Alice", "B_Bob", "B_Alice"),
            natural_language=(
                "Does Charles believe that Alice believes that Bob believes "
                "that Alice believes the coin is heads, believe that Alice "
                "believes that Bob believes that Alice believes it is tails, "
                "or have no belief either way?"
            ),
            choices=(
                AnswerChoice(
                    "Charles believes Alice believes Bob believes Alice believes heads",
                    B("Charles", B("Alice", B("Bob", B("Alice", H)))),
                ),
                AnswerChoice(
                    "Charles believes Alice believes Bob believes Alice believes tails",
                    B("Charles", B("Alice", B("Bob", B("Alice", T)))),
                ),
                no_belief,
            ),
        ),
    )

    narrative = clean_text("""
        Alice, Bob, and Charles are playing a game with a referee.
        The rules of the game are mutually understood by all three players, and have been for a long time.
        The rules forbid peeking at the coin without permission.
        There is a coin lying face-down under an opaque cup on the table.
        None of the three has yet seen which side is showing.
        All three believe, based on the referee's known habits, that the coin is heads-up.
        The coin is in fact heads-up.

        Alice and Charles step out of the room briefly.
        While they are gone, Bob lifts the cup, looks at the coin, sees that it is heads-up, and replaces the cup.
        Neither Alice nor Charles sees this and neither has any way of knowing it happened.
        The rules of the game are such that Alice and Charles have no reason whatsoever to suspect Bob would do this — Bob is known to play strictly by the rules — and it is mutually understood among all three that Alice and Charles fully trust Bob to do so.

        Alice and Charles return to the room.
        Bob then says aloud to Alice and Charles: "I am certain the coin is tails-up." Neither Alice nor Charles has seen the coin and neither has any other source of information about it.
        Both Alice and Charles fully trust Bob and are fully satisfied with what he has said.
        It is mutually understood among all three that this exchange has just taken place and that Alice and Charles trust Bob's word.
        """)

    scenario = Scenario(
        id="coin_successful_lie_3agent_v1",
        description="Three-agent lie: Bob lies about tails to trusting Alice and Charles.",
        template_family="lying",
        formal_model=model,
        narrative=narrative,
        probes=probes,
        pair_with="coin_successful_lie_v1",
        pair_diagnostic_probes=(),
    )

    for p in probes:
        gold_choice(p, model)

    return scenario


SCENARIO = build()
