"""Scenario I: completely unreliable announcement (Example 3.5, p.44).

Minimal pair with coin_soft_announcement_v1. Same prior, same announced
content (T), same narrative frame (third party claims tails), same probe
IDs and natural language. Only change: the source is completely
untrusted (equiplausible actions) vs. partially trusted (soft upgrade).

Since the agents don't trust the source at all, the honest and dishonest
alternatives are equiplausible. Original beliefs remain unchanged.

Diverging golds vs. soft: q2, q3 (beliefs stay H here, flip to T in
soft), q5, q6 (higher-order beliefs follow).
Shared golds: q1 (factual = H), q4 (nobody knows in either).
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

    am = canonical_actions.unreliable_announcement(
        P=T, agents=AGENTS, actual_truth_of_P=False
    )
    model = product(prior, am)

    assert holds(model, B("Alice", H))
    assert holds(model, B("Bob", H))
    assert not holds(model, B("Alice", T))
    assert not holds(model, B("Bob", T))

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
        ProbeQuestion(
            id="q8_depth4_bob_about_alice_about_bob_about_alice",
            depth=4,
            nesting=("B_Bob", "B_Alice", "B_Bob", "B_Alice"),
            natural_language=(
                "At four layers of belief, does Bob believe that Alice believes "
                "that Bob believes that Alice believes the coin is heads, "
                "believe that Alice believes that Bob believes that Alice "
                "believes it is tails, or have no belief either way?"
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
    )

    narrative = clean_text("""
        Alice and Bob are playing a game with a referee.
        The rules of the game are mutually understood by Alice and Bob, and have been for a long time.
        There is a coin lying face-down under an opaque cup on the table.
        Neither Alice nor Bob has seen which side is showing.
        Both believe, based on the referee's known habits, that the coin is heads-up.
        The coin is in fact heads-up.

        A stranger whom neither Alice nor Bob has ever met walks past and says aloud: "The coin is tails-up." Both Alice and Bob hear this clearly.
        However, both Alice and Bob know that this stranger has absolutely no information about the coin and is completely untrustworthy — the stranger is equally likely to be right or wrong and is known to make random guesses.
        It is mutually understood by Alice and Bob that neither of them places any trust whatsoever in what the stranger has said.

        The stranger leaves.
        """)

    scenario = Scenario(
        id="coin_unreliable_announcement_v1",
        description="Unreliable announcement: an untrustworthy stranger claims tails; beliefs should not change.",
        template_family="unreliable_announcement",
        formal_model=model,
        narrative=narrative,
        probes=probes,
        pair_with="coin_soft_announcement_v1",
        pair_diagnostic_probes=(
            "q2_alice_belief",
            "q3_bob_belief",
            "q5_alice_about_bob",
            "q6_bob_about_alice_about_bob",
            "q8_depth4_bob_about_alice_about_bob_about_alice",
        ),
    )

    for p in probes:
        gold_choice(p, model)

    return scenario


SCENARIO = build()
