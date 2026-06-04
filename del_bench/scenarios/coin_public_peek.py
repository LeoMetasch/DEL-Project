"""Scenario A: fair-game (public) peek.

Bob looks at the coin in plain view of Alice; Alice sees that Bob is
looking but cannot see the face herself.

Formal: prior model from Example 2.1 + fair_game_observation. Actual
state: coin is Heads, Bob has seen Heads.
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
    am = canonical_actions.fair_game_observation(
        observer="Bob", P=H, agents=AGENTS, actual_truth_of_P=True
    )
    model = product(prior, am)

    knows_side = Or(K("Bob", H), K("Bob", T))
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
        # Syntactic-control probe: a linguistically nested ("did the
        # referee say that the rules said ...") question whose answer is
        # directly retrievable from narrative inspection, with no
        # higher-order belief reasoning required.
        ProbeQuestion(
            id="q7_syntactic_control",
            depth=3,
            nesting=(),
            natural_language=(
                "According to the scenario as described, did the referee at "
                "any point lift the cup off the coin so that Bob could see "
                "the coin's face?"
            ),
            choices=(
                AnswerChoice("Yes, the referee lifted the cup so Bob could see", None),
                AnswerChoice("No, the referee did not lift the cup", None),
            ),
            syntactic_control=True,
            fixed_gold_label="Yes, the referee lifted the cup so Bob could see",
        ),
    )

    narrative = clean_text("""
        Alice and Bob are playing a game with a referee.
        The rules of the game are mutually understood by Alice and Bob, and have been for a long time.
        There is a coin lying face-down under an opaque cup on the table.
        Neither Alice nor Bob has yet seen which side is showing.
        Both believe, based on the referee's known habits, that the coin is heads-up.
        The coin is in fact heads-up.

        The referee lifts the cup briefly and shows the coin to Bob, in such a way that Alice can see Bob is looking at the coin but cannot see the coin's face herself.
        The referee then replaces the cup.
        Alice has clearly seen that Bob has now seen the coin's face.
        It is mutually understood by Alice and Bob that this is what just happened.
        """)

    scenario = Scenario(
        id="coin_public_peek_v1",
        description="Fair-game peek: Bob sees the coin while Alice watches.",
        template_family="public_peek",
        formal_model=model,
        narrative=narrative,
        probes=probes,
        pair_with="coin_private_peek_v1",
        pair_diagnostic_probes=(
            "q4_alice_about_bob_knows",
            "q6_bob_about_alice_about_bob_knows",
        ),
    )

    # Self-test: every probe must produce a unique gold.
    for p in probes:
        gold_choice(p, model)

    return scenario


SCENARIO = build()
