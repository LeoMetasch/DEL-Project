"""Scenario: conflicting claims (Axis 3 Hard).

Two speakers make conflicting claims about the coin. Bob claims tails
(lying), then Charles claims heads (truthfully). Alice hears both and
must reconcile: the second trusted claim overrides the first.

Chain: Prior → Σ₂(Bob sees H) → Σ₄(Bob lies K_b T)
      → Σ₂(Charles sees H) → Σ₄(Charles truth K_c H)

Pairs with coin_soft_announcement_v1 (trust-axis comparison). Shared
probes q1 and q4 have the same golds; diagnostic probes q2, q3, q5, q6
diverge because soft shifts beliefs to T while conflicting claims end at H.
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
    bob_peek = canonical_actions.fair_game_observation(
        observer="Bob", P=H, agents=AGENTS, actual_truth_of_P=True
    )
    after_bob_peek = product(prior, bob_peek)
    bob_lie = canonical_actions.successful_lie(
        speaker="Bob", lie_content=K("Bob", T), agents=AGENTS, actual_is_lie=True
    )
    after_lie = product(after_bob_peek, bob_lie)
    charles_peek = canonical_actions.fair_game_observation(
        observer="Charles", P=H, agents=AGENTS, actual_truth_of_P=True
    )
    after_charles_peek = product(after_lie, charles_peek)
    charles_truth = canonical_actions.successful_lie(
        speaker="Charles",
        lie_content=K("Charles", H),
        agents=AGENTS,
        actual_is_lie=False,
    )
    model = product(after_charles_peek, charles_truth)

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
                "After both claims about the coin, does Alice believe the coin "
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
                "After both claims about the coin, does Bob believe the coin "
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
                "According to the scenario, did two different people make "
                "conflicting claims about the coin?"
            ),
            choices=(
                AnswerChoice(
                    "Yes, two people made conflicting claims about the coin", None
                ),
                AnswerChoice("No, there were no conflicting claims", None),
            ),
            syntactic_control=True,
            fixed_gold_label="Yes, two people made conflicting claims about the coin",
        ),
        ProbeQuestion(
            id="q8_charles_belief",
            depth=1,
            nesting=("B_Charles",),
            natural_language=(
                "After both claims, does Charles believe the coin is heads, "
                "believe it is tails, or have no belief either way?"
            ),
            choices=(
                AnswerChoice("Heads", B("Charles", H)),
                AnswerChoice("Tails", B("Charles", T)),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q9_alice_about_charles",
            depth=2,
            nesting=("B_Alice", "B_Charles"),
            natural_language=(
                "Does Alice believe that Charles believes the coin is heads, "
                "believe that Charles believes it is tails, or have no belief "
                "either way?"
            ),
            choices=(
                AnswerChoice(
                    "Alice believes Charles believes heads", B("Alice", B("Charles", H))
                ),
                AnswerChoice(
                    "Alice believes Charles believes tails", B("Alice", B("Charles", T))
                ),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q10_alice_about_bob_about_alice_about_charles",
            depth=4,
            nesting=("B_Alice", "B_Bob", "B_Alice", "B_Charles"),
            natural_language=(
                "Looking four layers deep, does Alice believe that Bob believes "
                "that Alice believes that Charles believes the coin is heads, "
                "believe that Bob believes that Alice believes that Charles "
                "believes it is tails, or have no belief either way?"
            ),
            choices=(
                AnswerChoice(
                    "Alice believes Bob believes Alice believes Charles believes heads",
                    B("Alice", B("Bob", B("Alice", B("Charles", H)))),
                ),
                AnswerChoice(
                    "Alice believes Bob believes Alice believes Charles believes tails",
                    B("Alice", B("Bob", B("Alice", B("Charles", T)))),
                ),
                no_belief,
            ),
        ),
    )

    narrative = clean_text("""
        Alice, Bob, and Charles are playing a game with a referee.
        The rules of the game are mutually understood by all three players, and have been for a long time.
        There is a coin lying face-down under an opaque cup on the table.
        None of the three has yet seen which side is showing.
        All three believe, based on the referee's known habits, that the coin is heads-up.
        The coin is in fact heads-up.

        The referee lifts the cup briefly and shows the coin to Bob, in such a way that Alice and Charles can see Bob is looking at the coin but cannot see the coin's face themselves.
        The referee replaces the cup.
        It is mutually understood among all three that Bob has observed the coin while Alice and Charles have not seen the result.

        Bob speaks up first and says aloud to Alice and Charles: "I am certain the coin is tails-up." Both Alice and Charles hear this clearly.
        Bob is known to be knowledgeable and trustworthy about the game, and both Alice and Charles fully trust his word.
        It is mutually understood among all three that this exchange has taken place.

        The referee then lifts the cup briefly and shows the coin to Charles, in such a way that Alice and Bob can see Charles is looking at the coin but cannot see the coin's face themselves.
        The referee replaces the cup.
        It is mutually understood among all three that Charles has observed the coin while Alice and Bob have not seen his result.

        Then Charles speaks up and says aloud to Alice and Bob: "I am certain the coin is heads-up." Both Alice and Bob hear this clearly.
        Charles is known to be equally knowledgeable and trustworthy about the game, and both Alice and Bob fully trust his word.
        It is mutually understood among all three that this exchange has taken place and that both Bob's and Charles's claims have been heard by everyone.
        """)

    scenario = Scenario(
        id="coin_conflicting_claims_v1",
        description="Conflicting claims: Bob claims tails, then Charles truthfully claims heads.",
        template_family="conflicting_claims",
        formal_model=model,
        narrative=narrative,
        probes=probes,
        pair_with="coin_soft_announcement_v1",
        pair_diagnostic_probes=(
            "q2_alice_belief",
            "q3_bob_belief",
            "q5_alice_about_bob",
            "q6_bob_about_alice_about_bob",
        ),
    )

    for p in probes:
        gold_choice(p, model)

    return scenario


SCENARIO = build()
