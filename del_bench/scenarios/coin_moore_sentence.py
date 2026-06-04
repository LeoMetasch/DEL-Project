"""Scenario D: Moore-sentence test — DEL's signature contribution.

The classical "Moore sentence" is φ := p ∧ ¬B_a p ("p is true but a does
not believe p"). After a truthful public announcement of φ, agent a knows
p, hence believes p, hence ¬B_a p is now false — so the announced
sentence has *become false by being announced*. This is exactly the
phenomenon the AGM "Success" axiom mishandles and that motivated the
paper's distinction between static and dynamic belief revision (paper
§3, page 38-39).

Setup: Alice has been privately shown the coin (it is Heads); Bob has
not seen it and on prior grounds is inclined to believe Tails. Alice
then publicly announces to Bob a Moore sentence about the coin and
Bob's belief.

The depth-2 punchline probe asks whether Bob, after the announcement,
believes that he does not believe the coin is Heads. The naive answer
(matching what was said) is "yes." The logically correct answer is
"no": Bob now believes Heads, and Bob's belief is introspective in our
K□ semantics, so Bob believes he believes Heads — meaning he does NOT
believe he does not believe Heads. Failing this probe is the textbook
Moore-sentence error.
"""

from __future__ import annotations

from del_bench.kernel import (
    And,
    Atom,
    B,
    EPM,
    K,
    Not,
    Or,
    World,
    canonical_actions,
    discrete_partition,
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
    """Alice has seen the coin (knows H); Bob has not, and on prior grounds
    believes Tails. Actual world: H."""
    s = World("s", frozenset({"H"}))
    t = World("t", frozenset({"T"}))
    worlds = (s, t)
    indist = {
        "Alice": discrete_partition(worlds),  # Alice distinguishes — knows the face
        "Bob": total_partition(worlds),  # Bob does not
    }
    plaus = {
        # Alice: each world is the only one in its cell; just reflexive pairs
        "Alice": frozenset({(s, s), (t, t)}),
        # Bob: prior favors T (rank 0) over H (rank 1) — i.e. Bob believes T
        "Bob": total_preorder_pairs({t: 0, s: 1}),
    }
    return EPM(worlds=worlds, agents=AGENTS, indist=indist, plaus=plaus, actual=s)


def build() -> Scenario:
    prior = _build_prior()
    # Alice's announcement: the Moore sentence φ = H ∧ ¬B_b H.
    moore = And(H, Not(B("Bob", H)))

    # Sanity at the actual prior world: φ should be true (H true, B_b H false).
    from del_bench.kernel import holds

    assert holds(prior, H)
    assert not holds(prior, B("Bob", H))
    assert holds(prior, moore)

    am = canonical_actions.public_announcement(P=moore, agents=AGENTS)
    model = product(prior, am)

    # After the hard public announcement, only the H-world survives. So Bob
    # now knows H, hence believes H, hence ¬B_b H is false at the actual
    # world, hence the announced sentence is false there.
    assert holds(model, H)
    assert holds(model, K("Bob", H))
    assert holds(model, B("Bob", H))
    assert not holds(model, moore)  # the announcement is now false
    assert not holds(model, B("Bob", moore))  # and Bob does not believe it
    assert not holds(model, B("Bob", Not(B("Bob", H))))  # Moore punchline

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
            id="q2_bob_believes_heads_now",
            depth=1,
            nesting=("B_Bob",),
            natural_language=(
                "After Alice's announcement, does Bob believe the coin is heads, "
                "believe it is tails, or have no belief either way?"
            ),
            choices=(
                AnswerChoice("Heads", B("Bob", H)),
                AnswerChoice("Tails", B("Bob", T)),
                no_belief,
            ),
        ),
        # The Moore-sentence punchline. Naive answer (matching what was
        # announced) is "yes." Correct answer is "no" — Bob now believes
        # Heads, so he does not believe that he does not believe Heads.
        ProbeQuestion(
            id="q3_moore_punchline",
            depth=2,
            nesting=("B_Bob", "B_Bob"),
            natural_language=(
                "Alice's announcement included the claim that Bob does not "
                "believe the coin is heads. After hearing Alice's announcement "
                "and accepting it, does Bob now himself believe the claim that "
                "he does not believe the coin is heads, or does he not believe "
                "that claim?"
            ),
            choices=(
                AnswerChoice(
                    "Bob believes the claim that he does not believe heads",
                    B("Bob", Not(B("Bob", H))),
                ),
                AnswerChoice(
                    "Bob does not believe the claim that he does not believe heads",
                    Not(B("Bob", Not(B("Bob", H)))),
                ),
                no_belief,
            ),
        ),
        # Companion: does Bob believe the full announced sentence? Same
        # underlying reason: the announcement made itself false.
        ProbeQuestion(
            id="q4_bob_believes_full_announcement",
            depth=2,
            nesting=("B_Bob", "B_Bob"),
            natural_language=(
                "Considering Alice's full announcement as a single statement "
                '("the coin is heads, and Bob does not believe the coin is '
                'heads"): does Bob now believe that full statement is true, '
                "does Bob believe that full statement is false, or does Bob "
                "have no belief either way about that full statement?"
            ),
            choices=(
                AnswerChoice(
                    "Bob believes the full statement is true", B("Bob", moore)
                ),
                AnswerChoice(
                    "Bob believes the full statement is false", B("Bob", Not(moore))
                ),
                no_belief,
            ),
        ),
        ProbeQuestion(
            id="q5_alice_about_bob",
            depth=2,
            nesting=("B_Alice", "B_Bob"),
            natural_language=(
                "After the announcement, does Alice believe Bob believes the "
                "coin is heads, believe Bob believes it is tails, or have no "
                "belief either way?"
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
            id="q6_syntactic_control",
            depth=2,
            nesting=(),
            natural_language=(
                "According to the scenario as described, did Alice publicly "
                "make a statement to Bob about the coin?"
            ),
            choices=(
                AnswerChoice(
                    "Yes, Alice publicly stated something to Bob about the coin", None
                ),
                AnswerChoice("No, Alice made no public statement", None),
            ),
            syntactic_control=True,
            fixed_gold_label="Yes, Alice publicly stated something to Bob about the coin",
        ),
    )

    narrative = clean_text("""
        Alice and Bob are playing a game with a referee.
        The rules of the game are mutually understood by Alice and Bob, and have been for a long time.
        There is a coin lying face-down under an opaque cup on the table.

        Before the game began, the referee privately showed the coin to Alice (and only to Alice).
        Alice has therefore seen the coin's face and knows with certainty which side is showing.
        The coin is in fact heads-up.

        Bob has not seen the coin.
        On the basis of the way the referee placed the coin and the referee's known habits in earlier rounds, Bob's settled current belief is that the coin is tails-up.

        Alice now speaks to Bob, in clear earshot, and says: "The coin is heads-up, and you do not currently believe the coin is heads-up."

        Bob completely trusts Alice's truthfulness about the coin and accepts what she has just told him.
        Both Alice and Bob mutually understand that Alice has just spoken and that Bob has accepted what she said.
        """)

    scenario = Scenario(
        id="coin_moore_sentence_v1",
        description="Moore-sentence: a true announcement that becomes false by being announced.",
        template_family="moore_sentence",
        formal_model=model,
        narrative=narrative,
        probes=probes,
        pair_with=None,
        pair_diagnostic_probes=(),
    )

    for p in probes:
        gold_choice(p, model)

    return scenario


SCENARIO = build()
