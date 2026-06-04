"""Reproduce the worked examples from Baltag & Smets (LOFT7 2008).

These tests are non-negotiable: if any of them fails, the kernel is wrong
and every gold answer downstream is suspect.

  - Example 2.1 (page 26): prior model S — both agents believe coin Heads.
  - Example 2.2 (page 27): model W — Bob has seen the coin (fair-game peek);
    Alice still believes Heads but no longer in a 'safe' way.
  - Example 2.3 (page 31): model S' — Bob privately peeked; Alice believes
    nothing happened, still believes Heads.
  - Example 3.3 / page 46: S' ⊗ Σ_4 — successful lie continuation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from del_bench.kernel import (
    EPM,
    Atom,
    B,
    ConditionalB,
    K,
    Not,
    World,
    canonical_actions,
    evaluate,
    holds,
    product,
    total_partition,
    total_preorder_pairs,
)
from del_bench.kernel.formulas import nested

AGENTS = ("Alice", "Bob")
H = Atom("H")
T = Atom("T")


def make_prior() -> EPM:
    """Example 2.1: both agents believe Heads, coin actually Heads."""
    s = World("s", frozenset({"H"}))
    t = World("t", frozenset({"T"}))
    worlds = (s, t)
    indist = {ag: total_partition(worlds) for ag in AGENTS}
    # Both agents: s <_a t (s strictly more plausible).
    plaus = {ag: total_preorder_pairs({s: 0, t: 1}) for ag in AGENTS}
    return EPM(worlds=worlds, agents=AGENTS, indist=indist, plaus=plaus, actual=s)


# ---------------------------------------------------------------------------
# Example 2.1
# ---------------------------------------------------------------------------


def test_example_2_1_prior_beliefs():
    S = make_prior()
    assert holds(S, H)
    assert holds(S, B("Alice", H))
    assert holds(S, B("Bob", H))
    assert not holds(S, K("Alice", H))
    assert not holds(S, K("Bob", H))
    # Higher-order: Alice believes Bob believes H, and vice versa.
    assert holds(S, B("Alice", B("Bob", H)))
    assert holds(S, B("Bob", B("Alice", H)))
    # If Alice learnt T, she would revise her belief to T.
    assert holds(S, ConditionalB("Alice", T, T))
    assert not holds(S, ConditionalB("Alice", T, H))


# ---------------------------------------------------------------------------
# Example 2.2: fair-game observation.
# Result should match model W (page 27): Bob now knows the face; Alice
# still believes H but no longer holds a 'safe' belief.
# ---------------------------------------------------------------------------


def test_example_2_2_fair_game():
    S = make_prior()
    am = canonical_actions.fair_game_observation(
        observer="Bob", P=H, agents=AGENTS, actual_truth_of_P=True
    )
    W = product(S, am)

    # Bob now knows H.
    assert holds(W, K("Bob", H))
    # Alice does not know H.
    assert not holds(W, K("Alice", H))
    # Alice still believes H (her prior preference is preserved by the
    # equiplausibility of σ_H, σ_¬H for her, anti-lex update keeps prior
    # order on states).
    assert holds(W, B("Alice", H))
    # Alice believes Bob now knows whether H or T (he distinguishes them).
    # Phrased: Alice believes (K_b H ∨ K_b T). Equivalently: Alice
    # believes ¬(¬K_b H ∧ ¬K_b T).
    from del_bench.kernel import And, Or

    assert holds(W, B("Alice", Or(K("Bob", H), K("Bob", T))))
    # Bob believes Alice still believes H.
    assert holds(W, B("Bob", B("Alice", H)))


# ---------------------------------------------------------------------------
# Example 2.3: fully private observation.
# Alice still believes Heads. Alice believes Bob did not look (so Alice
# believes Bob does not know which side is showing).
# ---------------------------------------------------------------------------


def test_example_2_3_private_peek():
    S = make_prior()
    am = canonical_actions.fully_private_observation(
        observer="Bob", P=H, agents=AGENTS, actual_truth_of_P=True
    )
    Sp = product(S, am)

    # Bob actually knows H.
    assert holds(Sp, K("Bob", H))
    # Alice believes H (her prior).
    assert holds(Sp, B("Alice", H))
    # Alice does not know H.
    assert not holds(Sp, K("Alice", H))
    # Alice believes Bob doesn't know which side is showing — because she
    # believes nothing happened (τ).
    from del_bench.kernel import And, Or

    knows_side = Or(K("Bob", H), K("Bob", T))
    assert holds(Sp, B("Alice", Not(knows_side)))
    # Bob knows Alice doesn't know.
    assert holds(Sp, K("Bob", Not(K("Alice", H))))
    # Bob believes Alice believes H (since Alice's belief tracker is unchanged).
    assert holds(Sp, B("Bob", B("Alice", H)))
    # Higher-order: Bob believes Alice believes Bob does NOT know.
    assert holds(Sp, B("Bob", B("Alice", Not(knows_side))))


# ---------------------------------------------------------------------------
# Example 3.3 + composite update: successful lie atop a private peek.
# Page 46 model S' ⊗ Σ_4. Bob privately saw H, then publicly claims K_b(T).
# Alice trusts him so now believes T.
# ---------------------------------------------------------------------------


def test_successful_lie_after_private_peek():
    S = make_prior()
    peek = canonical_actions.fully_private_observation(
        observer="Bob", P=H, agents=AGENTS, actual_truth_of_P=True
    )
    Sp = product(S, peek)
    # Bob now lies by claiming that he knows T while he actually knows H.
    lie = canonical_actions.successful_lie(
        speaker="Bob", lie_content=K("Bob", T), agents=AGENTS, actual_is_lie=True
    )
    M = product(Sp, lie)

    # Coin still H factually.
    assert holds(M, H)
    # Bob privately knows H (he never forgot).
    assert holds(M, K("Bob", H))
    assert holds(M, B("Bob", H))
    # Alice now believes T (trusts Bob's announcement).
    assert holds(M, B("Alice", T))
    assert not holds(M, B("Alice", H))
    # Paper-faithful announced content: Alice believes Bob knows T.
    assert holds(M, B("Alice", K("Bob", T)))
    # Bob believes Alice believes T (lie was intentional, he knows it landed).
    assert holds(M, B("Bob", B("Alice", T)))
    assert holds(M, B("Bob", B("Alice", K("Bob", T))))
    # Depth-3 deception probe: Bob believes that Alice believes that Bob
    # believes T. Alice's most plausible action is the honest one (trust);
    # in the honest world Bob would believe T (because his most plausible
    # state under that branch is the T-state).
    assert holds(M, B("Bob", B("Alice", B("Bob", T))))


def test_nested_helper():
    S = make_prior()
    f = nested(["B_Alice", "B_Bob"], H)
    assert holds(S, f)


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
    if failed:
        sys.exit(1)
    print(f"\nAll {len(tests)} kernel canonical tests passed.")
