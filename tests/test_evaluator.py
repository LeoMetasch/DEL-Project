"""Direct evaluator unit tests on hand-built tiny models."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from del_bench.kernel import (
    And,
    Atom,
    B,
    ConditionalB,
    EPM,
    K,
    Not,
    Or,
    World,
    evaluate,
    holds,
    total_partition,
    total_preorder_pairs,
)
from del_bench.kernel.formulas import TOP


def make_two_world(actual_name: str = "s") -> EPM:
    s = World("s", frozenset({"H"}))
    t = World("t", frozenset({"T"}))
    worlds = (s, t)
    indist = {"a": total_partition(worlds)}
    plaus = {"a": total_preorder_pairs({s: 0, t: 1})}
    actual = s if actual_name == "s" else t
    return EPM(worlds=worlds, agents=("a",), indist=indist, plaus=plaus, actual=actual)


def test_atom_evaluation():
    M = make_two_world()
    s = M.worlds[0]
    t = M.worlds[1]
    assert evaluate(M, s, Atom("H"))
    assert not evaluate(M, s, Atom("T"))
    assert evaluate(M, t, Atom("T"))


def test_boolean_connectives():
    M = make_two_world()
    s = M.worlds[0]
    assert evaluate(M, s, Not(Atom("T")))
    assert evaluate(M, s, And(Atom("H"), Not(Atom("T"))))
    assert evaluate(M, s, Or(Atom("H"), Atom("T")))
    assert not evaluate(M, s, And(Atom("H"), Atom("T")))


def test_knowledge_quantifies_universally_in_cell():
    M = make_two_world()
    # Cell is {s, t}, so K_a(H) requires H true at both, fails.
    assert not holds(M, K("a", Atom("H")))
    # K_a(H ∨ T) holds since at least one of H or T is true at each world.
    assert holds(M, K("a", Or(Atom("H"), Atom("T"))))


def test_belief_uses_minimal_states_only():
    M = make_two_world()
    # s is more plausible than t; B_a(H) should hold.
    assert holds(M, B("a", Atom("H")))
    assert not holds(M, B("a", Atom("T")))


def test_belief_is_constant_within_cell():
    M = make_two_world()
    s = M.worlds[0]
    t = M.worlds[1]
    # B_a(H) is the same proposition at every world in the cell.
    assert evaluate(M, s, B("a", Atom("H")))
    assert evaluate(M, t, B("a", Atom("H")))


def test_conditional_belief_uses_minimal_condition_worlds_only():
    M = make_two_world()
    assert holds(M, ConditionalB("a", Atom("T"), Atom("T")))
    assert not holds(M, ConditionalB("a", Atom("T"), Atom("H")))


def test_unconditional_belief_is_conditional_belief_given_top():
    M = make_two_world()
    assert holds(M, B("a", Atom("H"))) == holds(M, ConditionalB("a", TOP, Atom("H")))


def test_conditional_belief_is_vacuously_true_for_impossible_condition():
    M = make_two_world()
    impossible = And(Atom("H"), Atom("T"))
    assert holds(M, ConditionalB("a", impossible, Atom("H")))
    assert holds(M, ConditionalB("a", impossible, Atom("T")))


if __name__ == "__main__":
    import traceback

    failed = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            try:
                v()
                print(f"PASS {k}")
            except Exception as e:
                failed += 1
                print(f"FAIL {k}: {e}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
