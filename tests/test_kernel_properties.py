"""Generated-model checks against direct encodings of the paper's semantics."""

from __future__ import annotations

import sys
from itertools import product as cartesian_product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from del_bench.kernel import (
    Action,
    ActionInapplicableError,
    ActionModel,
    And,
    Atom,
    B,
    ConditionalB,
    EPM,
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


def _same_cell(partition, left, right) -> bool:
    return any(left in cell and right in cell for cell in partition)


def _strict(relation, left, right) -> bool:
    return (left, right) in relation and (right, left) not in relation


def _equi(relation, left, right) -> bool:
    return (left, right) in relation and (right, left) in relation


def _partitions_of_two(items):
    left, right = items
    return (
        frozenset({frozenset({left, right})}),
        frozenset({frozenset({left}), frozenset({right})}),
    )


def _relation_from_ranks(cells, ranks):
    return frozenset(
        (left, right)
        for cell in cells
        for left in cell
        for right in cell
        if ranks[left] <= ranks[right]
    )


def _reference_evaluate(model: EPM, world: World, formula) -> bool:
    if isinstance(formula, Atom):
        return formula.name in world.valuation
    if isinstance(formula, Not):
        return not _reference_evaluate(model, world, formula.inner)
    if isinstance(formula, And):
        return _reference_evaluate(model, world, formula.left) and _reference_evaluate(
            model, world, formula.right
        )
    if isinstance(formula, K):
        return all(
            _reference_evaluate(model, candidate, formula.inner)
            for candidate in model.cell(formula.agent, world)
        )
    if isinstance(formula, B):
        cell = model.cell(formula.agent, world)
        minima = {
            candidate
            for candidate in cell
            if all((candidate, other) in model.plaus[formula.agent] for other in cell)
        }
        return all(
            _reference_evaluate(model, candidate, formula.inner) for candidate in minima
        )
    if isinstance(formula, ConditionalB):
        cell = model.cell(formula.agent, world)
        condition_worlds = {
            candidate
            for candidate in cell
            if _reference_evaluate(model, candidate, formula.condition)
        }
        minima = {
            candidate
            for candidate in condition_worlds
            if all(
                (candidate, other) in model.plaus[formula.agent]
                for other in condition_worlds
            )
        }
        return all(
            _reference_evaluate(model, candidate, formula.inner) for candidate in minima
        )
    raise TypeError(type(formula))


def test_evaluator_matches_direct_semantics_on_generated_models():
    atoms = [Atom("p"), Atom("q")]
    level_zero = atoms + [Not(atom) for atom in atoms] + [And(atoms[0], atoms[1])]
    level_one = (
        level_zero
        + [K("a", formula) for formula in level_zero]
        + [B("a", formula) for formula in level_zero]
        + [ConditionalB("a", atoms[0], formula) for formula in level_zero]
    )
    formulas = (
        level_one
        + [K("a", formula) for formula in level_one]
        + [B("a", formula) for formula in level_one]
        + [ConditionalB("a", atoms[1], formula) for formula in level_one]
    )

    valuations = (
        frozenset(),
        frozenset({"p"}),
        frozenset({"q"}),
        frozenset({"p", "q"}),
    )
    checks = 0
    for values in cartesian_product(valuations, repeat=2):
        worlds = (World("w0", values[0]), World("w1", values[1]))
        for cells in _partitions_of_two(worlds):
            for raw_ranks in cartesian_product(range(2), repeat=2):
                ranks = {worlds[index]: raw_ranks[index] for index in range(2)}
                model = EPM(
                    worlds,
                    ("a",),
                    {"a": cells},
                    {"a": _relation_from_ranks(cells, ranks)},
                    worlds[0],
                )
                for world, formula in cartesian_product(worlds, formulas):
                    assert evaluate(model, world, formula) == _reference_evaluate(
                        model, world, formula
                    )
                    checks += 1
    assert checks == 20480


def test_product_matches_direct_action_priority_rule_on_generated_models():
    checks = 0
    for values in cartesian_product((frozenset(), frozenset({"p"})), repeat=2):
        worlds = (World("w0", values[0]), World("w1", values[1]))
        for world_cells in _partitions_of_two(worlds):
            for raw_world_ranks in cartesian_product(range(2), repeat=2):
                world_ranks = {
                    worlds[index]: raw_world_ranks[index] for index in range(2)
                }
                world_relation = _relation_from_ranks(world_cells, world_ranks)
                model = EPM(
                    worlds,
                    ("a",),
                    {"a": world_cells},
                    {"a": world_relation},
                    worlds[0],
                )
                actions = (
                    Action("x", Atom("p")),
                    Action("y", Not(Atom("p"))),
                )
                for action_cells in _partitions_of_two(actions):
                    for raw_action_ranks in cartesian_product(range(2), repeat=2):
                        action_ranks = {
                            actions[index]: raw_action_ranks[index]
                            for index in range(2)
                        }
                        action_relation = _relation_from_ranks(
                            action_cells, action_ranks
                        )
                        actual_action = actions[0] if "p" in values[0] else actions[1]
                        action_model = ActionModel(
                            actions,
                            ("a",),
                            {"a": action_cells},
                            {"a": action_relation},
                            actual_action,
                        )
                        updated = product(model, action_model)
                        source_by_output = {
                            f"({world.name},{action.name})": (world, action)
                            for world in worlds
                            for action in actions
                            if evaluate(model, world, action.precondition)
                        }
                        expected_plaus = set()
                        expected_indist = set()
                        for left, right in cartesian_product(updated.worlds, repeat=2):
                            left_world, left_action = source_by_output[left.name]
                            right_world, right_action = source_by_output[right.name]
                            if (
                                _strict(action_relation, left_action, right_action)
                                and _same_cell(world_cells, left_world, right_world)
                            ) or (
                                _equi(action_relation, left_action, right_action)
                                and (left_world, right_world) in world_relation
                            ):
                                expected_plaus.add((left, right))
                            if _same_cell(
                                world_cells, left_world, right_world
                            ) and _same_cell(action_cells, left_action, right_action):
                                expected_indist.add((left, right))

                        actual_indist = {
                            (left, right)
                            for left, right in cartesian_product(
                                updated.worlds, repeat=2
                            )
                            if _same_cell(updated.indist["a"], left, right)
                        }
                        assert set(updated.plaus["a"]) == expected_plaus
                        assert actual_indist == expected_indist
                        checks += 1
    assert checks == 256


def test_public_announcement_filters_worlds_without_changing_facts():
    heads = World("heads", frozenset({"H"}))
    tails = World("tails", frozenset({"T"}))
    worlds = (heads, tails)
    model = EPM(
        worlds,
        ("a",),
        {"a": total_partition(worlds)},
        {"a": total_preorder_pairs({heads: 1, tails: 0})},
        heads,
    )
    updated = product(model, canonical_actions.public_announcement(Atom("H"), ("a",)))
    assert len(updated.worlds) == 1
    assert updated.actual.valuation == frozenset({"H"})
    assert holds(updated, K("a", Atom("H")))


def test_soft_and_unreliable_announcements_implement_distinct_policies():
    heads = World("heads", frozenset({"H"}))
    tails = World("tails", frozenset({"T"}))
    worlds = (heads, tails)
    model = EPM(
        worlds,
        ("a",),
        {"a": total_partition(worlds)},
        {"a": total_preorder_pairs({heads: 0, tails: 1})},
        heads,
    )

    soft = product(
        model,
        canonical_actions.lexicographic_upgrade(
            Atom("T"), ("a",), actual_truth_of_P=False
        ),
    )
    unreliable = product(
        model,
        canonical_actions.unreliable_announcement(
            Atom("T"), ("a",), actual_truth_of_P=False
        ),
    )

    assert holds(soft, Atom("H"))
    assert holds(soft, B("a", Atom("T")))
    assert holds(unreliable, Atom("H"))
    assert holds(unreliable, B("a", Atom("H")))


def test_product_rejects_inapplicable_actual_action():
    world = World("w", frozenset({"p"}))
    model = EPM(
        (world,),
        ("a",),
        {"a": total_partition((world,))},
        {"a": frozenset({(world, world)})},
        world,
    )
    possible = Action("possible", Atom("p"))
    impossible = Action("impossible", Not(Atom("p")))
    actions = (possible, impossible)
    action_model = ActionModel(
        actions,
        ("a",),
        {"a": total_partition(actions)},
        {"a": total_preorder_pairs({possible: 0, impossible: 0})},
        impossible,
    )
    try:
        product(model, action_model)
    except ActionInapplicableError:
        pass
    else:
        raise AssertionError("inapplicable actual action should fail")


def test_validators_reject_empty_cells_unknown_pairs_and_duplicate_agents():
    world = World("w", frozenset())
    ghost_world = World("ghost", frozenset())
    action = Action("a", Atom("p"))
    ghost_action = Action("ghost", Atom("p"))

    invalid_models = (
        lambda: EPM(
            (world,),
            ("a", "a"),
            {"a": total_partition((world,))},
            {"a": frozenset({(world, world)})},
            world,
        ),
        lambda: EPM(
            (world,),
            ("a",),
            {"a": total_partition((world,))},
            {"a": frozenset({(world, world), (ghost_world, ghost_world)})},
            world,
        ),
        lambda: ActionModel(
            (action,),
            ("a", "a"),
            {"a": total_partition((action,))},
            {"a": frozenset({(action, action)})},
            action,
        ),
        lambda: ActionModel(
            (action,),
            ("a",),
            {"a": frozenset({frozenset({action}), frozenset()})},
            {"a": frozenset({(action, action)})},
            action,
        ),
        lambda: ActionModel(
            (action,),
            ("a",),
            {"a": total_partition((action,))},
            {"a": frozenset({(action, action), (ghost_action, ghost_action)})},
            action,
        ),
    )
    for make_invalid in invalid_models:
        try:
            make_invalid()
        except ValueError:
            pass
        else:
            raise AssertionError("malformed model should fail validation")


if __name__ == "__main__":
    import traceback

    failed = 0
    for key, value in list(globals().items()):
        if key.startswith("test_"):
            try:
                value()
                print(f"PASS {key}")
            except Exception as exc:
                failed += 1
                print(f"FAIL {key}: {exc}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
