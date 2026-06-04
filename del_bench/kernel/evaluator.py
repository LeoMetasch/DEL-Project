"""Formula evaluation at a world.

Semantics from the paper, sec. 2.2:
  K_a P    at w  iff  for all w' in cell_a(w), w' |= P.
  B_a P    at w  iff  for all w' in min_<=_a (cell_a(w)), w' |= P.
  B^Q_a P  at w  iff  for all w' in min_<=_a (cell_a(w) intersect [[Q]]),
                       w' |= P.
"""

from __future__ import annotations

from .formulas import And, Atom, B, ConditionalB, Formula, K, Not
from .models import EPM, World


class ActionInapplicableError(Exception):
    pass


def evaluate(epm: EPM, world: World, formula: Formula) -> bool:
    if isinstance(formula, Atom):
        return world.satisfies_atom(formula.name)
    if isinstance(formula, Not):
        return not evaluate(epm, world, formula.inner)
    if isinstance(formula, And):
        return evaluate(epm, world, formula.left) and evaluate(
            epm, world, formula.right
        )
    if isinstance(formula, K):
        cell = epm.cell(formula.agent, world)
        return all(evaluate(epm, w, formula.inner) for w in cell)
    if isinstance(formula, B):
        cell = epm.cell(formula.agent, world)
        mins = epm.min_plaus(formula.agent, cell)
        return all(evaluate(epm, w, formula.inner) for w in mins)
    if isinstance(formula, ConditionalB):
        cell = epm.cell(formula.agent, world)
        condition_worlds = frozenset(
            w for w in cell if evaluate(epm, w, formula.condition)
        )
        mins = epm.min_plaus(formula.agent, condition_worlds)
        return all(evaluate(epm, w, formula.inner) for w in mins)
    raise TypeError(f"unknown formula type: {type(formula).__name__}")


def holds(epm: EPM, formula: Formula) -> bool:
    """Evaluate at the actual world."""
    return evaluate(epm, epm.actual, formula)
