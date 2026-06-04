"""Anti-lexicographic Action-Priority product update.

Paper sec. 3.2.2 (page 45). For each agent a:

    (s, σ) ≤_a (s', σ')   iff
        ( σ <_a σ'  AND  s ~_a s' )
        OR
        ( σ ≅_a σ'  AND  s ≤_a s' )

Indistinguishability (page 47):

    (s, σ) ~_a (s', σ')   iff   s ~_a s'  AND  σ ~_a σ'

Worlds in the updated model are pairs (s, σ) such that s |= pre(σ).
Valuation is inherited from s (purely doxastic actions, see paper sec. 3.1).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

from .evaluator import ActionInapplicableError, evaluate
from .models import EPM, Action, ActionModel, World


def _strict(am: ActionModel, ag: str, s: Action, t: Action) -> bool:
    return am.leq(ag, s, t) and not am.leq(ag, t, s)


def _equi(am: ActionModel, ag: str, s: Action, t: Action) -> bool:
    return am.leq(ag, s, t) and am.leq(ag, t, s)


def _action_indist(am: ActionModel, ag: str, s: Action, t: Action) -> bool:
    return am.cell(ag, s) == am.cell(ag, t)


def product(epm: EPM, am: ActionModel) -> EPM:
    if set(epm.agents) != set(am.agents):
        raise ValueError(f"agent sets differ: epm={epm.agents}, am={am.agents}")

    # 1. Build product worlds.
    new_worlds: list[World] = []
    pair_to_world: Dict[Tuple[World, Action], World] = {}
    world_to_pair: Dict[World, Tuple[World, Action]] = {}
    for w in epm.worlds:
        for sigma in am.actions:
            if evaluate(epm, w, sigma.precondition):
                nw = World(name=f"({w.name},{sigma.name})", valuation=w.valuation)
                new_worlds.append(nw)
                pair_to_world[(w, sigma)] = nw
                world_to_pair[nw] = (w, sigma)

    if not new_worlds:
        raise ActionInapplicableError("no (world, action) pair satisfies preconditions")

    if not evaluate(epm, epm.actual, am.actual.precondition):
        raise ActionInapplicableError(
            f"actual world {epm.actual.name} does not satisfy precondition of "
            f"actual action {am.actual.name}: {am.actual.precondition}"
        )

    new_actual = pair_to_world[(epm.actual, am.actual)]

    # 2 & 3. Compute new indist and plaus per agent.
    new_indist: Dict[str, FrozenSet[FrozenSet[World]]] = {}
    new_plaus: Dict[str, FrozenSet[Tuple[World, World]]] = {}

    for ag in epm.agents:
        # Indistinguishability via union-find over (s ~_a s') AND (σ ~_a σ')
        parent: Dict[World, World] = {w: w for w in new_worlds}

        def find(x: World) -> World:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: World, y: World) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for nw1 in new_worlds:
            for nw2 in new_worlds:
                w1, s1 = world_to_pair[nw1]
                w2, s2 = world_to_pair[nw2]
                # w1 ~_a w2 iff they share a cell in epm
                same_world_cell = epm.cell(ag, w1) == epm.cell(ag, w2)
                same_action_cell = _action_indist(am, ag, s1, s2)
                if same_world_cell and same_action_cell:
                    union(nw1, nw2)

        cells: Dict[World, set] = {}
        for nw in new_worlds:
            r = find(nw)
            cells.setdefault(r, set()).add(nw)
        new_indist[ag] = frozenset(frozenset(c) for c in cells.values())

        # Plausibility via Action-Priority
        pairs = set()
        for nw1 in new_worlds:
            for nw2 in new_worlds:
                w1, s1 = world_to_pair[nw1]
                w2, s2 = world_to_pair[nw2]
                cond1 = _strict(am, ag, s1, s2) and (
                    epm.cell(ag, w1) == epm.cell(ag, w2)
                )
                cond2 = _equi(am, ag, s1, s2) and epm.leq(ag, w1, w2)
                if cond1 or cond2:
                    pairs.add((nw1, nw2))
        new_plaus[ag] = frozenset(pairs)

    return EPM(
        worlds=tuple(new_worlds),
        agents=epm.agents,
        indist=new_indist,
        plaus=new_plaus,
        actual=new_actual,
    )
