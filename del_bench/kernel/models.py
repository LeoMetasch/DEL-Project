"""EPM and ActionModel data structures + validation.

Plausibility is stored as a set of ordered pairs (s, t) representing s <=_a t,
i.e. "s is at least as plausible as t" (paper sec. 2.1, p.20). Reflexive pairs
must be present. The minimal elements of a set are the most plausible ones.

Indistinguishability is stored as a partition (set of frozensets of worlds).

Coherence requirement (paper sec. 2.2): if s <=_a t then s ~_a t. Within each
~_a-equivalence class the restriction of <=_a is a well-preorder (for finite
models: connected, transitive, reflexive).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Mapping, Tuple

from .formulas import Formula

# ---------------------------------------------------------------------------
# Worlds and EPMs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class World:
    name: str
    valuation: FrozenSet[str]

    def satisfies_atom(self, atom: str) -> bool:
        return atom in self.valuation

    def __repr__(self) -> str:
        v = ",".join(sorted(self.valuation)) or "∅"
        return f"<{self.name}:{v}>"


def _is_partition(
    cells: FrozenSet[FrozenSet[World]], universe: Tuple[World, ...]
) -> Tuple[bool, str]:
    seen: set = set()
    for cell in cells:
        if not cell:
            return False, "empty cell"
        for w in cell:
            if w in seen:
                return False, f"world {w.name} appears in multiple cells"
            seen.add(w)
    missing = set(universe) - seen
    if missing:
        return False, f"worlds not in any cell: {[w.name for w in missing]}"
    extra = seen - set(universe)
    if extra:
        return False, f"unknown worlds in cells: {[w.name for w in extra]}"
    return True, ""


def _check_preorder(
    pairs: FrozenSet[Tuple[World, World]],
    universe: Tuple[World, ...],
    cells: FrozenSet[FrozenSet[World]],
    agent: str,
) -> None:
    """Reflexive on universe, transitive, coherent with cells, locally connected."""
    pairset = set(pairs)
    # Reflexivity
    for w in universe:
        if (w, w) not in pairset:
            raise ValueError(f"plaus[{agent}] missing reflexive pair for {w.name}")
    # Coherence with cells: if s <=_a t then s ~_a t (same cell)
    cell_of: Dict[World, FrozenSet[World]] = {}
    for cell in cells:
        for w in cell:
            cell_of[w] = cell
    for s, t in pairset:
        if s not in cell_of or t not in cell_of:
            raise ValueError(
                f"plaus[{agent}] contains unknown world pair ({s.name},{t.name})"
            )
        if cell_of[s] != cell_of[t]:
            raise ValueError(
                f"plaus[{agent}] has ({s.name},{t.name}) but they are in different ~_{agent}-cells"
            )
    # Transitivity
    for s, t in pairset:
        for t2, u in pairset:
            if t == t2 and (s, u) not in pairset:
                raise ValueError(
                    f"plaus[{agent}] not transitive: {s.name}<={t.name}<={u.name} but {s.name}<={u.name} missing"
                )
    # Local connectedness: within each cell every pair is comparable
    for cell in cells:
        for s in cell:
            for t in cell:
                if (s, t) not in pairset and (t, s) not in pairset:
                    raise ValueError(
                        f"plaus[{agent}] not connected within cell containing {s.name},{t.name}"
                    )


@dataclass(frozen=True)
class EPM:
    worlds: Tuple[World, ...]
    agents: Tuple[str, ...]
    indist: Mapping[str, FrozenSet[FrozenSet[World]]]
    plaus: Mapping[str, FrozenSet[Tuple[World, World]]]
    actual: World

    def __post_init__(self) -> None:
        if self.actual not in self.worlds:
            raise ValueError(f"actual world {self.actual.name} not in worlds")
        if len(set(self.agents)) != len(self.agents):
            raise ValueError("agent names are not unique")
        if len(set(w.name for w in self.worlds)) != len(self.worlds):
            raise ValueError("world names are not unique")
        for a in self.agents:
            if a not in self.indist:
                raise ValueError(f"missing indist for agent {a}")
            if a not in self.plaus:
                raise ValueError(f"missing plaus for agent {a}")
            ok, msg = _is_partition(self.indist[a], self.worlds)
            if not ok:
                raise ValueError(f"indist[{a}] not a partition: {msg}")
            _check_preorder(self.plaus[a], self.worlds, self.indist[a], a)

    def cell(self, agent: str, w: World) -> FrozenSet[World]:
        for c in self.indist[agent]:
            if w in c:
                return c
        raise KeyError(f"world {w} not in any cell for agent {agent}")

    def leq(self, agent: str, s: World, t: World) -> bool:
        return (s, t) in self.plaus[agent]

    def min_plaus(self, agent: str, subset: FrozenSet[World]) -> FrozenSet[World]:
        """<=_a-minimal elements of subset (the most plausible)."""
        if not subset:
            return frozenset()
        out = set()
        for s in subset:
            is_min = True
            for t in subset:
                # s minimal iff for all t in subset, s <=_a t
                if not self.leq(agent, s, t):
                    is_min = False
                    break
            if is_min:
                out.add(s)
        return frozenset(out)


# ---------------------------------------------------------------------------
# Actions and ActionModels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    name: str
    precondition: Formula

    def __repr__(self) -> str:
        return f"<{self.name}|{self.precondition}>"


@dataclass(frozen=True)
class ActionModel:
    actions: Tuple[Action, ...]
    agents: Tuple[str, ...]
    indist: Mapping[str, FrozenSet[FrozenSet[Action]]]
    plaus: Mapping[str, FrozenSet[Tuple[Action, Action]]]
    actual: Action

    def __post_init__(self) -> None:
        if self.actual not in self.actions:
            raise ValueError(f"actual action {self.actual.name} not in actions")
        if len(set(self.agents)) != len(self.agents):
            raise ValueError("agent names are not unique")
        if len(set(a.name for a in self.actions)) != len(self.actions):
            raise ValueError("action names are not unique")
        for ag in self.agents:
            if ag not in self.indist:
                raise ValueError(f"missing indist for agent {ag}")
            if ag not in self.plaus:
                raise ValueError(f"missing plaus for agent {ag}")
            # validate partition over actions
            seen = set()
            for cell in self.indist[ag]:
                if not cell:
                    raise ValueError(f"indist[{ag}] contains an empty cell")
                for a in cell:
                    if a in seen:
                        raise ValueError(f"action {a.name} in multiple cells for {ag}")
                    seen.add(a)
            if seen != set(self.actions):
                raise ValueError(f"indist[{ag}] does not cover all actions")
            # validate preorder
            pairs = self.plaus[ag]
            for a in self.actions:
                if (a, a) not in pairs:
                    raise ValueError(
                        f"action plaus[{ag}] missing reflexive ({a.name},{a.name})"
                    )
            cell_of = {}
            for cell in self.indist[ag]:
                for a in cell:
                    cell_of[a] = cell
            for s, t in pairs:
                if s not in cell_of or t not in cell_of:
                    raise ValueError(
                        f"action plaus[{ag}] contains unknown action pair ({s.name},{t.name})"
                    )
                if cell_of[s] != cell_of[t]:
                    raise ValueError(
                        f"action plaus[{ag}]: ({s.name},{t.name}) cross cells"
                    )
            for s, t in pairs:
                for t2, u in pairs:
                    if t == t2 and (s, u) not in pairs:
                        raise ValueError(
                            f"action plaus[{ag}] not transitive at {s.name},{t.name},{u.name}"
                        )
            for cell in self.indist[ag]:
                for s in cell:
                    for t in cell:
                        if (s, t) not in pairs and (t, s) not in pairs:
                            raise ValueError(
                                f"action plaus[{ag}] not connected in cell containing {s.name},{t.name}"
                            )

    def leq(self, agent: str, s: Action, t: Action) -> bool:
        return (s, t) in self.plaus[agent]

    def cell(self, agent: str, a: Action) -> FrozenSet[Action]:
        for c in self.indist[agent]:
            if a in c:
                return c
        raise KeyError(a)


# ---------------------------------------------------------------------------
# Builders for the common case of total preorders within cells
# ---------------------------------------------------------------------------


def total_preorder_pairs(
    ranks: Mapping[object, int],
) -> FrozenSet[Tuple[object, object]]:
    """Given a rank assignment (lower rank = more plausible), produce all pairs
    (s, t) such that rank(s) <= rank(t). Equal ranks yield both directions
    (equiplausibility)."""
    pairs = set()
    items = list(ranks.items())
    for s, rs in items:
        for t, rt in items:
            if rs <= rt:
                pairs.add((s, t))
    return frozenset(pairs)


def discrete_partition(items: Tuple[object, ...]) -> FrozenSet[FrozenSet[object]]:
    """Each item in its own cell (full distinguishability)."""
    return frozenset(frozenset({x}) for x in items)


def total_partition(items: Tuple[object, ...]) -> FrozenSet[FrozenSet[object]]:
    """All items in one cell (full indistinguishability)."""
    return frozenset({frozenset(items)})
