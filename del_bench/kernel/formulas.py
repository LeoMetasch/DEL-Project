"""Formula AST.

The constructors needed for the benchmark. ``Or`` and ``Implies`` are
derivable; safe belief remains out of scope.

References to the paper (Baltag & Smets, "A Qualitative Theory of Dynamic
Interactive Belief Revision", LOFT7 2008):
  - K_a P  is the standard S5 knowledge modality, [~_a]P (sec. 2.2).
  - B_a P  is belief, the Kripke modality for the doxastic accessibility
            relation that selects <=_a-minimal worlds in the agent's cell.
  - B^Q_a P is conditional belief: P is believed after revision with Q.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union


@dataclass(frozen=True)
class Atom:
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Not:
    inner: "Formula"

    def __repr__(self) -> str:
        return f"¬{self.inner}"


@dataclass(frozen=True)
class And:
    left: "Formula"
    right: "Formula"

    def __repr__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True)
class K:
    agent: str
    inner: "Formula"

    def __repr__(self) -> str:
        return f"K_{self.agent}({self.inner})"


@dataclass(frozen=True)
class B:
    agent: str
    inner: "Formula"

    def __repr__(self) -> str:
        return f"B_{self.agent}({self.inner})"


@dataclass(frozen=True)
class ConditionalB:
    agent: str
    condition: "Formula"
    inner: "Formula"

    def __repr__(self) -> str:
        return f"B^({self.condition})_{self.agent}({self.inner})"


Formula = Union[Atom, Not, And, K, B, ConditionalB]


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

TOP = Not(And(Atom("__top__"), Not(Atom("__top__"))))  # tautology placeholder


def Or(p: Formula, q: Formula) -> Formula:
    return Not(And(Not(p), Not(q)))


def Implies(p: Formula, q: Formula) -> Formula:
    return Or(Not(p), q)


def nested(modalities: Iterable[str], inner: Formula) -> Formula:
    """Build e.g. nested(["B_b", "B_a"], Atom("H")) = B(b, B(a, H))."""
    out = inner
    for m in reversed(list(modalities)):
        kind, _, agent = m.partition("_")
        if kind == "B":
            out = B(agent, out)
        elif kind == "K":
            out = K(agent, out)
        else:
            raise ValueError(f"unknown modality token: {m!r}")
    return out
