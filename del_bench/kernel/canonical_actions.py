"""Canonical action models from the paper.

- public_announcement (Σ_!P, sec. 3.3 first subsection): single action, all
  agents distinguish nothing because there is only one action; eliminates
  ¬P-worlds.

- fair_game_observation (Σ_2, Example 3.1, page 42): observer sees the
  truth value of P; non-observers know an observation took place but get
  no information about P, and consider the two possible outcomes
  equiplausible.

- fully_private_observation (Σ_3, Example 3.2, page 42): observer learns
  P; non-observers believe nothing is happening (the τ "trivial" action),
  yet the σ_P / σ_¬P alternatives remain epistemically possible. Among
  those, σ_P is more plausible to non-observers, matching the prior
  preference for P (paper-faithful default).

- successful_lie (Σ_4, Example 3.3, page 43): the speaker publicly
  announces some content C; listeners trust the speaker and consider the
  "honest" alternative more plausible; speakers know which action they
  are performing. Actual action is the dishonest one.
"""

from __future__ import annotations

from typing import Tuple

from .formulas import And, Atom, Formula, Not
from .models import (
    Action,
    ActionModel,
    discrete_partition,
    total_partition,
    total_preorder_pairs,
)

# ---------------------------------------------------------------------------


def public_announcement(P: Formula, agents: Tuple[str, ...]) -> ActionModel:
    """Single action !P with precondition P. Indist trivially singleton."""
    a = Action(name="!", precondition=P)
    actions = (a,)
    indist = {ag: frozenset({frozenset({a})}) for ag in agents}
    plaus = {ag: frozenset({(a, a)}) for ag in agents}
    return ActionModel(
        actions=actions, agents=agents, indist=indist, plaus=plaus, actual=a
    )


# ---------------------------------------------------------------------------


def fair_game_observation(
    observer: str,
    P: Formula,
    agents: Tuple[str, ...],
    actual_truth_of_P: bool = True,
) -> ActionModel:
    """Observer learns whether P holds; non-observers see only that the
    observation occurred.

    Σ_2 in the paper has two actions σ_P (precondition P) and σ_¬P
    (precondition ¬P), equiplausible to non-observers and distinguished by
    the observer.
    """
    sP = Action(name=f"σ_{P}", precondition=P)
    sNP = Action(name=f"σ_¬{P}", precondition=Not(P))
    actions = (sP, sNP)
    indist = {}
    plaus = {}
    for ag in agents:
        if ag == observer:
            indist[ag] = discrete_partition(actions)
            plaus[ag] = frozenset({(sP, sP), (sNP, sNP)})
        else:
            indist[ag] = total_partition(actions)
            # equiplausible: same rank
            plaus[ag] = total_preorder_pairs({sP: 0, sNP: 0})
    actual = sP if actual_truth_of_P else sNP
    return ActionModel(
        actions=actions, agents=agents, indist=indist, plaus=plaus, actual=actual
    )


# ---------------------------------------------------------------------------


def fully_private_observation(
    observer: str,
    P: Formula,
    agents: Tuple[str, ...],
    actual_truth_of_P: bool = True,
) -> ActionModel:
    """Observer privately learns whether P holds; non-observers believe
    nothing happened.

    Σ_3 in the paper: actions σ_P (pre P), σ_¬P (pre ¬P), τ (pre ⊤).
    Non-observer plausibility (Example 3.2): τ <_a σ_P <_a σ_¬P (τ most
    plausible; σ_P preferred to σ_¬P, matching prior preference for P).
    Observer distinguishes all three.
    """
    sP = Action(name=f"σ_{P}", precondition=P)
    sNP = Action(name=f"σ_¬{P}", precondition=Not(P))
    # ⊤ as P ∨ ¬P, kept simple as Not(And(p, Not p)) on a fresh atom; here we use the same P
    # to remain syntactically minimal: τ's precondition is the tautology P ∨ ¬P.
    tau_pre: Formula = Not(And(P, Not(P)))  # always true
    tau = Action(name="τ", precondition=tau_pre)
    actions = (sP, sNP, tau)
    indist = {}
    plaus = {}
    for ag in agents:
        if ag == observer:
            indist[ag] = discrete_partition(actions)
            plaus[ag] = frozenset({(sP, sP), (sNP, sNP), (tau, tau)})
        else:
            indist[ag] = total_partition(actions)
            # Lower rank = more plausible. τ most plausible (rank 0),
            # σ_P next (rank 1), σ_¬P least plausible (rank 2).
            plaus[ag] = total_preorder_pairs({tau: 0, sP: 1, sNP: 2})
    actual = sP if actual_truth_of_P else sNP
    return ActionModel(
        actions=actions, agents=agents, indist=indist, plaus=plaus, actual=actual
    )


# ---------------------------------------------------------------------------


def successful_lie(
    speaker: str,
    lie_content: Formula,
    agents: Tuple[str, ...],
    actual_is_lie: bool = True,
) -> ActionModel:
    """Public announcement that may be honest or a lie. Listeners trust the
    speaker; speakers know what they are saying.

    Σ_4 in the paper: two actions, λ_honest with precondition C (the lie
    content) and λ_dishonest with precondition ¬C. Listeners are
    indistinguishable between them and consider λ_honest more plausible.

    ``lie_content`` is the full announced proposition. For the paper's
    Example 3.3, where Bob claims to know that the coin is Tails, pass
    ``K("Bob", T)`` rather than the weaker factual proposition ``T``.
    """
    honest = Action(name=f"λ_honest({lie_content})", precondition=lie_content)
    dishonest = Action(
        name=f"λ_dishonest({lie_content})", precondition=Not(lie_content)
    )
    actions = (honest, dishonest)
    indist = {}
    plaus = {}
    for ag in agents:
        if ag == speaker:
            indist[ag] = discrete_partition(actions)
            plaus[ag] = frozenset({(honest, honest), (dishonest, dishonest)})
        else:
            indist[ag] = total_partition(actions)
            # honest more plausible than dishonest
            plaus[ag] = total_preorder_pairs({honest: 0, dishonest: 1})
    actual = dishonest if actual_is_lie else honest
    return ActionModel(
        actions=actions, agents=agents, indist=indist, plaus=plaus, actual=actual
    )


# ---------------------------------------------------------------------------


def lexicographic_upgrade(
    P: Formula,
    agents: Tuple[str, ...],
    actual_truth_of_P: bool = True,
) -> ActionModel:
    """Soft public announcement / lexicographic upgrade (§3.3, page 47-48).

    Two actions: ⇑(P) with precondition P and ⇓(P) with precondition ¬P.
    All agents see both as indistinguishable. ⇑ is strictly more plausible
    than ⇓, so P-worlds get promoted but ¬P-worlds are not eliminated.
    """
    up = Action(name=f"⇑({P})", precondition=P)
    down = Action(name=f"⇓({P})", precondition=Not(P))
    actions = (up, down)
    indist = {ag: total_partition(actions) for ag in agents}
    plaus = {ag: total_preorder_pairs({up: 0, down: 1}) for ag in agents}
    actual = up if actual_truth_of_P else down
    return ActionModel(
        actions=actions, agents=agents, indist=indist, plaus=plaus, actual=actual
    )


# ---------------------------------------------------------------------------


def unreliable_announcement(
    P: Formula,
    agents: Tuple[str, ...],
    actual_truth_of_P: bool = True,
) -> ActionModel:
    """Completely unreliable announcement (Example 3.5, page 44).

    Two actions σ (precondition P) and σ' (precondition ¬P), equiplausible
    for all agents and indistinguishable. Agents don't trust the source at
    all, so original beliefs should remain unchanged.
    """
    sigma = Action(name=f"σ({P})", precondition=P)
    sigma_prime = Action(name=f"σ'({P})", precondition=Not(P))
    actions = (sigma, sigma_prime)
    indist = {ag: total_partition(actions) for ag in agents}
    plaus = {ag: total_preorder_pairs({sigma: 0, sigma_prime: 0}) for ag in agents}
    actual = sigma if actual_truth_of_P else sigma_prime
    return ActionModel(
        actions=actions, agents=agents, indist=indist, plaus=plaus, actual=actual
    )
