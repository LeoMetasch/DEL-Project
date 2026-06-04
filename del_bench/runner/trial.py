"""Run a single (scenario × model × strategy × paraphrase seed) trial."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional

from del_bench.llm.openrouter import OpenRouterClient
from del_bench.nl.parser import ParseStatus, parse_response
from del_bench.nl.renderer import render
from del_bench.nl.strategies import PromptStrategy
from del_bench.scenarios.schema import Scenario, gold_choice


@dataclass
class ProbeResult:
    probe_id: str
    depth: int
    nesting: tuple
    is_syntactic_control: bool
    raw_response: str
    parsed_label: Optional[str]
    gold_label: str
    correct: bool
    parse_status: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    reasoning: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TrialResult:
    trial_id: str
    scenario_id: str
    model: str
    strategy: str
    seed: int
    timestamp_iso: str
    probes: List[ProbeResult] = field(default_factory=list)


def run_trial(
    scenario: Scenario,
    model: str,
    strategy: PromptStrategy,
    seed: int,
    client: OpenRouterClient,
) -> TrialResult:
    trial_id = f"{scenario.id}|{model}|{strategy.name}|seed{seed}"
    result = TrialResult(
        trial_id=trial_id,
        scenario_id=scenario.id,
        model=model,
        strategy=strategy.name,
        seed=seed,
        timestamp_iso=datetime.now(timezone.utc).isoformat(),
    )

    for probe in scenario.probes:
        rp = render(scenario, probe, strategy, seed=seed)
        completion = client.complete(
            model=model,
            system=rp.system,
            user=rp.user,
            response_format=rp.response_format,
        )
        parsed = parse_response(completion.raw_text, rp.choice_labels)
        gold = gold_choice(probe, scenario.formal_model)
        correct = parsed.status == ParseStatus.OK and parsed.chosen_label == gold.label

        result.probes.append(
            ProbeResult(
                probe_id=probe.id,
                depth=probe.depth,
                nesting=probe.nesting,
                is_syntactic_control=probe.syntactic_control,
                raw_response=completion.raw_text,
                parsed_label=parsed.chosen_label,
                gold_label=gold.label,
                correct=correct,
                parse_status=parsed.status.value,
                latency_s=completion.latency_s,
                tokens_in=completion.tokens_in,
                tokens_out=completion.tokens_out,
                reasoning=parsed.reasoning,
                error=completion.error,
            )
        )

    return result


def trial_to_jsonable(trial: TrialResult) -> dict:
    return {
        "trial_id": trial.trial_id,
        "scenario_id": trial.scenario_id,
        "model": trial.model,
        "strategy": trial.strategy,
        "seed": trial.seed,
        "timestamp_iso": trial.timestamp_iso,
        "probes": [
            {
                **asdict(p),
                "nesting": list(p.nesting),
            }
            for p in trial.probes
        ],
    }
