"""Cross-product runner. Sequential by default for the pilot — every probe
is one network call so a small concurrency parameter is enough."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from del_bench.llm.openrouter import OpenRouterClient
from del_bench.nl.strategies import PromptStrategy
from del_bench.runner.trial import TrialResult, run_trial, trial_to_jsonable
from del_bench.scenarios.registry import SCENARIOS


def already_completed(results_path: Path) -> set:
    """Read existing JSONL and collect completed trial_ids."""
    if not results_path.exists():
        return set()
    done = set()
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = obj.get("trial_id")
            if tid:
                done.add(tid)
    return done


def run_grid(
    scenario_ids: Iterable[str],
    models: Iterable[str],
    strategies: Iterable[PromptStrategy],
    seeds: Iterable[int],
    client: OpenRouterClient,
    results_path: Path,
    max_workers: int = 4,
    on_complete=None,
) -> List[TrialResult]:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_completed(results_path)

    plan: List[Tuple[str, str, PromptStrategy, int]] = []
    for sid in scenario_ids:
        scenario = SCENARIOS[sid]
        for model in models:
            for strategy in strategies:
                for seed in seeds:
                    trial_id = f"{scenario.id}|{model}|{strategy.name}|seed{seed}"
                    if trial_id in done:
                        continue
                    plan.append((sid, model, strategy, seed))

    print(f"Plan: {len(plan)} trials ({len(done)} already complete, skipping).")

    out: List[TrialResult] = []
    if not plan:
        return out

    def _run_one(item):
        sid, model, strategy, seed = item
        return run_trial(SCENARIOS[sid], model, strategy, seed, client)

    with results_path.open("a") as fout:
        if max_workers <= 1:
            for item in plan:
                trial = _run_one(item)
                out.append(trial)
                fout.write(json.dumps(trial_to_jsonable(trial)) + "\n")
                fout.flush()
                if on_complete:
                    on_complete(trial)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_run_one, item): item for item in plan}
                for fut in as_completed(futures):
                    trial = fut.result()
                    out.append(trial)
                    fout.write(json.dumps(trial_to_jsonable(trial)) + "\n")
                    fout.flush()
                    if on_complete:
                        on_complete(trial)

    return out
