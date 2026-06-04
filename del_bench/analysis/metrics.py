"""Compute report metrics from a JSONL of TrialResult dicts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from del_bench.scenarios.registry import SCENARIOS


def load_trials(jsonl_path: Path) -> List[dict]:
    out = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + (z * z / total)
    centre = p + (z * z / (2 * total))
    margin = z * math.sqrt((p * (1 - p) / total) + (z * z / (4 * total * total)))
    return ((centre - margin) / denom, (centre + margin) / denom)


def run_coverage(trials: List[dict]) -> dict:
    """High-level coverage summary for a results file."""
    probe_records = [p for t in trials for p in t["probes"]]
    logical = [p for p in probe_records if not p["is_syntactic_control"]]
    syntactic = [p for p in probe_records if p["is_syntactic_control"]]
    failures = [p for p in probe_records if p["parse_status"] != "OK"]

    scenario_probe_counts: Dict[str, Tuple[int, int]] = {}
    for sid, scenario in SCENARIOS.items():
        logical_count = sum(1 for p in scenario.probes if not p.syntactic_control)
        syntactic_count = sum(1 for p in scenario.probes if p.syntactic_control)
        scenario_probe_counts[sid] = (logical_count, syntactic_count)

    return {
        "trials": len(trials),
        "scenario_count": len({t["scenario_id"] for t in trials}),
        "model_count": len({t["model"] for t in trials}),
        "strategy_count": len({t["strategy"] for t in trials}),
        "seeds": sorted({t.get("seed", 0) for t in trials}),
        "probe_records": len(probe_records),
        "logical_probe_records": len(logical),
        "syntactic_probe_records": len(syntactic),
        "parse_failures": len(failures),
        "registered_scenarios": len(SCENARIOS),
        "registered_total_probes": sum(len(s.probes) for s in SCENARIOS.values()),
        "registered_logical_probes": sum(v[0] for v in scenario_probe_counts.values()),
        "registered_syntactic_probes": sum(
            v[1] for v in scenario_probe_counts.values()
        ),
    }


def overall_accuracy(trials: List[dict]) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """(model, strategy) -> (correct, total). Excludes syntactic-control probes."""
    bucket: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
    for t in trials:
        key = (t["model"], t["strategy"])
        for p in t["probes"]:
            if p["is_syntactic_control"]:
                continue
            bucket[key][1] += 1
            if p["correct"]:
                bucket[key][0] += 1
    return {k: tuple(v) for k, v in bucket.items()}


def per_depth_accuracy(
    trials: List[dict],
) -> Dict[Tuple[str, str, int], Tuple[int, int]]:
    """(model, strategy, depth) -> (correct, total). Excludes syntactic-control probes."""
    bucket: Dict[Tuple[str, str, int], List[int]] = defaultdict(lambda: [0, 0])
    for t in trials:
        for p in t["probes"]:
            if p["is_syntactic_control"]:
                continue
            key = (t["model"], t["strategy"], p["depth"])
            bucket[key][1] += 1
            if p["correct"]:
                bucket[key][0] += 1
    return {k: tuple(v) for k, v in bucket.items()}


def per_action_type_accuracy(
    trials: List[dict],
) -> Dict[Tuple[str, str, str], Tuple[int, int]]:
    """(model, strategy, template_family) -> (correct, total). Excludes syntactic-control."""
    fam = {sid: s.template_family for sid, s in SCENARIOS.items()}
    bucket: Dict[Tuple[str, str, str], List[int]] = defaultdict(lambda: [0, 0])
    for t in trials:
        family = fam.get(t["scenario_id"], "?")
        for p in t["probes"]:
            if p["is_syntactic_control"]:
                continue
            key = (t["model"], t["strategy"], family)
            bucket[key][1] += 1
            if p["correct"]:
                bucket[key][0] += 1
    return {k: tuple(v) for k, v in bucket.items()}


def matched_pair_sensitivity(trials: List[dict]) -> Dict[Tuple[str, str, str], dict]:
    """For each (model, strategy, pair-key), report performance on diagnostic probes.

    pair-key = sorted("scenA,scenB"). For each (model, strategy, seed) tuple,
    a 'pair instance' is the joint outcome on the diagnostic probes across
    the two scenarios. We tally:
      - both_correct
      - same_wrong (chose the same answer on diagnostic probes — fact-matching failure)
      - other (one correct one wrong, or different wrong answers)
    """
    # Build pair-key from registry.
    pairs: Dict[str, str] = {}
    diagnostics: Dict[str, Tuple[str, ...]] = {}
    for sid, s in SCENARIOS.items():
        if s.pair_with:
            pairs[sid] = s.pair_with
            diagnostics[sid] = s.pair_diagnostic_probes

    # Index trials by (model, strategy, seed, scenario).
    idx: Dict[Tuple[str, str, int, str], dict] = {}
    for t in trials:
        idx[(t["model"], t["strategy"], t["seed"], t["scenario_id"])] = t

    bucket: Dict[Tuple[str, str, str], dict] = defaultdict(
        lambda: {"both_correct": 0, "same_wrong": 0, "other": 0, "pair_count": 0}
    )

    seen_pairs = set()
    for sidA, sidB in pairs.items():
        pair_key = ",".join(sorted([sidA, sidB]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        diagA = diagnostics.get(sidA, ())
        for (model, strategy, seed, scen), _ in list(idx.items()):
            if scen != sidA:
                continue
            tA = idx.get((model, strategy, seed, sidA))
            tB = idx.get((model, strategy, seed, sidB))
            if not tA or not tB:
                continue
            for probe_id in diagA:
                pA = next((p for p in tA["probes"] if p["probe_id"] == probe_id), None)
                pB = next((p for p in tB["probes"] if p["probe_id"] == probe_id), None)
                if not pA or not pB:
                    continue
                key = (model, strategy, pair_key)
                b = bucket[key]
                b["pair_count"] += 1
                if pA["correct"] and pB["correct"]:
                    b["both_correct"] += 1
                elif (
                    not pA["correct"]
                    and not pB["correct"]
                    and pA.get("parsed_label") == pB.get("parsed_label")
                ):
                    b["same_wrong"] += 1
                else:
                    b["other"] += 1
    return dict(bucket)


def syntactic_vs_logical(trials: List[dict]) -> Dict[Tuple[str, str], dict]:
    """(model, strategy) -> {logical_d3, syntactic_d3} accuracy."""
    bucket: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(
        lambda: {"logical": [0, 0], "syntactic": [0, 0]}
    )
    for t in trials:
        key = (t["model"], t["strategy"])
        for p in t["probes"]:
            if p["depth"] != 3:
                continue
            kind = "syntactic" if p["is_syntactic_control"] else "logical"
            bucket[key][kind][1] += 1
            if p["correct"]:
                bucket[key][kind][0] += 1
    return {k: {kk: tuple(vv) for kk, vv in v.items()} for k, v in bucket.items()}


def parse_failure_rate(trials: List[dict]) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """(model, strategy) -> (failures, total)."""
    bucket: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
    for t in trials:
        for p in t["probes"]:
            key = (t["model"], t["strategy"])
            bucket[key][1] += 1
            if p["parse_status"] != "OK":
                bucket[key][0] += 1
    return {k: tuple(v) for k, v in bucket.items()}


def paraphrase_accuracy_by_scenario(
    trials: List[dict],
) -> Dict[Tuple[str, str, str], dict]:
    """Accuracy by paraphrase seed for each (model, strategy, scenario)."""
    seed_bucket: Dict[Tuple[str, str, str, int], List[int]] = defaultdict(
        lambda: [0, 0]
    )
    for t in trials:
        key_base = (t["model"], t["strategy"], t["scenario_id"], t.get("seed", 0))
        for p in t["probes"]:
            if p["is_syntactic_control"]:
                continue
            seed_bucket[key_base][1] += 1
            if p["correct"]:
                seed_bucket[key_base][0] += 1

    grouped: Dict[Tuple[str, str, str], dict] = defaultdict(lambda: {"by_seed": {}})
    for (model, strategy, scenario, seed), (correct, total) in seed_bucket.items():
        acc = correct / total if total else 0.0
        grouped[(model, strategy, scenario)]["by_seed"][seed] = {
            "correct": correct,
            "total": total,
            "accuracy": acc,
        }

    for data in grouped.values():
        accs = [v["accuracy"] for _, v in sorted(data["by_seed"].items())]
        mean = sum(accs) / len(accs) if accs else 0.0
        variance = sum((a - mean) ** 2 for a in accs) / len(accs) if accs else 0.0
        data["mean"] = mean
        data["std"] = math.sqrt(variance)
        data["seed_count"] = len(accs)
    return dict(grouped)


def robust_hard_probes(
    trials: List[dict], threshold: float = 0.5, min_seeds: int = 2
) -> List[dict]:
    """Probes that stay hard across all observed paraphrase seeds."""
    bucket: Dict[Tuple[str, str], Dict[int, List[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    depth: Dict[Tuple[str, str], int] = {}
    for t in trials:
        seed = t.get("seed", 0)
        for p in t["probes"]:
            if p["is_syntactic_control"]:
                continue
            key = (t["scenario_id"], p["probe_id"])
            depth[key] = p["depth"]
            bucket[key][seed][1] += 1
            if p["correct"]:
                bucket[key][seed][0] += 1

    out = []
    for (scenario, probe_id), by_seed in bucket.items():
        if len(by_seed) < min_seeds:
            continue
        seed_acc = {
            seed: (correct / total if total else 0.0)
            for seed, (correct, total) in by_seed.items()
        }
        if all(acc <= threshold for acc in seed_acc.values()):
            out.append(
                {
                    "scenario": scenario,
                    "probe_id": probe_id,
                    "depth": depth[(scenario, probe_id)],
                    "seed_accuracy": seed_acc,
                    "mean": sum(seed_acc.values()) / len(seed_acc),
                }
            )
    return sorted(out, key=lambda r: (r["mean"], r["scenario"], r["probe_id"]))
