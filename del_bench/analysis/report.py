"""Render a markdown report from metrics computed over a results JSONL."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from del_bench.analysis import metrics


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  —  "
    return f"{100.0 * num / denom:5.1f}% ({num}/{denom})"


def _pct_ci(num: int, denom: int) -> str:
    if denom == 0:
        return "  —  "
    lo, hi = metrics.wilson_interval(num, denom)
    return f"{100.0 * num / denom:5.1f}% ({num}/{denom}), 95% CI [{100 * lo:.1f}, {100 * hi:.1f}]"


def render_markdown(jsonl_path: Path) -> str:
    trials = metrics.load_trials(jsonl_path)
    if not trials:
        return "# DEL-Bench Pilot Report\n\n_No trials found._\n"

    out = []
    out.append("# DEL-Bench Pilot Report")
    out.append("")
    out.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    out.append("")
    out.append(f"- Source: `{jsonl_path}`")
    models = sorted({t["model"] for t in trials})
    strategies = sorted({t["strategy"] for t in trials})
    scenarios = sorted({t["scenario_id"] for t in trials})
    coverage = metrics.run_coverage(trials)
    seeds = coverage["seeds"]
    seed_text = ", ".join(str(s) for s in seeds)
    pf_num = coverage["parse_failures"]
    pf_den = coverage["probe_records"]
    out.append(f"- Trials: {coverage['trials']}")
    out.append(
        f"- Run coverage: {coverage['scenario_count']} scenarios, {coverage['model_count']} models, {coverage['strategy_count']} strategies"
    )
    out.append(f"- Paraphrase seeds: {seed_text}")
    out.append(
        "- Registered benchmark size: "
        f"{coverage['registered_total_probes']} total probes "
        f"({coverage['registered_logical_probes']} logical + "
        f"{coverage['registered_syntactic_probes']} syntactic controls)"
    )
    out.append(
        "- Probe records in this run: "
        f"{coverage['probe_records']} total "
        f"({coverage['logical_probe_records']} logical + "
        f"{coverage['syntactic_probe_records']} syntactic controls)"
    )
    out.append(f"- Parse-failure rate: {_pct(pf_num, pf_den)}")
    out.append(f"- Models: {', '.join(models)}")
    out.append(f"- Strategies: {', '.join(strategies)}")
    out.append(f"- Scenarios: {', '.join(scenarios)}")
    out.append("")

    # --- Overall accuracy ---
    out.append("## Overall logical accuracy")
    out.append("")
    out.append(
        "Accuracy excludes syntactic-control probes. Intervals are Wilson 95% confidence intervals."
    )
    out.append("")
    overall = metrics.overall_accuracy(trials)
    out.append("| Model | Strategy | Accuracy |")
    out.append("|---|---|---|")
    for (m, s), (num, den) in sorted(overall.items()):
        out.append(f"| {m} | {s} | {_pct_ci(num, den)} |")
    out.append("")

    # --- Per-depth accuracy ---
    out.append("## Per-depth accuracy (excludes syntactic-control)")
    out.append("")
    by_depth = metrics.per_depth_accuracy(trials)
    depths = sorted({d for (_, _, d) in by_depth.keys()})
    out.append("| Model | Strategy | " + " | ".join(f"d={d}" for d in depths) + " |")
    out.append("|" + "---|" * (2 + len(depths)))
    for m in models:
        for s in strategies:
            row = [m, s]
            for d in depths:
                num, den = by_depth.get((m, s, d), (0, 0))
                row.append(_pct(num, den))
            out.append("| " + " | ".join(row) + " |")
    out.append("")

    # --- Per-action-type ---
    out.append("## Per-action-type accuracy")
    out.append("")
    by_fam = metrics.per_action_type_accuracy(trials)
    fams = sorted({f for (_, _, f) in by_fam.keys()})
    out.append("| Model | Strategy | " + " | ".join(fams) + " |")
    out.append("|" + "---|" * (2 + len(fams)))
    for m in models:
        for s in strategies:
            row = [m, s]
            for f in fams:
                num, den = by_fam.get((m, s, f), (0, 0))
                row.append(_pct(num, den))
            out.append("| " + " | ".join(row) + " |")
    out.append("")

    # --- Paraphrase robustness ---
    out.append("## Paraphrase robustness")
    out.append("")
    para = metrics.paraphrase_accuracy_by_scenario(trials)
    if len(seeds) < 2:
        out.append(
            "_Only one paraphrase seed is present in this results file. Robustness metrics require at least two seeds and are intended for the 3-seed run._"
        )
        out.append("")
    else:
        out.append(
            "Scenario-level logical accuracy aggregated across paraphrase seeds. `std` is the standard deviation of seed-level accuracies."
        )
        out.append("")
        out.append("| Model | Strategy | Scenario | Mean | Std | Seeds |")
        out.append("|---|---|---|---|---|---|")
        rows = sorted(
            para.items(),
            key=lambda item: (item[1]["mean"], item[0][0], item[0][1], item[0][2]),
        )
        for (m, s, scenario), data in rows[:30]:
            seed_cells = ", ".join(
                f"{seed}: {100 * vals['accuracy']:.1f}%"
                for seed, vals in sorted(data["by_seed"].items())
            )
            out.append(
                f"| {m} | {s} | {scenario} | {100 * data['mean']:.1f}% | "
                f"{100 * data['std']:.1f} pp | {seed_cells} |"
            )
        hard = metrics.robust_hard_probes(trials)
        out.append("")
        out.append("### Probes hard across paraphrases")
        out.append("")
        if hard:
            out.append("| Scenario | Probe | Depth | Mean | Seed accuracies |")
            out.append("|---|---|---|---|---|")
            for row in hard[:30]:
                seed_cells = ", ".join(
                    f"{seed}: {100 * acc:.1f}%"
                    for seed, acc in sorted(row["seed_accuracy"].items())
                )
                out.append(
                    f"| {row['scenario']} | {row['probe_id']} | {row['depth']} | "
                    f"{100 * row['mean']:.1f}% | {seed_cells} |"
                )
        else:
            out.append(
                "_No logical probes stayed at or below 50% accuracy across all observed paraphrase seeds._"
            )
        out.append("")

    # --- Matched-pair sensitivity ---
    out.append("## Matched-pair sensitivity")
    out.append("")
    out.append(
        "Diagnostic-probe outcomes across paired scenarios (public/private peek)."
    )
    out.append(
        "`same_wrong` = fact-matching failure: model gave the same wrong answer in both scenarios."
    )
    out.append("")
    pair_data = metrics.matched_pair_sensitivity(trials)
    if pair_data:
        out.append(
            "| Model | Strategy | Pair | both_correct | same_wrong | other | total |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for (m, s, pk), b in sorted(pair_data.items()):
            out.append(
                f"| {m} | {s} | {pk} | {_pct(b['both_correct'], b['pair_count'])} | "
                f"{_pct(b['same_wrong'], b['pair_count'])} | "
                f"{_pct(b['other'], b['pair_count'])} | {b['pair_count']} |"
            )
    else:
        out.append("_No matched pairs in this run._")
    out.append("")

    # --- Syntactic vs logical ---
    out.append("## Depth-3 syntactic vs. logical")
    out.append("")
    sl = metrics.syntactic_vs_logical(trials)
    out.append("| Model | Strategy | Logical d=3 | Syntactic d=3 |")
    out.append("|---|---|---|---|")
    for (m, s), b in sorted(sl.items()):
        ln, ld = b["logical"]
        sn, sd = b["syntactic"]
        out.append(f"| {m} | {s} | {_pct(ln, ld)} | {_pct(sn, sd)} |")
    out.append("")

    # --- Parse failures ---
    out.append("## Parse-failure rate")
    out.append("")
    pf = metrics.parse_failure_rate(trials)
    out.append("| Model | Strategy | Failures |")
    out.append("|---|---|---|")
    for (m, s), (n, d) in sorted(pf.items()):
        out.append(f"| {m} | {s} | {_pct(n, d)} |")
    out.append("")

    # --- Appendix: depth-3 wrong-answer worked examples ---
    out.append("## Appendix: depth-3 errors (selected)")
    out.append("")
    shown = 0
    for t in trials:
        if shown >= 12:
            break
        for p in t["probes"]:
            if p["depth"] != 3 or p["is_syntactic_control"]:
                continue
            if p["correct"]:
                continue
            shown += 1
            out.append(
                f"### {t['model']} | {t['strategy']} | {t['scenario_id']} | {p['probe_id']}"
            )
            out.append("")
            out.append(f"- Gold: `{p['gold_label']}`")
            out.append(
                f"- Chose: `{p.get('parsed_label')}` (status={p['parse_status']})"
            )
            if p.get("reasoning"):
                out.append(f"- Reasoning: {p['reasoning'][:600]}")
            out.append(f"- Raw: `{(p['raw_response'] or '')[:600]}`")
            out.append("")
            if shown >= 12:
                break
    if shown == 0:
        out.append("_No depth-3 errors observed._")
        out.append("")

    return "\n".join(out)


def write_report(jsonl_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(jsonl_path)
    out_path = out_dir / (jsonl_path.stem + "_report.md")
    out_path.write_text(md)
    return out_path
