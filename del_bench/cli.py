"""DEL-Bench CLI.

Subcommands:

  verify    Run kernel canonical tests + scenario sanity. No API calls.
  run       Execute the (scenario × model × strategy × paraphrase seed) grid.
  analyze   Produce a markdown report from a JSONL of trial results.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

from del_bench.analysis import report
from del_bench.llm.openrouter import from_env
from del_bench.nl.strategies import STRATEGIES, get as get_strategy
from del_bench.runner.orchestrator import run_grid
from del_bench.scenarios.registry import SCENARIOS, all_ids
from del_bench.scenarios.schema import gold_choice

# ---------------------------------------------------------------------------


def _run_test_module(filename: str) -> int:
    test_mod = Path(__file__).resolve().parent.parent / "tests" / filename
    spec = importlib.util.spec_from_file_location(test_mod.stem, str(test_mod))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    failed = 0
    for k, v in vars(mod).items():
        if k.startswith("test_") and callable(v):
            try:
                v()
                print(f"  PASS {k}")
            except Exception as e:
                failed += 1
                print(f"  FAIL {k}: {e}")
    return failed


def cmd_verify(_: argparse.Namespace) -> int:
    print("[verify] running kernel canonical tests…")
    failed = _run_test_module("test_kernel_canonical.py")
    if failed:
        print(f"[verify] {failed} kernel test(s) failed — aborting.")
        return 1

    print("[verify] running evaluator semantics tests…")
    failed = _run_test_module("test_evaluator.py")
    if failed:
        print(f"[verify] {failed} evaluator test(s) failed — aborting.")
        return 1

    print("[verify] running generated-model property checks…")
    failed = _run_test_module("test_kernel_properties.py")
    if failed:
        print(f"[verify] {failed} property check(s) failed — aborting.")
        return 1

    print("[verify] running scenario semantic contracts…")
    failed = _run_test_module("test_scenario_semantics.py")
    if failed:
        print(f"[verify] {failed} scenario contract(s) failed — aborting.")
        return 1

    print("[verify] running scenario integrity tests…")
    failed = _run_test_module("test_scenarios.py")
    if failed:
        print(f"[verify] {failed} scenario integrity test(s) failed — aborting.")
        return 1

    print("[verify] checking scenario gold consistency…")
    for sid, scenario in SCENARIOS.items():
        for probe in scenario.probes:
            try:
                gold_choice(probe, scenario.formal_model)
            except Exception as e:
                print(f"  FAIL {sid}/{probe.id}: {e}")
                return 1
        # Pair-diagnostic check
        if scenario.pair_with:
            other = SCENARIOS.get(scenario.pair_with)
            if other is None:
                print(f"  FAIL {sid}: pair_with={scenario.pair_with} not in registry")
                return 1
            for pid in scenario.pair_diagnostic_probes:
                pa = next((p for p in scenario.probes if p.id == pid), None)
                pb = next((p for p in other.probes if p.id == pid), None)
                if pa is None or pb is None:
                    print(f"  FAIL {sid}/{pid}: missing in pair")
                    return 1
                ga = gold_choice(pa, scenario.formal_model).label
                gb = gold_choice(pb, other.formal_model).label
                if ga == gb:
                    print(
                        f"  WARN {sid}/{pid}: pair-diagnostic gold matches partner ({ga!r})"
                    )
        print(f"  OK   {sid} ({len(scenario.probes)} probes)")

    print("[verify] all checks passed.")
    return 0


# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    scenario_ids = args.scenarios or list(all_ids())
    for sid in scenario_ids:
        if sid not in SCENARIOS:
            print(f"unknown scenario: {sid}", file=sys.stderr)
            return 2
    strategies = [get_strategy(s) for s in (args.strategies or ["direct"])]
    seeds = args.seeds or [0]
    models = args.models
    if not models:
        print("--models is required (at least one)", file=sys.stderr)
        return 2

    results_path = args.results
    if results_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_path = Path("results") / f"run_{ts}.jsonl"
    else:
        results_path = Path(results_path)

    client = from_env(max_calls=args.max_calls)
    print(f"[run] writing to {results_path}")

    def on_done(trial):
        n = sum(1 for p in trial.probes if p.correct)
        print(
            f"[run] done {trial.trial_id}: {n}/{len(trial.probes)} correct "
            f"({sum(p.tokens_in for p in trial.probes)} in / {sum(p.tokens_out for p in trial.probes)} out tokens)"
        )

    run_grid(
        scenario_ids=scenario_ids,
        models=models,
        strategies=strategies,
        seeds=seeds,
        client=client,
        results_path=results_path,
        max_workers=args.max_workers,
        on_complete=on_done,
    )
    print(f"[run] complete. results in {results_path}")
    return 0


# ---------------------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    jsonl_path = Path(args.results_file)
    if not jsonl_path.exists():
        print(f"results file not found: {jsonl_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir or "reports")
    md_path = report.write_report(jsonl_path, out_dir)
    md_text = md_path.read_text()
    print(md_text)
    print(f"\n[analyze] wrote {md_path}")
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="del_bench")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("verify", help="kernel + scenario sanity checks")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("run", help="execute trials")
    sp.add_argument("--scenarios", nargs="*", help="scenario ids (default: all)")
    sp.add_argument("--models", nargs="+", required=True, help="OpenRouter model ids")
    sp.add_argument(
        "--strategies",
        nargs="*",
        default=["direct"],
        choices=sorted(STRATEGIES.keys()),
        help="prompt strategies",
    )
    sp.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[0],
        help="paraphrase seeds; 0, 1, 2 select the three scenario narrative variants",
    )
    sp.add_argument("--results", default=None, help="output JSONL path")
    sp.add_argument("--max-workers", type=int, default=4)
    sp.add_argument(
        "--max-calls",
        type=int,
        default=500,
        help="hard cap on API calls per process (cost protection)",
    )
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("analyze", help="render markdown report")
    sp.add_argument("--results-file", required=True)
    sp.add_argument("--out-dir", default="reports")
    sp.set_defaults(func=cmd_analyze)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
