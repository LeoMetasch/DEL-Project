# DEL-Bench

A benchmark for testing LLM higher-order belief tracking against ground
truth derived from a Baltag-Smets dynamic epistemic logic (DEL) kernel.

The kernel implements the Action-Priority product update from:

> Baltag, A. & Smets, S. (2008). *A Qualitative Theory of Dynamic
> Interactive Belief Revision*. In Bonanno, van der Hoek & Wooldridge
> (eds.), Logic and the Foundations of Game and Decision Theory
> (LOFT7), Texts in Logic and Games 3, Amsterdam University Press, pp.
> 13–60.

**18 scenarios / 146 total probes** organized along four difficulty axes
(128 logical probes + 18 syntactic controls; Simple → Medium → Hard),
all using the running coin example from §2 of the paper. Each axis varies
one complexity factor across a 3-level gradient.

### Difficulty Axes

```
Axis 1 — Observation:   public_peek → private_peek → double_peek
Axis 2 — Deception:     truthful_announcement → successful_lie → lie_then_peek
Axis 3 — Trust:         unreliable → soft → conflicting_claims
Axis 4 — Asymmetry:     successful_lie → lie_3agent → witness_deception
```

### Scenario Grid

| Scenario | Action chain | Pairs with | Pair type |
| --- | --- | --- | --- |
| `coin_public_peek_v1` | Σ₂ fair-game obs. | `private_peek` | diagnostic |
| `coin_private_peek_v1` | Σ₃ private obs. | `public_peek` | diagnostic |
| `coin_double_peek_v1` | Σ₃(Bob) + Σ₃(Alice) | `private_peek` | degradation |
| `coin_successful_lie_v1` | Σ₃ + Σ₄ lie(K_b T) | `truthful_announcement` | diagnostic |
| `coin_truthful_announcement_v1` | Σ₃ + Σ₄ truth(K_b H) | `successful_lie` | diagnostic |
| `coin_lie_then_peek_v1` | Σ₃ + Σ₄ + Σ₂(Alice) | `successful_lie` | diagnostic |
| `coin_soft_announcement_v1` | lex. upgrade(T) | `unreliable_announcement` | diagnostic |
| `coin_unreliable_announcement_v1` | equiplausible(T) | `soft_announcement` | diagnostic |
| `coin_conflicting_claims_v1` | Σ₂(Bob) + Σ₄(K_b T) + Σ₂(Charles) + Σ₄(K_c H) | `soft_announcement` | diagnostic |
| `coin_trusted_lie_v1` | Σ₂(Bob) + Σ₄ lie(K_b T) | `soft_announcement` | diagnostic |
| `coin_peek_then_soft_v1` | Σ₃ + lex. upgrade(T) | `soft_announcement` | diagnostic |
| `coin_successful_lie_3agent_v1` | Σ₃ + Σ₄ + Charles | `successful_lie` | degradation |
| `coin_witness_deception_v1` | Σ₂(Bob) + Σ₂(Charles) + Σ₄ | `lie_3agent` | diagnostic |
| `coin_private_peek_3agent_v1` | Σ₃ + Charles | `private_peek` | degradation |
| `coin_public_peek_tails_v1` | Σ₂ (coin=T) | `public_peek` | baseline |
| `coin_peek_then_announce_v1` | Σ₂ + !ignorance | `public_peek` | degradation |
| `coin_moore_sentence_v1` | !(H ∧ ¬B_b H) | — | standalone |
| `coin_conditional_belief_v1` | static B^marked(T) revision | — | standalone |

### Cross-Axis & Degradation Tests

```
CROSS-AXIS:     peek_then_soft (observation × trust)
                trusted_lie (trust × speaker knowledge)
DEGRADATION:    peek_then_announce (step count)
                private_peek_3agent (agent count, private)
                lie_3agent (agent count, lie)
                double_peek (mutual private observation)
BASELINE:       public_peek_tails (factual flip control)
STANDALONE:     moore_sentence
                conditional_belief
```

## Repository layout

```
del_bench/
  kernel/                Pure DEL math. Has no I/O, no LLM, no English.
    formulas.py            Formula AST: Atom, Not, And, K, B, ConditionalB (+ Or, Implies, nested helper)
    models.py              EPM, ActionModel, World, Action; partition + preorder validators
    evaluator.py           Recursive evaluation of formulas at worlds
    product.py             Anti-lexicographic Action-Priority product update
    canonical_actions.py   Σ_!P, Σ₂, Σ₃, Σ₄, lex. upgrade, unreliable announcement
  scenarios/             Scenario data + registry (18 scenarios)
    schema.py              Scenario, ProbeQuestion, AnswerChoice, narrative variants, gold_choice()
    paraphrases.py         Hand-authored narrative variants for final robustness runs
    coin_public_peek.py    Fair-game observation (Axis 1 Simple)
    coin_private_peek.py   Fully private observation (Axis 1 Medium)
    coin_double_peek.py    Both agents peek privately (Axis 1 Hard)
    coin_truthful_announcement.py  Private peek + truth (Axis 2 Simple)
    coin_successful_lie.py Private peek + lie (Axis 2 Medium)
    coin_lie_then_peek.py  Lie then fair-game discovery (Axis 2 Hard)
    coin_unreliable_announcement.py  Unreliable announcement (Axis 3 Simple)
    coin_soft_announcement.py      Lexicographic upgrade (Axis 3 Medium)
    coin_conflicting_claims.py     Two conflicting speakers (Axis 3 Hard)
    coin_successful_lie_3agent.py  3-agent lie (Axis 4 Medium)
    coin_witness_deception.py      Witness to deception (Axis 4 Hard)
    coin_trusted_lie.py    Fair-game observation + trusted lie (cross-axis)
    coin_peek_then_soft.py Private peek + soft (cross-axis)
    coin_private_peek_3agent.py    3-agent private peek (degradation)
    coin_public_peek_tails.py      Reversed actual coin=T (baseline)
    coin_peek_then_announce.py     Multi-step degradation
    coin_moore_sentence.py Moore sentence (standalone)
    coin_conditional_belief.py Static conditional revision (standalone)
    registry.py            SCENARIOS dict with paraphrases attached
  nl/                    Natural-language layer
    renderer.py            Scenario × Probe × Strategy → (system, user, schema)
    strategies.py          DIRECT and COT prompt strategies
    parser.py              LLM raw → ParseResult, with fallback parsing
  llm/
    openrouter.py          Minimal sync OpenRouter client w/ cost cap + retries
  runner/
    trial.py               Run all probes for one (scenario, model, strategy, paraphrase seed)
    orchestrator.py        Cross-product runner; resumes on existing JSONL
  analysis/
    metrics.py             Per-depth, per-action-type, matched-pair, syntactic-vs-logical
    report.py              Markdown report writer
  cli.py                 verify / run / analyze subcommands
tests/                   Standalone runnable tests (no pytest dep needed)
results/                 Ignored JSONL outputs from `run`
reports/                 Ignored Markdown reports and plots
```

## Setup

```bash
# Recommended for this repo.
uv sync

# Provide your OpenRouter API key.
cp .env.example .env
$EDITOR .env   # paste OPENROUTER_API_KEY=sk-or-v1-...
```

The recommended invocation is `uv run python -m del_bench ...` from the
repo root. The `del-bench` console-script entry point also exists, but
using the module form keeps the path behavior explicit.

## Running

### 1. Verify (no API calls)

Reproduces the paper's worked Examples 2.1, 2.2, 2.3 and the
post-private-peek successful-lie composite, then sanity-checks every
scenario's gold-answer table and pair-diagnostic invariants.

```bash
uv run python -m del_bench verify
```

If any kernel assertion fails the run subcommand should be considered
invalid, since every gold answer downstream depends on the kernel
reproducing the paper examples.

### 2. Run trials

```bash
uv run python -m del_bench run \
    --scenarios coin_public_peek_v1 coin_private_peek_v1 coin_successful_lie_v1 \
    --models openai/gpt-4o-mini anthropic/claude-haiku-4.5 google/gemini-2.5-flash \
    --strategies direct cot \
    --seeds 0 \
    --max-workers 4 \
    --max-calls 200
```

Notes:

- Output goes to `results/run_YYYYMMDD_HHMMSS.jsonl` unless you pass
  `--results path/to/file.jsonl`.
- Each line is a complete `TrialResult` (one scenario × model × strategy
  × paraphrase seed); the runner is resume-safe — re-running with the same path
  skips trials whose `trial_id` is already present.
- `--max-calls` is a hard per-process cap to protect against runaway
  spend. The pilot above is roughly 3 × 3 × 2 × 1 = 18 trials × ~6
  probes ≈ 108 calls.
- `--seeds 0 1 2` runs the three narrative variants for each scenario.
  Seed 0 is the canonical narrative; seeds 1 and 2 are hand-authored
  paraphrase variants. Probe wording, answer choices, and DEL gold
  labels are fixed.
- Determinism: `temperature=0`. Each probe is a separate API call to
  avoid ordering effects.

The final 5-provider robustness run used one model per provider:

```bash
uv run python -m del_bench run \
    --models openai/gpt-4o-mini anthropic/claude-haiku-4.5 google/gemini-2.5-flash mistralai/mistral-small-3.2-24b-instruct qwen/qwen3-32b \
    --strategies direct cot \
    --seeds 0 1 2 \
    --max-workers 4 \
    --max-calls 4500 \
    --results results/final_5provider_mixed_3seed.jsonl
```

For the current benchmark, the full-run call count is:

```text
146 probes × number_of_models × number_of_strategies × number_of_seeds
```

The final run therefore contains `146 × 5 × 2 × 3 = 4380` probe-level
model calls.

### 3. Analyze

```bash
uv run python -m del_bench analyze \
    --results-file results/final_5provider_mixed_3seed.jsonl \
    --out-dir reports/
```

Produces a markdown report with:

- Run coverage: scenarios, trials, model count, strategy count,
  paraphrase seeds, logical/syntactic probe counts, and parse failures
- Overall logical accuracy with Wilson 95% confidence intervals
- Per-depth accuracy (excluding syntactic-control probes)
- Per-action-type accuracy
- Paraphrase robustness metrics when multiple seeds are present
- Matched-pair sensitivity (public/private peek): `both_correct` vs.
  `same_wrong` vs. `other`. `same_wrong` is the "fact-matching failure"
  diagnostic — model gave the same wrong answer in both members of the
  pair.
- Depth-3 syntactic vs. logical accuracy
- Parse-failure rate per (model, strategy)
- Appendix of selected depth-3 errors with full raw responses

To generate the paper-oriented plots:

```bash
uv run python make_paper_plots.py \
    --results-file results/final_5provider_mixed_3seed.jsonl \
    --out-dir reports/plots
```

Generated run artifacts are ignored by git:

```text
results/final_5provider_mixed_3seed.jsonl
reports/final_5provider_mixed_3seed_report.md
reports/plots/
```

## Tests

Each test file is also runnable directly:

```bash
uv run python tests/test_kernel_canonical.py    # paper Examples 2.1/2.2/2.3/3.3
uv run python tests/test_evaluator.py           # K, B, and conditional-B semantics on tiny models
uv run python tests/test_kernel_properties.py   # generated-model reference checks
uv run python tests/test_scenario_semantics.py   # explicit benchmark semantic contracts
uv run python tests/test_parser.py              # JSON robustness
uv run python tests/test_renderer.py            # paraphrase seed rendering
uv run python tests/test_scenarios.py           # gold uniqueness, pair invariants
```

If you have `pytest` installed, `pytest tests/` works too.

## Adding a scenario

1. Create `del_bench/scenarios/my_scenario.py`.
2. Build the formal model: prior EPM → product with one or more
   canonical action models (`del_bench.kernel.canonical_actions`).
3. Author probes as `ProbeQuestion(...)` with `AnswerChoice(label,
   formula)` triples. The label is the literal string the LLM sees and
   that appears in the JSON enum; the formula is what the kernel
   evaluates to determine gold.
4. The `None`-formula choice is gold iff every other choice's formula
   is false at the actual world (handles "no settled belief").
5. For syntactic-control probes (linguistically deep but logically
   shallow), set `syntactic_control=True` and use `fixed_gold_label`
   instead of formulas.
6. Self-test: call `gold_choice(p, model)` for each probe inside
   `build()`. Authoring will fail fast if a probe has zero or multiple
   gold answers.
7. Register in `del_bench/scenarios/registry.py`.
8. `python -m del_bench verify` to re-check everything.

## Design notes

- **Kernel correctness.** The kernel reproduces the paper's Examples
  2.1–2.3 and the page-46 lie composite. Adding a new action template
  should be paired with a canonical test that exercises a published
  worked example.
- **Plausibility convention.** Following the paper (§2.1, p.20) we use
  `s ≤_a t` to mean "s is at least as plausible as t". The
  ≤-*minimal* states in an information cell are therefore the
  *most* plausible — these are the doxastic accessibility targets for
  `B_a`.
- **Conditional belief.** Following §2.2, `ConditionalB(a, Q, P)`
  represents `B^Q_a P`: `P` holds at every most-plausible `Q`-world in
  agent `a`'s current information cell. Ordinary `B_a P` is `B^top_a P`.
- **Action-Priority Update** (§3.2.2, p.45):

      (s, σ) ≤_a (s', σ')   iff
          (σ <_a σ' AND s ~_a s')  OR  (σ ≅_a σ' AND s ≤_a s')

  with indistinguishability `(s, σ) ~_a (s', σ')` iff `s ~_a s'` and
  `σ ~_a σ'`. The product is computed in `kernel/product.py`.
- **EPMs are validated on construction.** Partition over worlds,
  preorder reflexivity, transitivity, coherence (`s ≤_a t ⇒ s ~_a t`),
  and local connectedness within each indist cell are all checked. A
  malformed model raises immediately rather than silently producing
  wrong gold answers later.
- **Paraphrase seeds.** Each registered scenario exposes three narrative
  variants. The canonical `narrative` is variant 0, and seeds 1 and 2
  select hand-authored paraphrases. This varies the English scenario
  while preserving probes, answer choices, formal models, and gold
  labels.
- **Determinism.** Given the same scenario, the kernel produces the
  same gold answers bit-for-bit. Given the same scenario × strategy ×
  paraphrase seed, the rendered prompt is bit-for-bit identical. Only
  the LLM is stochastic, and that is pinned at `temperature=0`.

## Limitations

- Statistical power depends on running multiple paraphrase seeds and/or
  models.
- Kernel implements the finite EPM fragment needed by the benchmark:
  Boolean formulas, `K`, `B`, `B^Q` (conditional belief), and
  Action-Priority product updates. The paper's `□` (safe belief) is not
  exposed by the AST or evaluator yet.
- New scenarios should receive two explicit paraphrases before being
  used in a robustness run.
