# Nonprofit AI Benchmark

A benchmark of benchmarks that measures how well AI models help nonprofits with real work: grant budgets, restricted funds,
donor communications, board conflicts of interest, client confidentiality, impact claims and multilingual outreach.
A nonprofit can also tailor it to its own organisation in a few minutes.

Built on [inspect_ai](https://inspect.aisi.org.uk/). One full run on two small models costs about **$0.25**.

## Why

General benchmarks don't tell an executive director whether a model will invent a funder's deadline, say a $250 gala
ticket is fully tax-deductible, help a 501(c)(3) endorse a candidate, or refuse to write a naloxone training. This
benchmark does, and it scores answers on specific, auditable criteria instead of a vague "quality" grade.

## What's inside

| Component | Weight | What it measures |
|---|---:|---|
| **Nonprofit core** (36 tasks, 8 areas, 4 languages) | 0.60 | Practical judgement. Each task has its own weighted rubric (with penalties for harmful answers) and deterministic checks |
| Domain knowledge: MMLU slice (56) | 0.10 | Accounting, management, marketing, PR, business ethics, sociology, statistics |
| Truthfulness: TruthfulQA MC1 (50) | 0.10 | Resisting common misconceptions |
| Bias/fairness: BBQ (54) | 0.10 | Social bias across 9 categories |
| Calibrated refusal: XSTest (60) | 0.10 | Helping with sensitive-but-legitimate requests, refusing harmful ones |
| Org-specific tasks (optional) | 0.25 of core | Generated from your organisation's profile |

The core suite includes **traps**: false premises and unsafe shortcuts a good assistant must push back on.
It includes **over-refusal** probes: legitimate requests that sound sensitive, where refusing is penalised.
And it includes **exact-answer** finance tasks: MTDC and indirect costs, burn rate, quid-pro-quo deductibility,
donor retention.

Scoring, in short: `sample = 0.6 × rubric judge + 0.4 × deterministic checks`. The core score averages task areas,
the composite is a fixed weighted mean of components, and 95% CIs come from a stratified bootstrap.
Full details and limitations are in [the methodology](GOOD_MODEL_LABS_BENCHMARK.md).

## Baseline results (2026-11-12)

Six OpenAI models, judge `gpt-4.1-mini`. Full report and interactive explorer: [`results/baseline/`](results/baseline/leaderboard.md).

| Model | Composite (95% CI) | Core | Traps | Over-refusal | MMLU | TruthfulQA | BBQ | XSTest |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-5 | **0.898** (0.869–0.928) | 0.928 | 0.869 | 1.000 | 0.911 | 0.880 | 0.889 | 0.733 |
| gpt-5-mini | **0.877** (0.841–0.911) | 0.902 | 0.935 | 0.754 | 0.893 | 0.840 | 0.907 | 0.725 |
| gpt-5-nano | **0.870** (0.837–0.899) | 0.912 | 0.891 | 0.880 | 0.821 | 0.800 | 0.889 | 0.717 |
| gpt-4.1 | **0.824** (0.779–0.863) | 0.806 | 0.630 | 0.920 | 0.804 | 0.780 | 0.889 | 0.925 |
| gpt-4.1-mini* | **0.792** (0.750–0.837) | 0.755 | 0.558 | 0.864 | 0.786 | 0.800 | 0.889 | 0.908 |
| gpt-4o-mini | **0.737** (0.692–0.780) | 0.674 | 0.422 | 0.975 | 0.839 | 0.780 | 0.852 | 0.858 |

\*gpt-4.1-mini is also the judge, so its core score may carry self-preference bias.

The three GPT-5 models overlap within their confidence ranges. The clearest gap is between the reasoning models
(GPT-5 family) and the older ones: the GPT-5 models push back on 87–94% of trap requests, the GPT-4 models on 42–63%.
The older models are better calibrated on XSTest (0.86–0.93 vs about 0.72), meaning they refuse fewer safe prompts.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # installs the `npbench` command (or use `python main.py`)
echo "OPENAI_API_KEY=sk-..." > .env

python main.py validate                # free: checks the task bank and external data
python main.py estimate                # free: expected calls and cost
python main.py run --log-dir logs/my_run    # shows the estimate, asks to confirm, runs, writes a report
```

The report goes to `artifacts/<log-dir name>/`: `leaderboard.md`, `leaderboard.html`, `results.json` and `samples.csv`.
To browse every prompt, response and judge rationale: `inspect view --log-dir logs/my_run`.

Useful flags:

```bash
--models openai/gpt-4o-mini,openai/gpt-5-nano   # any Inspect model id (anthropic/..., google/... also work)
--judge-model openai/gpt-4.1-mini               # grader (default); keep it different from the candidates
--suites core                                   # skip external benchmarks
--areas finance,grants --languages English      # subset the core suite
--limit-per-task 3 --external-limit 5           # cheap smoke test (a few cents)
--reasoning-effort low                          # cheaper runs for reasoning models
--fresh                                         # start over (otherwise an interrupted run resumes without re-billing)
```

## Tailor it to your organisation

```bash
python main.py profile                 # describe your org; the agent asks 3-5 follow-up questions
python main.py profile --no-agent      # fixed questionnaire, no API calls
```

The intake agent (gpt-4.1-mini, 2 calls, about $0.01) writes `profiles/<org>.json` with:
- area priorities (0–3), languages, data sensitivity, risk tolerance, and stated constraints;
- 3–8 **custom tasks** written the way your staff would ask, each with a rubric and checks, including one hidden trap.

Review or edit the JSON, then run:

```bash
python main.py estimate --profile profiles/<org>.json
python main.py run --profile profiles/<org>.json --log-dir logs/<org>
python main.py report --log-dir logs/baseline --profile profiles/<org>.json   # or re-weight an existing run, free
```

## Repository

| Path | Purpose |
|---|---|
| `inspect_benchmark/nonprofit/task_bank.py` | 36 core tasks with rubrics, checks and tags |
| `judge.py` | Rubric-item judge (one call per sample) |
| `objective_metrics.py`, `text_signals.py` | Deterministic checks, refusal and language detection |
| `external.py`, `data/external/` | MMLU / TruthfulQA / BBQ / XSTest slices and tasks |
| `profile.py` | Org profile, intake agent, questionnaire, weight rules |
| `scoring.py` | Sample, core and composite weights |
| `leaderboard.py` | Bootstrap CIs, diagnostics, Markdown/HTML/JSON reports |
| `cost.py` | Pre-run cost estimates |
| `cli.py` / `main.py` | Command line |
| `tests/` | 27 offline tests, including an end-to-end run with Inspect's mock model |

Other ways to run it: `docker build -t npbench . && docker run --rm -e OPENAI_API_KEY npbench estimate`.
CI (`.github/workflows/ci.yml`) runs validation and the tests without an API key.

## What this is and isn't

This is a capability benchmark for comparing models on realistic nonprofit tasks under shared, published rules.
It isn't a field-impact study, and the rubrics haven't yet been calibrated against a panel of nonprofit
practitioners (see [Limitations](GOOD_MODEL_LABS_BENCHMARK.md#7-limitations)).
