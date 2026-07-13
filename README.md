# Good Model Labs Benchmark Framework

This repository benchmarks complete AI systems that solve recurring nonprofit operational tasks.

For a plain-language deep dive, see [GOOD_MODEL_LABS_BENCHMARK.md](GOOD_MODEL_LABS_BENCHMARK.md).

It does not benchmark nonprofits.
It does not benchmark deployment outcomes.

The core abstraction is SystemUnderTest.
A system can be prompt-only, RAG, agentic workflow, parser pipeline, or eventually a fine-tuned model.

Every benchmarked system implements an async black-box interface:

```python
class SystemUnderTest(ABC):
	async def run(self, example: BenchmarkExample) -> SystemOutput:
		...
```

## Design Goal

Build a reusable benchmark foundation for Good Model Labs, where systems are compared on standardized datasets for recurring tasks.

Example task:

- Food pantry intake note -> structured client JSON

The benchmark answers:

- Which implementation performs best on this task?

## Architecture

```text
.
├── benchmark.py
├── runner.py
├── workflow.py
├── providers.py
├── dataset.py
├── metrics.py
├── evaluators.py
├── leaderboard.py
├── reports.py
├── registry.py
├── workflows/
│   └── food_pantry_intake.py
├── datasets/
│   └── food_pantry_intake.py
├── examples/
│   └── run_food_pantry_benchmark.py
├── tests/
│   └── test_labs_benchmark.py
└── main.py
```

## Core Concepts

1. SystemUnderTest

- Black-box task system interface.
- Benchmark does not assume prompts, agents, or model type.

2. Provider abstraction

- Unified provider interface for OpenAI, Anthropic, Gemini, and Mock.
- Real API usage when environment keys are present:
	- OPENAI_API_KEY
	- ANTHROPIC_API_KEY
	- GOOGLE_API_KEY

3. BenchmarkDataset

- List of DatasetExample records.
- Each example includes input_text, ground_truth, metadata.

4. Runner

- Executes a system over dataset examples.
- Captures output, latency, and runtime errors.

5. Metric

- Pluggable class with a compute(context) method.
- New metrics can be added without touching core benchmark logic.

6. Benchmark

- Runs Runner + metric evaluator.
- Returns BenchmarkResult with metric values and failure cases.

7. Leaderboard

- Compares multiple systems.
- Uses caller-provided ranking weights (no hardcoded rank strategy in core).

8. Registry

- Systems, datasets, and metrics self-register.
- Automatic discovery via module and package scanning.

## Built-in Example

Included benchmark task:

- Food Pantry Intake Structuring
- Dataset id: food_pantry_intake_v1 (5 examples)

Included systems:

- Food Pantry Intake Workflow A
- Food Pantry Intake Workflow B
- Food Pantry Intake Workflow C

Included metrics:

- accuracy
- precision
- recall
- hallucination_rate
- json_validity
- latency_ms
- cost_usd
- robustness
- safety
- pii_leakage
- prompt_injection
- cross_org
- cross_language
- cross_model
- prompt_tokens
- completion_tokens
- total_api_calls
- failure_rate
- retry_count
- peak_latency_ms
- average_call_latency_ms
- cost_per_success
- token_efficiency
- llm_judge_quality

## Evaluator Types

The framework supports three evaluator families:

1. Rule-based evaluators
- JSON validity, latency, cost, token usage, API call count, failure rate.

2. Reference-based evaluators
- Accuracy, precision, recall, hallucination, robustness, cross-org/language/model.

3. LLM-as-a-Judge evaluators
- Rubric-based quality scoring for tasks where exact matching is insufficient.

## Quick Start

Run the benchmark demo:

```bash
python main.py
```

When API keys are configured, systems use real provider calls automatically.
Without keys, the framework falls back to MockProvider for local testing.

Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Programmatic Usage

```python
import metrics  # ensures built-in metrics register
from benchmark import Benchmark
from leaderboard import Leaderboard
from registry import (
	DATASET_REGISTRY,
	SYSTEM_REGISTRY,
	build_metric_suite,
	discover_package_modules,
)

discover_package_modules("datasets")
discover_package_modules("workflows")

dataset = DATASET_REGISTRY["food_pantry_intake_v1"]()
systems = [
	SYSTEM_REGISTRY["food_pantry_intake_a"](),
	SYSTEM_REGISTRY["food_pantry_intake_b"](),
	SYSTEM_REGISTRY["food_pantry_intake_c"](),
]

metrics = build_metric_suite(["accuracy", "json_validity", "hallucination_rate", "latency_ms", "cost_usd"])
benchmark = Benchmark(dataset=dataset, metrics=metrics)

leaderboard = Leaderboard(
	benchmark=benchmark,
	ranking_weights={"accuracy": 0.7, "json_validity": 0.2, "cost_usd": -0.1},
)

rankings = leaderboard.rank(systems)
```

## Telemetry, Traces, and Failure Logs

Each run automatically records telemetry fields such as:

1. Total API calls
2. Prompt and completion tokens
3. Total, average, and peak latency
4. Retry count
5. Provider/model/temperature/context length
6. Total cost and cost-per-success
7. Failures and rate-limit events

Artifacts are saved under artifacts/:

1. traces/: serialized step-by-step execution traces for all evaluated examples
2. failure_logs/: failed-example logs with input, expected, actual, prompts, model responses, intermediate outputs, and stack traces

## Extending to New Benchmarks

To add a new benchmark (Grant Reporting, Missed Call Triage, Knowledge Retrieval, etc):

1. Add a dataset module under datasets.
2. Add one or more system modules under workflows.
3. Register systems and dataset with decorators.
4. Add task-specific metrics if needed.
5. Run through Benchmark + Leaderboard without changing core benchmark code.
