# Good Model Labs Benchmark Explained

This guide explains exactly what happens from start to finish when you run this benchmark.

## One-line purpose

Compare multiple AI systems on the same recurring nonprofit task, using the same dataset and the same metric suite.

## What this framework benchmarks

It benchmarks a SystemUnderTest, not a deployment outcome.

A SystemUnderTest is any async system that can do:

1. Take one benchmark example as input.
2. Return a normalized SystemOutput.

That system can be prompt-only, RAG, agentic, parser-based, or something else.

## End-to-end flow

When you run python main.py, this is the full flow.

1. main.py calls examples/run_food_pantry_benchmark.py.
2. The example script imports metrics.py so built-in metrics register themselves.
3. The script discovers modules in datasets/ and workflows/.
4. Discovery imports those modules, which run decorators that populate registries.
5. Registries populated during discovery: DATASET_REGISTRY, SYSTEM_REGISTRY, METRIC_REGISTRY.
6. The script creates one dataset, three systems, and a metric suite.
7. A Benchmark object is created with dataset and metrics.
8. A Leaderboard object is created with ranking weights.
9. Leaderboard.rank runs Benchmark.arun for each system.
10. Benchmark.arun runs the system on every example, computes metrics, writes artifacts, aggregates telemetry, and returns one BenchmarkResult.
11. Leaderboard computes weighted scores, sorts descending, and assigns rank numbers.
12. reports.leaderboard_to_markdown formats results into the final table printed to terminal.

## What happens inside one example run

For each example in the dataset:

1. Runner starts a timer.
2. Runner calls await system.run(example).
3. If system.run succeeds, Runner stores output and latency.
4. If system.run raises, Runner captures error text and stack trace and creates a fallback empty SystemOutput.
5. Runner returns ExampleRun objects for all examples.

## What a system output contains

A SystemOutput can include:

1. structured_output: normalized dict used by many metrics
2. text_output: optional plain text representation
3. raw_output: provider or pipeline raw payload
4. telemetry: per-call token, cost, latency, retry, failure data
5. traces: step events with timestamps and status
6. prompts and raw_model_responses for debugging
7. intermediate_outputs and exceptions

## Metrics and scoring

Metric types used in this framework:

1. Rule-based metrics
2. Reference-based metrics
3. LLM-as-a-Judge metrics

The benchmark computes metric values per system across the full dataset.

The leaderboard score is a weighted average over selected metrics:

1. Positive weight means higher is better.
2. Negative weight means lower is better.
3. Missing metrics are skipped.

## Artifacts written on each benchmark run

Benchmark.arun writes files to artifacts/:

1. artifacts/traces/: one JSON trace file per system+dataset run, with per-example trace events.
2. artifacts/failure_logs/: one folder per system and one JSON file per failed example, including input, expected output, actual output, prompts, model responses, intermediates, exceptions, stack trace, and timing.

## Built-in example in this repository

Task:

1. Food Pantry Intake Structuring

Dataset:

1. food_pantry_intake_v1
2. 5 examples
3. Includes one prompt-injection-tagged example

Systems:

1. Food Pantry Intake Workflow A
2. Food Pantry Intake Workflow B
3. Food Pantry Intake Workflow C

## File map by responsibility

Core orchestration:

1. benchmark.py
2. runner.py
3. leaderboard.py
4. reports.py

Core data contracts:

1. workflow.py
2. dataset.py
3. evaluators.py

Plugin and discovery:

1. registry.py
2. metrics.py
3. datasets/
4. workflows/

Entry points:

1. main.py
2. examples/run_food_pantry_benchmark.py

## How to run

Run benchmark:

1. python main.py

Run tests:

1. python -m unittest discover -s tests -p "test_*.py"

## Practical interpretation

Use this framework to answer:

1. Which implementation should be the baseline system for this recurring task?

Do not use it to answer:

1. Did a specific nonprofit save time this week?
