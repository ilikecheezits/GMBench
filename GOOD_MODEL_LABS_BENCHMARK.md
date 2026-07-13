# Good Model Labs Benchmark Explained

This document explains the new architecture in plain language.

## What changed

The old abstraction was Deployment.
The new abstraction is SystemUnderTest.

That means the benchmark now compares AI task systems directly, not nonprofit deployment outcomes.

## What gets benchmarked

A SystemUnderTest can be any implementation that solves a recurring nonprofit task:

1. Prompt workflow
2. RAG pipeline
3. Agent graph
4. Parser-orchestration service
5. Fine-tuned model (later)

The benchmark treats all of these as black boxes with one interface:

- input text plus metadata in
- structured output plus metadata out

The interface is asynchronous so systems can run multi-step pipelines with multiple API calls naturally.

## Current repository layout

Core framework modules live at the repository root:

1. benchmark.py
2. runner.py
3. workflow.py
4. providers.py
5. dataset.py
6. metrics.py
7. evaluators.py
8. leaderboard.py
9. reports.py
10. registry.py

Task implementations and fixtures live in:

1. workflows/
2. datasets/
3. examples/
4. tests/

## Core benchmark loop

1. Load a standardized task dataset.
2. Execute one system on every example.
3. Collect per-example outputs, latency, and errors.
4. Compute metric suite across the run.
5. Return BenchmarkResult.

## Dataset shape

Each dataset example has:

1. input_text
2. ground_truth
3. metadata

Metadata can include organization, language, difficulty, prompt-injection flag, OCR quality, or anything else needed for slicing.

## Metric design

Metrics are modular classes.

Built-in examples include:

1. accuracy
2. precision
3. recall
4. hallucination_rate
5. json_validity
6. latency_ms
7. cost_usd
8. robustness
9. safety
10. pii_leakage
11. prompt_injection
12. cross_org
13. cross_language
14. cross_model

To add a metric, implement one class with compute(context).

Evaluator families:

1. Rule-based evaluators
2. Reference-based evaluators
3. LLM-as-a-Judge evaluators

LLM judges are important for tasks where exact string matching is not enough.

## Leaderboard behavior

Leaderboard ranking is weight-driven by caller input.

That means ranking policy is configurable per benchmark and not hardcoded in the core engine.

## Registry and auto-discovery

Systems, datasets, and metrics register themselves with decorators.

The framework discovers modules automatically by scanning packages and importing them.

This makes extension simple:

1. Add a new dataset module.
2. Add new systems.
3. Optionally add metrics.
4. Run benchmark with no core-code edits.

## Provider support

The framework includes a provider abstraction so benchmark code stays unchanged across model vendors.

Built-in providers:

1. OpenAIProvider
2. AnthropicProvider
3. GeminiProvider
4. MockProvider (testing/local fallback)

Environment variables:

1. OPENAI_API_KEY
2. ANTHROPIC_API_KEY
3. GOOGLE_API_KEY

If keys are available, systems use real API calls.

## Built-in task example

Included today:

Food Pantry Intake Structuring

Current dataset version:

1. food_pantry_intake_v1
2. 5 benchmark examples
3. Includes one prompt-injection tagged example for robustness testing

Input:

- raw intake note

Output:

- structured client record JSON

Systems included:

1. Workflow A
2. Workflow B
3. Workflow C

Purpose:

- demonstrate how Labs can compare candidate workflow implementations before declaring reusable infrastructure.

## Why this architecture is future-proof

Because core code depends on SystemUnderTest rather than deployment assumptions, future systems can be swapped in without changing benchmark fundamentals.

Future compatible targets:

1. LangGraph
2. CrewAI
3. MCP-connected systems
4. vendor APIs
5. local model services
6. fine-tuned models

## Telemetry, traces, and failure logs

Each run records telemetry including:

1. Total API calls
2. Prompt/completion tokens
3. Cost
4. Total, average, and peak latency
5. Retry count
6. Failures, rate limits, and exceptions

Each example also records a serializable execution trace.

Failed examples are automatically logged with input, expected output, actual output, prompts, raw model responses, intermediate outputs, and stack traces.

## Practical use

Use this framework when Labs asks:

- which implementation should become the reusable baseline for this recurring nonprofit task?

Do not use it to answer:

- did organization X save six hours this week?

## How to run today

Run end-to-end benchmark:

1. python main.py

Run tests:

1. python -m unittest discover -s tests -p "test_*.py"
