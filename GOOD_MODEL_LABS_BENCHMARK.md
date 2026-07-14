# Good Model Labs Benchmark Explained

This document is written for project reviewers and decision makers.

## What this gives you

This framework gives you a fair side-by-side comparison of AI system options for one recurring nonprofit task.

You can use it to answer:

- Which implementation is strongest overall?
- What does each option trade off in quality, speed, and cost?
- Which system should be the baseline for the next phase?

## What "Workflow" Means In This Benchmark

A workflow is one full system approach for handling the task, not just one model call.

Think of it as the full recipe:

1. How input is interpreted
2. How the model is prompted
3. What post-processing or safeguards are applied
4. How final structured output is produced

Why this is important for reviewers:

- The benchmark compares complete approaches, not isolated prompt snippets.
- A winning workflow is the strongest operational candidate to standardize.
- Differences in scores reflect process design differences, not only model differences.

### Concrete Example

Task:

- Convert a free-text food pantry intake note into structured JSON.

Sample input note:

- "Caller Amina Rahman, family of 6, needs groceries and baby formula. Bengali interpreter needed."

What a workflow does step by step:

1. Reads the note.
2. Identifies key facts (name, size, services, language, urgency).
3. Applies rules and checks to improve consistency.
4. Produces a final structured output used for scoring.

Sample output:

```json
{
	"client_name": "Amina Rahman",
	"household_size": 6,
	"urgency": "high",
	"requested_services": ["groceries", "baby formula"],
	"preferred_language": "Bengali",
	"notes_summary": "High-need household of 6 requesting groceries, baby formula."
}
```

This is why workflow comparison matters: different workflows can produce different quality from the same input.

## What this does not give you

This benchmark does not directly measure real-world impact at a nonprofit site.

It evaluates system behavior in a controlled test setting.

## Simple start-to-finish view

When you run python main.py, this happens:

1. The benchmark loads one dataset for one task.
2. It loads all registered system variants for that task.
3. It runs each system on the same examples.
4. It scores each system with the same metric suite.
5. It computes a weighted overall score.
6. It prints a ranked leaderboard.
7. It stores traces and failure logs for auditability.

## Why reviewers should trust the comparison

1. Same test cases for all systems.
2. Same scoring rules for all systems.
3. Weighted ranking is explicit and configurable.
4. Failed runs and traces are saved for investigation.

## Current built-in benchmark

Task:

- Food Pantry Intake Structuring

Dataset:

- ID: food_pantry_intake_v1
- Size: 5 examples
- Includes a prompt-injection-tagged case

Systems compared:

- Food Pantry Intake Workflow A
- Food Pantry Intake Workflow B
- Food Pantry Intake Workflow C

Default model settings:

- Workflow model: gpt-4o-mini
- Judge model: gpt-4o-mini

## How scoring works

Each system receives multiple metric scores, then one weighted overall score.

Metric families include:

- Quality metrics
- Reliability and safety metrics
- Speed and cost metrics
- LLM judge quality metric

Weighted score logic:

- Positive weight means higher is better.
- Negative weight means lower is better.
- Missing metrics are ignored.

## Performance and runtime behavior

To keep runtime practical, the framework uses bounded parallelism:

- System ranking concurrency default: 3
- Per-system example concurrency default: 4
- LLM judge concurrency default: 4

This speeds up runs while keeping control over API pressure.

## Governance and traceability outputs

Each run produces:

1. Leaderboard output for decision-making
2. Trace files showing step-by-step system execution
3. Failure logs with input, expected output, actual output, and error context

These artifacts help explain results to leadership and support audit reviews.

## Credentials and provider behavior

The framework auto-loads .env from repository root.

Minimum real-provider setup:

1. OPENAI_API_KEY=your_key_here

If no provider key is available, it falls back to MockProvider for local testing.

## Practical use guidance

Use this framework to choose a technical baseline implementation for a recurring task.

Do not use this framework alone to claim operational impact outcomes.

## How New Workflows Can Be Added

The current implementation is set up so new workflow variants can be added with limited code changes.

In practice, adding a new workflow means:

1. Define a new workflow profile with its own prompt style and behavior rules.
2. Register a new workflow class that points to that profile.
3. Run the same dataset and compare the new result against the others.

This makes it easier to test new strategies without rebuilding the full benchmark each time.
