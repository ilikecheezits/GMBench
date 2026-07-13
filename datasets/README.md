# Datasets

This directory contains benchmark dataset modules used by the evaluation framework.

## Current dataset modules

1. food_pantry_intake.py

Registered dataset id:

1. food_pantry_intake_v1

## Dataset contract

Each dataset module should register a builder with register_dataset and return a BenchmarkDataset.

Each example should provide:

1. id
2. input_text
3. ground_truth
4. metadata

Common metadata keys used by built-in metrics include:

1. organization
2. language
3. difficulty
4. prompt_injection
5. reference_model

## Notes

1. Keep examples anonymized and safe for local development.
2. Prefer deterministic, reviewable ground-truth outputs.
3. Add new dataset modules in this folder and register them so discovery can load them.
