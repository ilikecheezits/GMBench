# Datasets

This folder contains the test cases used to compare systems.

For reviewers, think of each dataset as the benchmark's source of truth.

## Current dataset

1. Module: food_pantry_intake.py
2. Dataset ID: food_pantry_intake_v1

## What each example contains

1. id: unique case identifier
2. input_text: the raw task input
3. ground_truth: the expected structured answer
4. metadata: tags for slicing and analysis

## Why metadata matters

Metadata allows reporting by segment, such as:

1. organization
2. language
3. difficulty
4. prompt_injection
5. reference_model

## Data quality expectations

1. Keep examples anonymized and safe.
2. Keep ground truth consistent and reviewable.
3. Add new datasets in this folder and register them so the benchmark can discover them.
