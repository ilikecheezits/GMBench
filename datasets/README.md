# Datasets

This folder contains the test cases used to compare systems.

For reviewers, think of each dataset as the benchmark's source of truth.

The dataset is what all workflows are judged against.
If two workflows get different scores, this dataset is the shared reference that makes that comparison fair.

## Current datasets

1. Module: nonprofit_tool_matching.py
2. Dataset ID: nonprofit_tool_matching_v1
3. Focus: nonprofit AI tool/package recommendation with multilingual and high-risk coverage
4. Module: food_pantry_intake.py
5. Dataset ID: food_pantry_intake_v1
6. Focus: pantry intake structuring from free text

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

For the nonprofit tool matching track, metadata is especially important for:

1. cross-language consistency checks
2. high-risk scenario robustness checks
3. organization-level fairness checks

## Data quality expectations

1. Keep examples anonymized and safe.
2. Keep ground truth consistent and reviewable.
3. Add new datasets in this folder and register them so the benchmark can discover them.
