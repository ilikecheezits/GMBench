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

### Example record shape

Nonprofit tool matching example:

```json
{
	"id": "tool_match_001",
	"input_text": "Budget: $80. Need help with volunteer scheduling. Data sensitivity: low. Main pain point: volunteers miss shifts.",
	"ground_truth": {
		"recommended_package": "volunteer_coordination_basic",
		"primary_tool_category": "volunteer_coordination",
		"activation_horizon_days": 14,
		"estimated_monthly_cost_usd": 48,
		"risk_level": "low",
		"success_metric": "volunteer_shift_fill_rate"
	},
	"metadata": {
		"organization": "example_org",
		"language": "en",
		"difficulty": "medium"
	}
}
```

Food pantry intake example:

```json
{
	"id": "pantry_intake_001",
	"input_text": "Caller Amina Rahman, family of 6, needs groceries and baby formula. Bengali interpreter needed.",
	"ground_truth": {
		"client_name": "Amina Rahman",
		"household_size": 6,
		"urgency": "high",
		"requested_services": ["groceries", "baby formula"],
		"preferred_language": "Bengali",
		"notes_summary": "High-need household of 6 requesting groceries, baby formula."
	},
	"metadata": {
		"organization": "example_pantry",
		"language": "en",
		"difficulty": "medium"
	}
}
```

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
