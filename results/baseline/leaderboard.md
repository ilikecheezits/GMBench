# Nonprofit AI Benchmark Leaderboard

Composite weights: core 0.60, knowledge 0.10, truthfulness 0.10, fairness 0.10, calibrated_refusal 0.10. Judge: `openai/gpt-4.1-mini`.

| Rank | Model | Composite | 95% CI | Nonprofit core tasks | Domain knowledge (MMLU) | Truthfulness (TruthfulQA) | Bias/fairness (BBQ) | Calibrated refusal (XSTest) |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | openai/gpt-5 | **0.898** | 0.869–0.928 | 0.928 | 0.911 | 0.880 | 0.889 | 0.733 |
| 2 | openai/gpt-5-mini | **0.877** | 0.841–0.911 | 0.902 | 0.893 | 0.840 | 0.907 | 0.725 |
| 3 | openai/gpt-5-nano | **0.870** | 0.837–0.899 | 0.912 | 0.821 | 0.800 | 0.889 | 0.717 |
| 4 | openai/gpt-4.1 | **0.824** | 0.779–0.863 | 0.806 | 0.804 | 0.780 | 0.889 | 0.925 |
| 5 | openai/gpt-4.1-mini | **0.792** | 0.750–0.837 | 0.755 | 0.786 | 0.800 | 0.889 | 0.908 |
| 6 | openai/gpt-4o-mini | **0.737** | 0.692–0.780 | 0.674 | 0.839 | 0.780 | 0.852 | 0.858 |

## Core suite by task area

| Model | fundraising strategy | grant writing | operations prioritization | community communications | impact measurement | board governance | compliance and privacy | finance and accounting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| openai/gpt-4.1 | 0.874 | 0.902 | 0.800 | 0.896 | 0.653 | 0.882 | 0.853 | 0.591 |
| openai/gpt-4.1-mini | 0.733 | 0.686 | 0.860 | 0.842 | 0.621 | 0.844 | 0.816 | 0.643 |
| openai/gpt-4o-mini | 0.675 | 0.770 | 0.645 | 0.669 | 0.421 | 0.680 | 0.862 | 0.666 |
| openai/gpt-5 | 0.988 | 0.927 | 0.956 | 0.974 | 0.904 | 0.972 | 0.983 | 0.721 |
| openai/gpt-5-mini | 0.981 | 0.902 | 0.910 | 0.942 | 0.872 | 0.980 | 0.820 | 0.806 |
| openai/gpt-5-nano | 0.906 | 0.936 | 0.921 | 0.947 | 0.910 | 0.980 | 0.903 | 0.789 |

## Stress subsets and scorer split

| Model | Traps | Over-refusal | Exact-answer | High-risk | Judge mean | Objective mean |
|---|---:|---:|---:|---:|---:|---:|
| openai/gpt-4.1 | 0.630 | 0.920 | 0.669 | 0.852 | 0.816 | 0.791 |
| openai/gpt-4.1-mini | 0.558 | 0.864 | 0.719 | 0.804 | 0.754 | 0.760 |
| openai/gpt-4o-mini | 0.422 | 0.975 | 0.709 | 0.647 | 0.648 | 0.683 |
| openai/gpt-5 | 0.869 | 1.000 | 0.774 | 0.972 | 0.961 | 0.880 |
| openai/gpt-5-mini | 0.935 | 0.754 | 0.831 | 0.876 | 0.939 | 0.851 |
| openai/gpt-5-nano | 0.891 | 0.880 | 0.855 | 0.914 | 0.948 | 0.860 |

## Diagnostics

- Graded samples: 216 of 1536 total.
- Judge vs objective Spearman rank correlation (graded samples): 0.384
- ⚠️ Judge model openai/gpt-4.1-mini is also a candidate; its core/custom scores may carry self-preference bias.

Sample score = 0.60 × rubric judge + 0.40 × deterministic checks. Core score weights each task area equally unless a profile says otherwise. CIs are stratified bootstrap over samples (1,000 draws).
