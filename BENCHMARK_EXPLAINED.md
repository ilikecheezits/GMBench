# Good Model Benchmark Explained

This document explains how the benchmark works from input to final decision.

## What this benchmark is for

The benchmark evaluates one nonprofit AI deployment and answers:

1. Is this deployment technically reliable?
2. Is it actually improving operations?
3. Is there real evidence for those improvements?
4. Can the nonprofit maintain it over time?
5. Can it generalize to other contexts?
6. Is it safe and governed well enough?

It then gives:

- pillar scores
- an overall score
- a phase decision (Not Ready, Ready for Alpha, Ready for Beta, Ready for Labs)
- governance pass or block
- claim confidence level

## What you provide as input

A deployment object includes:

1. Technical metrics
Examples: schema validity, factual correctness, latency, API failures.

2. Operational impact metrics
Examples: time saved, workload reduction, adoption rate, satisfaction.

3. Continuity metrics
Examples: documentation quality, maintenance complexity, onboarding effort.

4. Governance metrics
Examples: privacy risk, legal compliance, auditability, security posture.

5. Generalization signals
Examples: cross-organization robustness, cross-language robustness.

6. Capacity evidence
Examples: baseline vs follow-up task time, error rates, sample size, measurement period, evidence source.

7. Governance profile
Examples: data sensitivity tier, vulnerable population flag, human-in-the-loop, incident response plan, contract controls.

8. Stage
One of: Needs Assessment, AI Setup, Workflow Automation, Custom Build.

## How scoring works

### 1) Metric normalization

Most numeric metrics are normalized to a 0 to 100 scale.

- If higher is better, higher values map upward.
- If lower is better (like error rates), the score is inverted.

Each metric also has:

- min and max bounds
- a weight
- optional required flag

### 2) Pillar scores

Each pillar computes a weighted average of its metric scores:

1. Technical Reliability
2. Operational Impact
3. Evidence and Capacity Delta
4. Continuity
5. Generalization
6. Risk and Governance

### 3) Evidence and Capacity Delta pillar

This pillar checks whether improvements are both real and trustworthy.

It combines:

- evidence source reliability (system logs are stronger than estimates)
- task time improvement
- error rate improvement
- staff confidence change
- continuity improvement
- sample size confidence
- measurement duration

If evidence is weak (for example estimate-only, tiny sample, short duration), notes are added and the score drops.

### 4) Overall score

Overall score is a weighted blend of the six pillar scores.

The exact weights are policy-driven and can be changed at runtime.

## Safety and governance hard gates

Before final phase decision, governance hard gates run.

Examples of blocking conditions:

1. Deployment is marked as excluded use case.
2. High or critical sensitivity without human-in-the-loop.
3. PII storage without contract controls.
4. Vulnerable-population workflow without incident response plan.
5. High-sensitivity governance score below policy minimum.
6. Critical-tier data retention above policy maximum.

If any block is triggered:

- governance gate = BLOCKED
- phase result forced to Not Ready
- claim confidence forced very low

## Phase gate logic

If governance passes, phase progression is evaluated.

### Alpha requires

- minimum impact score
- no critical technical failure
- no critical governance failure
- case study evidence exists

### Beta requires

- Alpha pass
- minimum organization evidence count
- onboarding succeeded
- documentation exists
- minimum technical score
- minimum governance score

### Labs requires

- Beta pass
- minimum generalization score
- high enough adoption
- low enough maintenance complexity
- minimum continuity score
- higher governance threshold
- repeated success count threshold

All thresholds are policy-configurable.

## Stage caps (maturity guardrails)

Even if raw scores are high, stage can cap max phase:

1. Needs Assessment caps at Not Ready
2. AI Setup caps at Ready for Alpha
3. Workflow Automation caps at Ready for Beta
4. Custom Build can reach Ready for Labs

This prevents over-claiming maturity too early.

## Claim confidence

Claim confidence summarizes trust in the deployment claim.

It is computed from:

- evidence score
- governance score

Then mapped to labels:

- High
- Moderate
- Low
- Very Low

If governance is blocked, confidence is automatically very low.

## Compare and generalize modes

### Compare

You can run multiple deployments and get:

- sorted report list by overall score
- average score across the set

### Generalize

For one workflow name, the benchmark aggregates historical deployments and computes:

- workflow-level generalization score
- organization coverage
- candidate-for-labs flag

Candidate-for-labs is based on policy thresholds for:

- generalization score
- minimum number of deployments
- minimum number of organizations

## Runtime configuration (swap without code edits)

Hardcoded values have been moved into policy configuration.

You can change behavior through:

1. policy file override
Set environment variable GMBENCH_POLICY_FILE to a JSON file.

2. selected environment overrides
Examples include alpha impact threshold and governance sensitivity thresholds.

This means you can tune logic by domain, region, or partner requirements without changing core code.

## What the final report contains

A report includes:

1. Each pillar score and metric details
2. Overall score
3. Capacity delta score
4. Claim confidence
5. Governance gate status and reasons
6. Final phase result

## Practical interpretation guide

Use this quick decision guide:

1. Governance blocked?
Stop and remediate safety/governance first.

2. Impact high but evidence low?
Improve instrumentation and measurement quality before scaling.

3. Good single deployment but weak generalization?
Replicate in more organizations before Labs claims.

4. Strong scores but low continuity?
Invest in documentation, handoff, and maintenance simplification.

5. Phase capped by stage?
Advance deployment maturity before claiming higher phase readiness.

## Typical flow end to end

1. Build deployment input with metrics, evidence, governance profile, and stage.
2. Run benchmark.
3. Review pillar scores and notes.
4. Check governance gate first.
5. Check phase result and stage cap effects.
6. Use confidence level to judge trust in the claim.
7. Compare across deployments and monitor trend over time.
