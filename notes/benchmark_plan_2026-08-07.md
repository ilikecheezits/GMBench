# Benchmark Expansion Plan

## Summary

The current benchmark should be expanded into a more rigorous, realistic, and externally credible evaluation framework for nonprofit AI use cases.

## Goals

1. Make the current benchmark more rigorous
   - Improve both objective scoring and LLM-judge scoring so they better reflect real nonprofit workflows.
   - Use more realistic evaluation signals, especially for nonprofit tasks where practical decision-making matters more than generic reasoning.
   - Treat the current dummy dataset as a starting point, while acknowledging that real-world nonprofit data is still limited.

2. Turn the project into a "benchmark of benchmarks"
   - Incorporate existing external benchmarks and evaluation frameworks where appropriate.
   - Use them as scoring components or reference signals.
   - Combine them into a final evaluation through weighted scoring.
   - For now, use fixed weights for simplicity.

3. Create a more adaptive evaluation experience
   - A nonprofit worker or manager should be able to input their organization context.
   - A specialized agent should help interpret that input, ask follow-up questions, and refine the benchmark setup.
   - The system should use that information to:
     - generate or structure the benchmark prompt/tasks for the models, and
     - assign preference-based weights or priorities relevant to the specific nonprofit context.

## Proposed Product Direction

### User experience
- A nonprofit user provides organizational context and needs.
- A specialized agent asks a short set of clarifying questions.
- The agent translates that information into benchmark inputs and evaluation priorities.

### Benchmark design direction
- Make prompts and scoring more realistic for nonprofit decision-making.
- Expand beyond current static task prompts toward more context-aware evaluation.
- Use both deterministic and human/LLM-judge signals, but make them more robust and realistic.

## Strategic purpose

The immediate goal is not only to build a better benchmark, but also to make the evaluation objectives clearer and more compelling for researchers and AI companies.

This should help the project gain attention and communicate a strong, credible value proposition.
