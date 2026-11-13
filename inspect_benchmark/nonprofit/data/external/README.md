# External benchmark slices

Fixed-seed subsets (seed `20260821`) regenerated with `python main.py fetch-external`.
Only the items listed here are used; see `external.py` for the selection code.

| File | Source (Hugging Face) | Items | License | Citation |
|---|---|---:|---|---|
| mmlu_nonprofit.jsonl | cais/mmlu — professional_accounting, business_ethics, management, marketing, public_relations, sociology, high_school_statistics (8 each) | 56 | MIT | Hendrycks et al., ICLR 2021 |
| truthfulqa_mc1.jsonl | truthfulqa/truthful_qa, multiple_choice (MC1, answer order shuffled) | 50 | Apache-2.0 | Lin et al., ACL 2022 |
| bbq.jsonl | heegyu/bbq — 9 categories × (3 ambiguous + 3 disambiguated) | 54 | CC-BY-4.0 | Parrish et al., ACL Findings 2022 |
| xstest.jsonl | Paul/XSTest — 30 safe + 30 unsafe prompts | 60 | CC-BY-4.0 | Röttger et al., NAACL 2024 |
