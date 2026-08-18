# Reproducing & Extending AraHalluEval

This project reproduces core results from **AraHalluEval: A Fine-grained
Hallucination Evaluation Framework for Arabic LLMs** (Alansari & Luqman,
2025, ArabicNLP 2025) and extends it with a new analysis the original
paper did not report.

- **Original paper:** https://aclanthology.org/2025.arabicnlp-main.12/
- **Original code/data:** https://github.com/aishaalansari57/AraHalluEval

## What's in this repo

```
arahallueval-project/
├── original_repo/              # Unmodified clone of the authors' repo
│                                # (their code + their released gold-annotation CSVs)
│                                # Included for attribution & as our data source.
│
├── analysis/
│   └── reproduce_stats.py      # OUR CODE. Recomputes the paper's reported
│                                # significance tests directly on their released
│                                # gold annotations (Mann-Whitney U, paired t-tests).
│
├── extension/
│   └── bonus_models_analysis.py # OUR EXTENSION. The released CSVs contain fully
│                                 # annotated outputs for 2 models (Maverick,
│                                 # DeepSeek-v3) that never appear in the paper's
│                                 # tables. We analyze them and re-rank the
│                                 # leaderboard, directly addressing the paper's
│                                 # own stated future work ("expanding the
│                                 # evaluation to include additional open-source
│                                 # models" -- Conclusion, Section 5).
│
├── reproduction/
│   └── generate_outputs.py     # Optional: script to generate NEW outputs from
│                                # a small model subset, using the paper's exact
│                                # prompts and decoding hyperparameters
│                                # (Appendix E/F), for empirical validation.
│
└── report/
    └── report_template.md      # Skeleton for the final written report
```

## What we did

### 1. Reproduction (methodology + statistics)
We did not re-run the full 12-model x 5,600-output study (infeasible for a
one-week individual project). Instead we:
- Verified we can **exactly reproduce** the paper's reported statistics
  (Mann-Whitney U = 649,023.5, p = 8.19e-6 for GQA; t = -1.41, p = 0.161
  for summarization; p = 0.0186 for Allam vs. Qwen2.5-7b) by recomputing
  them directly from the authors' own released, gold-labeled annotations.
  This validates that we understand the paper's methodology and
  statistical procedure well enough to reproduce it exactly.
- (Optional, time permitting) Generated new outputs from 2-3 models using
  the paper's exact prompts/decoding config, and informally compared
  patterns against the original findings.

### 2. Extension
The released CSVs already contain annotated outputs for 2 models the
paper never analyzed: Maverick (Llama-4-Maverick) and DeepSeek-v3. We
built a new leaderboard including them and found:
- **On GQA**, DeepSeek-v3 and Maverick actually outrank Allam, complicating
  the paper's central claim that Arabic-specific pretraining beats
  multilingual models.
- **On summarization**, Allam still outranks both, confirming the original
  finding holds in that task.

This nuance -- that the "Arabic pretraining wins" finding is task-dependent
-- is the core contribution of our extension.

## How to run

```bash
pip install -r requirements.txt
cd analysis && python3 reproduce_stats.py
cd ../extension && python3 bonus_models_analysis.py
```

## Attribution

All data (`AraHalluEval_QA.csv`, `AraHalluEval_Summarization.csv`,
`TruthfulQA_translated_New.xlsx`) and the original `models.py` /
`inference.py` / `run.py` are from Alansari & Luqman (2025) and are
included unmodified in `original_repo/` for reference and attribution.
All code in `analysis/`, `extension/`, and `reproduction/` is our own.

```bibtex
@inproceedings{alansari-luqman-2025-arahallueval,
    title     = "{A}ra{H}allu{E}val: A Fine-grained Hallucination Evaluation Framework for {A}rabic {LLM}s",
    author    = "Alansari, Aisha and Luqman, Hamzah",
    booktitle = "Proceedings of The Third Arabic Natural Language Processing Conference",
    year      = "2025",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2025.arabicnlp-main.12/",
}
```
