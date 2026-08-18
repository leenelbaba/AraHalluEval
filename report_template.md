# [Your Title Here, e.g. "Reproducing and Extending AraHalluEval: How Robust Is the Arabic-Pretraining Advantage?"]

## 1. Project Title & Abstract
*(~150 words)*

We reproduce and extend AraHalluEval (Alansari & Luqman, 2025), a
fine-grained hallucination evaluation framework for Arabic LLMs across
generative question answering (GQA) and summarization. We independently
recompute the paper's reported significance tests directly from their
released gold-annotation data, exactly matching all three reported
statistics. We then extend the study using two models (Maverick,
DeepSeek-v3) present in the authors' released annotations but never
discussed in the paper, directly addressing the paper's own stated
future work. Our extension finds that the paper's central claim --
Arabic-pretrained models like Allam hallucinate less than multilingual
models -- holds on summarization but does NOT hold on GQA, where
DeepSeek-v3 and Maverick outrank Allam. [Add 1-2 sentences on
implications.]

## 2. Introduction & Problem Statement
- Motivate: hallucination in Arabic LLMs is underexplored relative to English.
- State the paper's core claims you're testing.
- State your specific question: does the "Arabic pretraining reduces
  hallucination" finding generalize to models not in the original study?

## 3. Methodology (Approach / Reproduction Details)

### 3.1 Original paper's methodology (summarize, don't copy verbatim)
- 12-indicator fine-grained hallucination taxonomy (7 factuality + 2
  faithfulness for GQA; 5 factuality + 3 faithfulness + density + rating
  for summarization).
- 300 TyDiQA-GoldP-AR samples, 100 XLSum samples, 12 models, two
  annotators + adjudicator.
- Significance testing: Mann-Whitney U (GQA, Arabic vs multilingual),
  paired t-test (summarization density; TruthfulQA Arabic vs English).

### 3.2 Our reproduction
- We used the authors' own released, gold-labeled annotation CSVs
  (`AraHalluEval_QA.csv`, `AraHalluEval_Summarization.csv`) rather than
  re-running the full 5,600-output human annotation process, which is
  infeasible at individual/one-week scale.
- We recomputed their reported statistical tests independently in
  `analysis/reproduce_stats.py`.
- [If you did new generation]: describe which models, how many samples,
  exact prompts/decoding settings used (cite Appendix E/F), and your own
  annotation process.

### 3.3 Our extension
- The released CSVs include full annotations for Maverick and
  DeepSeek-v3 -- models never covered in the paper's Table 1/2.
- We recomputed the same leaderboard construction (factuality total +
  faithfulness total, averaged) for all 14 models including these two.

## 4. Implementation Details & Results

### 4.1 Statistical reproduction (exact match)

| Test | Paper's reported value | Our reproduced value |
|---|---|---|
| GQA Mann-Whitney U (Arabic vs Multilingual) | U=649,023.5, p=8.19e-6 | U=649023.5, p=8.195e-06 |
| Summarization paired t-test (Arabic vs Multilingual) | t=-1.41, p=0.161 | t=-1.412, p=0.161 |
| Summarization paired t-test (Allam vs Qwen2.5-7b) | p=0.0186 | t=-2.393, p=0.0186 |

*(All three exactly match the paper's reported statistics, confirming we
correctly understood and reproduced the analysis methodology.)*

### 4.2 Extension: extended leaderboard

**GQA task** (lower = less hallucination):

| Rank | Model | Factual total | Faithful total | Average |
|---|---|---|---|---|
| 1 | GPT-4o | 0.253 | 0.067 | 0.160 |
| 2 | DeepSeek-r1 | 0.337 | 0.090 | 0.213 |
| 3 | **DeepSeek-v3 (new)** | 0.343 | 0.213 | 0.278 |
| 4 | **Maverick (new)** | 0.470 | 0.120 | 0.295 |
| 5 | GPT-o3 | 0.683 | 0.027 | 0.355 |
| 6 | **Allam** | 0.727 | 0.037 | **0.382** |
| ... | (remaining 8 models) | | | |

**Summarization task**:

| Rank | Model | Factual rate | Faithful total | Average |
|---|---|---|---|---|
| 1 | gpt-4o | 0.021 | 0.110 | 0.066 |
| 2 | gpt-o3 | 0.032 | 0.130 | 0.081 |
| 3 | **Allam** | 0.066 | 0.220 | **0.143** |
| 4 | DeepSeek-r1 | 0.075 | 0.290 | 0.182 |
| 5 | **DeepSeek-v3 (new)** | 0.096 | 0.370 | 0.233 |
| 6 | **Maverick (new)** | 0.141 | 0.360 | 0.250 |
| ... | (remaining 8 models) | | | |

*(Insert your full tables/figures here -- consider a bar chart comparing
Allam vs. the 2 new models across both tasks.)*

## 5. Discussion & Analysis

- **Reproduction fidelity**: exact statistical match validates our
  understanding of the paper's methodology.
- **Task-dependent generalization**: Allam's advantage over multilingual/
  newer models holds on summarization but not GQA. Discuss possible
  reasons: [e.g., GQA requires broader world/factual knowledge where
  larger frontier-scale models (Maverick 17B active, DeepSeek-v3 671B
  MoE) may have an edge despite lacking Arabic-specific pretraining;
  summarization is more about faithfulness to a given context, where
  Allam's Arabic fluency may matter more than raw scale].
- **Limitations**: single annotator (us) vs. the paper's two annotators +
  adjudicator; no new human annotation was performed for the extension
  (we relied on the original authors' existing labels for Maverick/
  DeepSeek-v3, which is a strength for consistency but means we cannot
  independently verify those specific labels).
- **Future work**: dialectal Arabic, larger/newer model coverage,
  topic-fairness slicing.

## 6. Reflection on Learnings

- What was rewarding: e.g., seeing an exact statistical match validate
  understanding of significance testing methodology.
- Challenges: e.g., a `StringDtype` bug initially caused a silent
  all-zero faithfulness computation in the extension analysis, caught by
  sanity-checking outputs against the raw CSV values -- illustrates the
  importance of validating intermediate results rather than trusting
  code that runs without errors.
- What you'd do differently with more time: full new-model generation
  and annotation, dialectal Arabic testing, more statistical rigor
  (e.g., inter-annotator agreement if you had a second annotator).
