"""
bonus_models_analysis.py
--------------------------
EXTENSION to Alansari & Luqman (2025), AraHalluEval.

The paper's Table 1 / Table 2 report results for 12 models. However, the
authors' released gold-annotation CSVs (AraHalluEval_QA.csv,
AraHalluEval_Summarization.csv) already contain fully annotated outputs
for TWO ADDITIONAL models that never appear in the paper's tables or
discussion:

    - Maverick      (meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)
    - DeepSeek-v3    (deepseek-ai/DeepSeek-V3)

The paper's own "Future Work" (Conclusion, Section 5) explicitly calls
for "expanding the evaluation to include additional open-source models."
This script does exactly that, using data the original authors already
annotated but did not report on -- no new generation or annotation is
required, which keeps this extension fully grounded in the original
authors' own methodology and labels.

We:
  1. Compute the same hallucination score tables (Table 1 / Table 2 style)
     for Maverick and DeepSeek-v3.
  2. Place them in the paper's original leaderboard for direct comparison.
  3. Test whether Allam's reported advantage (lowest hallucination score)
     still holds against these newer/larger models.
"""

import pandas as pd
import numpy as np

QA_PATH = "../original_repo/AraHalluEval_QA.csv"
SUMM_PATH = "../original_repo/AraHalluEval_Summarization.csv"

QA_FACTUAL_COLS = [
    "Named-Entity Hallucination",
    "Temporal/Number Hallucination",
    "Factual Contradiction",
    "Conflict Hallucination",
    "K0wledge Source Conflict",
    "Grammar Hallucination",
    "Generic/Imprecise Hallucination",
]
QA_FAITHFUL_COLS = ["Instruction Inconsistency", "Code-Switching"]

SUMM_FACTUAL_COLS = ["Temporal/Number Error", "Named-Entity Error", "Fabrication", "Inference", "Grammar Error"]
SUMM_FAITHFUL_COLS = ["Instruction Inconsistency", "Context Inconsistency", "Code-switch"]

# The paper's original Table 1 averages, for reference/comparison
PAPER_QA_AVERAGES = {
    "Allam": 0.382, "Fanar": 0.508, "Jais-6.7b": 0.777, "Noon": 0.763,
    "Gemma": 0.645, "bloom-7b": 0.730, "llama": 0.542, "qwen2.5-7b": 0.655,
    "DeepSeek-r1": 0.377, "GPT-4o": 0.235, "GPT-o3": 0.255, "Qwq": 0.471,
}

PAPER_SUMM_AVERAGES = {
    "Allam": 0.215, "Fanar": 1.172, "Jais-6.7b": 0.638, "Noon": 0.743,
    "Gemma": 0.925, "bloom-7b": 0.783, "llama": 0.515, "qwen2.5-7b": 0.477,
    "DeepSeek-r1": 0.245, "gpt-4o": 0.105, "gpt-o3": 0.145, "Qwq": 0.730,
}


def yn_to_binary(series):
    """Handles both 0/1 numeric columns (QA CSV) and Yes/No string columns (Summ CSV).

    Note: pandas may read text columns as the nullable StringDtype rather than
    plain 'object', so we detect by attempting the Yes/No map first instead of
    checking dtype directly.
    """
    mapped = series.map({"Yes": 1, "No": 0})
    if mapped.notna().any():
        return mapped.fillna(0).astype(float)
    return pd.to_numeric(series, errors="coerce").fillna(0)


def analyze_qa_bonus_models():
    print("=" * 70)
    print("GQA TASK -- extending the leaderboard with Maverick & DeepSeek-v3")
    print("=" * 70)

    qa = pd.read_csv(QA_PATH)
    qa["model"] = qa["model"].replace({"0on": "Noon"})
    qa = qa[qa["model"].isin(list(PAPER_QA_AVERAGES.keys()) + ["Maverick", "DeepSeek-v3"])]

    for col in QA_FACTUAL_COLS + QA_FAITHFUL_COLS:
        qa[col] = yn_to_binary(qa[col])

    qa["factual_total"] = qa[QA_FACTUAL_COLS].sum(axis=1)
    qa["faithful_total"] = qa[QA_FAITHFUL_COLS].sum(axis=1)
    qa["avg_score"] = qa[QA_FACTUAL_COLS + QA_FAITHFUL_COLS].mean(axis=1) * len(QA_FACTUAL_COLS + QA_FAITHFUL_COLS) / 2
    # Note: paper's "Average" = mean(Total factual per-sample rate, Total faithfulness per-sample rate)
    # We approximate consistently with Table 1's construction below.

    per_model_factual = qa.groupby("model")[QA_FACTUAL_COLS].mean().sum(axis=1)
    per_model_faithful = qa.groupby("model")[QA_FAITHFUL_COLS].mean().sum(axis=1)
    per_model_avg = (per_model_factual + per_model_faithful) / 2

    leaderboard = pd.DataFrame({
        "factual_total": per_model_factual,
        "faithfulness_total": per_model_faithful,
        "average_score": per_model_avg,
    }).sort_values("average_score")

    print(leaderboard.round(3))
    print()

    rank = leaderboard["average_score"].rank()
    print(f"Allam's rank among all models (1 = best): {rank.get('Allam', 'N/A')}")
    print(f"Maverick's rank: {rank.get('Maverick', 'N/A')} | score = {leaderboard.loc['Maverick','average_score']:.3f}" if "Maverick" in leaderboard.index else "Maverick not found")
    print(f"DeepSeek-v3's rank: {rank.get('DeepSeek-v3', 'N/A')} | score = {leaderboard.loc['DeepSeek-v3','average_score']:.3f}" if "DeepSeek-v3" in leaderboard.index else "DeepSeek-v3 not found")
    print()
    return leaderboard


def analyze_summ_bonus_models():
    print("=" * 70)
    print("SUMMARIZATION TASK -- extending the leaderboard")
    print("=" * 70)

    summ = pd.read_csv(SUMM_PATH)
    summ = summ[summ["model"].isin(list(PAPER_SUMM_AVERAGES.keys()) + ["Maverick", "DeepSeek-v3"])]

    for col in SUMM_FACTUAL_COLS + SUMM_FAITHFUL_COLS:
        summ[col] = yn_to_binary(summ[col])

    summ["hallucination_density"] = pd.to_numeric(summ["hallucination_density"], errors="coerce")

    leaderboard = summ.groupby("model").agg(
        factual_rate=("hallucination_density", "mean"),
        faithful_total=(SUMM_FAITHFUL_COLS[0], "mean"),  # placeholder; expand below
    )
    # Faithfulness total = mean rate across the 3 faithfulness columns, summed
    faith = summ.groupby("model")[SUMM_FAITHFUL_COLS].mean().sum(axis=1)
    leaderboard["faithful_total"] = faith
    leaderboard["average_score"] = (leaderboard["factual_rate"] + leaderboard["faithful_total"]) / 2
    leaderboard = leaderboard.sort_values("average_score")

    print(leaderboard.round(3))
    print()
    return leaderboard


if __name__ == "__main__":
    qa_lb = analyze_qa_bonus_models()
    summ_lb = analyze_summ_bonus_models()
    print("Interpretation guide for your report:")
    print("- If Maverick/DeepSeek-v3 beat Allam -> the paper's 'Arabic pretraining")
    print("  wins' finding may not generalize to newer/larger multilingual models,")
    print("  which COMPLICATES the original conclusion.")
    print("- If Allam still wins -> this CONFIRMS and strengthens the original")
    print("  finding even against stronger, more recent competitors.")
