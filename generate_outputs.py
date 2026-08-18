"""
generate_outputs.py
--------------------
OPTIONAL empirical reproduction step: generates NEW model outputs on a
small sample of your own GQA/summarization questions, using the ORIGINAL
paper's exact prompts and decoding hyperparameters (Appendix E / F).

This is meant to run on a GPU (Colab recommended, matching the paper's
own setup: A100, Colab Pro, fp16).

USAGE (run from inside the `reproduction/` folder, with `original_repo/`
one level up, e.g. after `git clone` of your project repo):

    python3 generate_outputs.py --model Allam --task qa --sample_csv my_qa_sample.csv

`my_qa_sample.csv` must have a `question_text` column (for task=qa) or a
`text` column (for task=summarization) -- matching the original repo's
`run.py` conventions exactly, so we can reuse their code unmodified.

WHY THIS SCRIPT EXISTS:
The original repo's run.py expects specific pre-named input files
(QA_merged_predictions.csv, summ_merged_predictions.csv). This wrapper
lets you point it at YOUR OWN small sample file instead, while reusing
their model-loading and inference logic unmodified for a faithful
reproduction of their generation setup.

Before running, fill in your API keys in original_repo/models.py:
    login(token="YOUR_HF_TOKEN")
    openai.api_key = "YOUR_OPENAI_KEY"
    together.api_key = "YOUR_TOGETHER_KEY"
"""

import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm

# Import the ORIGINAL authors' code, unmodified, for a faithful reproduction
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "original_repo"))
from models import load_model_and_tokenizer, MODEL_MAP        # noqa: E402
from inference import get_response                            # noqa: E402

# Paper's exact prompts (Appendix E, Figure 7)
PROMPTS = {
    "qa": "اجب على السؤال التالي باللغة العربية",
    "summarization": "لخص النص الآتي في جملة واحدة فقط، وأجب باللغة العربية",
}

TEXT_COLUMN = {"qa": "question_text", "summarization": "text"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--task", required=True, choices=["qa", "summarization"])
    parser.add_argument("--sample_csv", required=True,
                         help="Your own small sample CSV (see module docstring for required column).")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.sample_csv)
    text_col = TEXT_COLUMN[args.task]
    if text_col not in df.columns:
        raise ValueError(f"Expected column '{text_col}' in {args.sample_csv} for task='{args.task}'")

    print(f"Loading {args.model} ...")
    model, tokenizer, device = load_model_and_tokenizer(args.model)

    responses = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{args.model} / {args.task}"):
        prompt = f"{PROMPTS[args.task]}:\n{row[text_col]}\n"
        response = get_response(prompt, model, tokenizer, device, args.model, args.task, debug=False)
        responses.append(response)

    df["response"] = responses

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.task}_{args.model.replace(':', '_')}_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} generated outputs to {out_path}")
    print("Next step: manually annotate these outputs using the paper's Figure 3")
    print("indicator definitions, following the same Yes/No labeling scheme as")
    print("the original AraHalluEval_QA.csv / AraHalluEval_Summarization.csv files.")


if __name__ == "__main__":
    main()
