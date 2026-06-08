import argparse
import glob
import json
import os
from typing import Dict, Tuple

DATASET_ORDER = ["stackoverflow", "mathematics", "superuser", "english"]
MODEL_ORDER = ["bert-base-uncased", "meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-v0.3"]
BIAS_ORDER = ["reputation", "location", "engagement"]

DATASET_DISPLAY = {
    "stackoverflow": r"\logoso StackOverflow",
    "mathematics": r"\logoms Mathematics",
    "superuser": r"\logosu SuperUser",
    "english": r"\logoes English",
}
MODEL_DISPLAY = {
    "bert-base-uncased": "BERT",
    "meta-llama/Llama-3.1-8B": "Llama3.1-8B",
    "mistralai/Mistral-7B-v0.3": "Mistral-7B",
}


def pick_json(result_dir: str, prefer: str) -> str:
    paths = sorted(glob.glob(os.path.join(result_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No json files found in {result_dir}")

    preferred = []
    fallback = []
    for p in paths:
        name = os.path.basename(p)
        if prefer == "mlp" and "predict_z_from_x_mlp_raw_unique_balanced" in name:
            preferred.append(p)
        elif prefer == "logreg" and "predict_z_from_x_logreg_raw_unique_balanced" in name:
            preferred.append(p)
        else:
            fallback.append(p)

    candidates = preferred if preferred else fallback
    if len(candidates) > 1:
        candidates = sorted(candidates, key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_score(path: str) -> float:
    with open(path, "r") as f:
        data = json.load(f)
    best = data["best_result"]
    return float(best["test_accuracy"]) * 100.0


def build_latex(results_root: str, experiment: str, split: str, prefer: str) -> str:
    values: Dict[Tuple[str, str, str], str] = {}
    for dataset in DATASET_ORDER:
        for model_name in MODEL_ORDER:
            for bias_type in BIAS_ORDER:
                result_dir = os.path.join(
                    results_root,
                    f"{experiment}-{dataset}-{split}",
                    model_name,
                    bias_type,
                )
                try:
                    path = pick_json(result_dir, prefer=prefer)
                    score = load_score(path)
                    values[(dataset, model_name, bias_type)] = f"{score:.1f}"
                except FileNotFoundError:
                    values[(dataset, model_name, bias_type)] = ""

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{ll rrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Language Model & \multicolumn{3}{c}{Natural Disjoint-split Accuracy} \\")
    lines.append(r"& & Reputation & Location & Engagement \\")
    lines.append(r"\midrule")
    lines.append("")

    for d_i, dataset in enumerate(DATASET_ORDER):
        for m_i, model_name in enumerate(MODEL_ORDER):
            rep = values[(dataset, model_name, "reputation")]
            loc = values[(dataset, model_name, "location")]
            eng = values[(dataset, model_name, "engagement")]
            if m_i == 0:
                lines.append(
                    rf"\multirow{{3}}{{*}}{{{DATASET_DISPLAY[dataset]}}} & {MODEL_DISPLAY[model_name]} & {rep} & {loc} & {eng} \\"
                )
            else:
                lines.append(
                    rf" & {MODEL_DISPLAY[model_name]} & {rep} & {loc} & {eng} \\"
                )
        if d_i != len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")
            lines.append("")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{\textbf{Predictability of protected attributes from profile-text representations.} We report test accuracy for predicting the protected attribute $\rv{Z}$ from the representation $\rv{H}$ of users' self-descriptions under the natural disjoint split. Higher accuracy indicates that the embedding retains more information about the attribute, implying stronger attribute leakage from textual representations.}"
    )
    lines.append(r"\label{fig:accuracy-on-self-description}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="results/analysis_on_Z_aboutme_cls_raw_unique_balanced")
    parser.add_argument("--experiment", type=str, default="STE")
    parser.add_argument("--split", type=str, default="raw_unique_balanced")
    parser.add_argument("--prefer", type=str, default="mlp", choices=["mlp", "logreg"])
    parser.add_argument("--output_tex", type=str, required=True)
    args = parser.parse_args()

    latex = build_latex(
        results_root=args.results_root,
        experiment=args.experiment,
        split=args.split,
        prefer=args.prefer,
    )
    os.makedirs(os.path.dirname(args.output_tex) or ".", exist_ok=True)
    with open(args.output_tex, "w") as f:
        f.write(latex + "\n")
    print(f"Saved LaTeX table to: {args.output_tex}")
