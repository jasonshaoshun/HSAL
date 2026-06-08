import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Tuple

DATASET_ORDER = ["stackoverflow", "mathematics", "superuser", "english"]
BIAS_ORDER = ["reputation", "location", "engagement"]
SPLIT_ORDER = ["natural_joint", "natural_disjoint", "joint", "disjoint"]
MODEL_ORDER = ["bert-base-uncased", "meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-v0.3"]

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

SPLIT_TITLE = {
    "natural_joint": "Natural Joint-split Accuracy",
    "natural_disjoint": "Natural Disjoint-split Accuracy",
    "joint": "Joint-split Accuracy",
    "disjoint": "Disjoint-split Accuracy",
}

SPLIT_LABEL = {
    "natural_joint": "tab:accuracy-on-self-description-natural-joint",
    "natural_disjoint": "tab:accuracy-on-self-description-natural-disjoint",
    "joint": "tab:accuracy-on-self-description-joint",
    "disjoint": "tab:accuracy-on-self-description-disjoint",
}


def fmt(acc):
    if acc is None:
        return "-"
    return f"{100.0 * float(acc):.1f}"


def load_results(results_root: str) -> List[dict]:
    pattern = os.path.join(results_root, "**", "*.json")
    records = []
    for path in glob.glob(pattern, recursive=True):
        with open(path, "r") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            obj["_path"] = path
            records.append(obj)
    return records


def infer_dataset_split_model_bias_from_path(path: str) -> Tuple[str | None, str | None, str | None, str | None]:
    parts = path.replace("\\", "/").split("/")
    try:
        idx = parts.index("analysis_on_Z_simple")
    except ValueError:
        return None, None, None, None
    if len(parts) <= idx + 5:
        return None, None, None, None
    # results/analysis_on_Z_simple/STE/dataset/split/model/bias/file.json
    dataset = parts[idx + 2]
    split = parts[idx + 3]
    model_name_or_path = parts[idx + 4]
    bias_type = parts[idx + 5]
    return dataset, split, model_name_or_path, bias_type


def build_index(records: List[dict]) -> Tuple[Dict[tuple, dict], List[dict]]:
    index: Dict[tuple, dict] = {}
    csv_rows: List[dict] = []

    for rec in records:
        dataset = rec.get("dataset")
        split = rec.get("split")
        model_name_or_path = rec.get("model_name_or_path")
        bias_type = rec.get("bias_type")

        if not all([dataset, split, model_name_or_path, bias_type]):
            p_dataset, p_split, p_model, p_bias = infer_dataset_split_model_bias_from_path(rec.get("_path", ""))
            dataset = dataset or p_dataset
            split = split or p_split
            model_name_or_path = model_name_or_path or p_model
            bias_type = bias_type or p_bias

        if not all([dataset, split, model_name_or_path, bias_type]):
            continue

        key = (dataset, split, model_name_or_path, bias_type)
        index[key] = rec

        best = rec.get("best_result", {})
        dedup = rec.get("deduplication", {})
        split_sizes = rec.get("split_sizes", {})
        csv_rows.append({
            "dataset": dataset,
            "split": split,
            "model_name_or_path": model_name_or_path,
            "bias_type": bias_type,
            "train_accuracy": best.get("train_accuracy"),
            "dev_accuracy": best.get("dev_accuracy"),
            "test_accuracy": best.get("test_accuracy"),
            "best_params": json.dumps(best.get("params", {}), sort_keys=True),
            "train_raw_n": dedup.get("train_raw_n"),
            "test_raw_n": dedup.get("test_raw_n"),
            "train_dedup_n": dedup.get("train_dedup_n"),
            "test_dedup_n": dedup.get("test_dedup_n"),
            "n_train": split_sizes.get("n_train"),
            "n_dev": split_sizes.get("n_dev"),
            "n_test": split_sizes.get("n_test"),
            "result_path": rec.get("_path", ""),
        })

    return index, csv_rows


def render_table_for_split(index: Dict[tuple, dict], split_name: str) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll rrr}")
    lines.append(r"\toprule")
    lines.append(rf"Dataset & Language Model & \multicolumn{{3}}{{c}}{{{SPLIT_TITLE[split_name]}}} \\")
    lines.append(r"& & Reputation & Location & Engagement \\")
    lines.append(r"\midrule")
    lines.append("")

    for d_i, dataset in enumerate(DATASET_ORDER):
        first_row = True
        for model_name in MODEL_ORDER:
            cells = []
            for bias in BIAS_ORDER:
                rec = index.get((dataset, split_name, model_name, bias))
                acc = None if rec is None else rec.get("best_result", {}).get("test_accuracy")
                cells.append(fmt(acc))

            dataset_cell = rf"\multirow{{3}}{{*}}{{{DATASET_DISPLAY[dataset]}}}" if first_row else ""
            lines.append(
                f"{dataset_cell} & {MODEL_DISPLAY[model_name]} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
            first_row = False

        if d_i != len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")
            lines.append("")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{\textbf{Intrinsic Bias in Self-Descriptions.} We report the bias prediction accuracy for the User-overlap (Joint) and User-disjoint (Disjoint) splits. Here, prediction accuracy serves as a proxy for attribute leakage; a higher score indicates that the textual embeddings (implicit $Z$) are strongly correlated with the protected attribute.}"
    )
    lines.append(rf"\label{{{SPLIT_LABEL[split_name]}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    records = load_results(args.results_root)
    index, csv_rows = build_index(records)

    os.makedirs(args.output_dir, exist_ok=True)
    for split_name in SPLIT_ORDER:
        out_path = os.path.join(args.output_dir, f"ste_predict_z_from_x_{split_name}.tex")
        with open(out_path, "w") as f:
            f.write(render_table_for_split(index, split_name))

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        fieldnames = [
            "dataset", "split", "model_name_or_path", "bias_type",
            "train_accuracy", "dev_accuracy", "test_accuracy", "best_params",
            "train_raw_n", "test_raw_n", "train_dedup_n", "test_dedup_n",
            "n_train", "n_dev", "n_test", "result_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"Loaded {len(records)} result files from {args.results_root}")
    print(f"Wrote split-specific LaTeX tables to: {args.output_dir}")
    print(f"Wrote CSV summary to: {args.output_csv}")
