#!/usr/bin/env python3
"""
Generate combined LaTeX tables for STE results (U=1..10).

Expected result layout:
{persistent_dir}/results/
  STE-{dataset}-{split}-{z_attr}/
    U_{k}/
      {model_name_or_path}/{debiasing}/{bias_type}/
        *.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional


DATASETS_ORDER = [
    ("stackoverflow", r"\textcolor{black}{\logoso \textbf{Stackoverflow}}"),
    ("mathematics", r"\textcolor{black}{\logoms \textbf{Mathematics}}"),
    ("superuser", r"\textcolor{black}{\logosu \textbf{SuperUser}}"),
    ("english", r"\textcolor{black}{\logoes \textbf{English}}"),
]

Z_ATTR_ORDER = [
    ("random_textual", "SAL (random)"),
    ("binary", "SAL (explicit)"),
    ("textual", "SAL (implicit)"),
]

MODEL_ALIAS = {
    "bert-base-uncased": "BERT",
    "meta-llama/Llama-3.1-8B": "Llama3.1-8B",
    "mistralai/Mistral-7B-v0.3": "Mistral-7B",
}


def short_model_name(model_name_or_path: str) -> str:
    return MODEL_ALIAS.get(model_name_or_path, model_name_or_path.split("/")[-1])


def pretty_split_name(split: str) -> str:
    mapping = {
        "joint": "User-overlap",
        "disjoint": "User-disjoint",
        "natural_joint": "Natural User-overlap",
        "natural_disjoint": "Natural User-disjoint",
    }
    return mapping.get(split, split.replace("_", " ").title())


def split_description(split: str) -> str:
    mapping = {
        "joint": r"\textbf{User-overlap (joint) split}: the same users may appear in both the training and test sets.",
        "disjoint": r"\textbf{User-disjoint split}: users in the training and test sets are non-overlapping.",
        "natural_joint": r"\textbf{Natural user-overlap split}: the naturally imbalanced split where the same users may appear in both the training and test sets.",
        "natural_disjoint": r"\textbf{Natural user-disjoint split}: the naturally imbalanced split where users in the training and test sets are non-overlapping.",
    }
    return mapping.get(split, "")


def pretty_bias_name(bias_type: str) -> str:
    mapping = {
        "reputation": "Reputation",
        "location": "Location",
        "engagement": "Engagement",
        "name": "Name",
    }
    return mapping.get(bias_type, bias_type.replace("_", " ").title())


def make_caption(split: str, bias_type: str, k: int) -> str:
    title = f"\\textbf{{{pretty_bias_name(bias_type)} {pretty_split_name(split)} Split}} debiasing on StackOverflow, Mathematics, SuperUser and English ($K={k}$)."
    body_1 = (
        r"Main = helpfulness accuracy; TPR-Gap = generalized disparity "
        r"($2 \times$ StdDev) of true positive rates across groups (equivalent to difference for binary)."
    )
    body_2 = r"SAL removes the top $K$ singular directions from the representation projection."
    body_3 = split_description(split)
    return "\n".join([title, body_1, body_2, body_3]).strip()


def make_label(split: str, bias_type: str) -> str:
    return f"tab:{bias_type}-{split.replace('_', '-')}"


def load_json_as_list(path: str) -> List[dict]:
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError(f"Unexpected JSON type in {path}: {type(obj)}")


def mean_metrics(exps: List[dict]) -> Optional[Dict[str, float]]:
    if not exps:
        return None

    mb, ma, tb, ta = [], [], [], []
    for e in exps:
        if (
            "main_task_accuracy_before" not in e
            or "main_task_accuracy_after" not in e
            or "tpr_gap_before" not in e
            or "tpr_gap_after" not in e
        ):
            continue

        mb.append(float(e["main_task_accuracy_before"]) * 100.0)
        ma.append(float(e["main_task_accuracy_after"]) * 100.0)
        tb.append(float(e["tpr_gap_before"]) * 100.0)
        ta.append(float(e["tpr_gap_after"]) * 100.0)

    if not mb:
        return None

    return {
        "main_before": sum(mb) / len(mb),
        "main_after": sum(ma) / len(ma),
        "tpr_before": sum(tb) / len(tb),
        "tpr_after": sum(ta) / len(ta),
    }


def pretty_main(after: float, before: float) -> str:
    delta = after - before
    if abs(delta) < 0.05:
        return f"{after:.1f}"
    if delta > 0:
        return rf"\uag{{{abs(delta):.1f}}} {after:.1f}"
    return rf"\dab{{{abs(delta):.1f}}} {after:.1f}"


def pretty_tpr(after: float, before: float) -> str:
    delta = after - before
    if abs(delta) < 0.05:
        return f"{after:.1f}"
    if delta > 0:
        return rf"\ua{{{abs(delta):.1f}}} {after:.1f}"
    return rf"\da{{{abs(delta):.1f}}} {after:.1f}"


def find_leaf_dir(
    persistent_dir: str,
    results_root: str,
    dataset: str,
    z_attr: str,
    k: int,
    model_name_or_path: str,
    debiasing: str,
    split: str,
    bias_type: str,
) -> str:
    return os.path.join(
        persistent_dir,
        results_root,
        f"STE-{dataset}-{split}-{z_attr}",
        f"U_{k}",
        model_name_or_path,
        debiasing,
        bias_type,
    )


def load_metrics_for_setting(leaf_dir: str) -> Optional[Dict[str, float]]:
    json_paths = sorted(glob.glob(os.path.join(leaf_dir, "*.json")))
    if not json_paths:
        return None
    exps: List[dict] = []
    for p in json_paths:
        exps.extend(load_json_as_list(p))
    return mean_metrics(exps)


def latex_table_for_k(
    k: int,
    persistent_dir: str,
    results_root: str,
    model_list: List[str],
    debiasing: str,
    bias_type: str,
    split: str,
) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table*}[th!]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{l rr  rr  rr  rr}")
    lines.append(r"\toprule")
    lines.append(
        r"Language Model & \multicolumn{2}{c}{Baseline} & "
        r"\multicolumn{2}{c}{SAL (random)} & \multicolumn{2}{c}{SAL (explicit)} & \multicolumn{2}{c}{SAL (implicit)} \\"
    )
    lines.append(
        r"& Main & TPR-Gap & Main & TPR-Gap & Main & TPR-Gap & Main & TPR-Gap \\"
    )
    lines.append(r"\midrule")

    for dataset_idx, (dataset, dataset_title) in enumerate(DATASETS_ORDER):
        lines.append(rf"\multicolumn{{9}}{{l}}{{{dataset_title}}} \\")
        lines.append(r"\addlinespace[2pt]")

        for model_name_or_path in model_list:
            model_disp = short_model_name(model_name_or_path)

            by_z: Dict[str, Optional[Dict[str, float]]] = {}
            for z_attr, _colname in Z_ATTR_ORDER:
                leaf = find_leaf_dir(
                    persistent_dir,
                    results_root,
                    dataset,
                    z_attr,
                    k,
                    model_name_or_path,
                    debiasing,
                    split,
                    bias_type,
                )
                by_z[z_attr] = load_metrics_for_setting(leaf)

            baseline_src = None
            for z_attr, _ in Z_ATTR_ORDER:
                if by_z[z_attr] is not None:
                    baseline_src = by_z[z_attr]
                    break

            if baseline_src is None:
                row = rf"{model_disp} & -- & -- & -- & -- & -- & -- & -- & -- \\"
                lines.append(row)
                continue

            base_main = baseline_src["main_before"]
            base_tpr = baseline_src["tpr_before"]

            cells: List[str] = [f"{base_main:.1f}", f"{base_tpr:.1f}"]
            for z_attr, _colname in Z_ATTR_ORDER:
                m = by_z[z_attr]
                if m is None:
                    cells.extend(["--", "--"])
                else:
                    cells.append(pretty_main(m["main_after"], base_main))
                    cells.append(pretty_tpr(m["tpr_after"], base_tpr))

            row = rf"{model_disp} & " + " & ".join(cells) + r" \\"
            lines.append(row)

        if dataset_idx < len(DATASETS_ORDER) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(rf"\caption{{{make_caption(split, bias_type, k)}}}")
    lines.append(rf"\label{{{make_label(split, bias_type)}}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines) + "\n"


def parse_model_list(s: str) -> List[str]:
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
    else:
        parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persistent_dir", type=str, required=True)
    ap.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help='Models as "|" separated (recommended) or comma separated.',
    )
    ap.add_argument("--debiasing", type=str, default="SAL")
    ap.add_argument(
        "--bias_type",
        type=str,
        default="reputation",
        help='e.g. "reputation" or "reputation|location"',
    )
    ap.add_argument(
        "--split",
        type=str,
        default="disjoint",
        help='e.g. "disjoint" or "joint|disjoint"',
    )
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="results/summary_textual_erasure/STE_tables_clean",
    )
    ap.add_argument("--k_min", type=int, default=1)
    ap.add_argument("--k_max", type=int, default=10)
    args = ap.parse_args()

    bias_types = [x.strip() for x in args.bias_type.split("|") if x.strip()]
    splits = [x.strip() for x in args.split.split("|") if x.strip()]
    models = parse_model_list(args.model_name_or_path)
    if not models:
        raise ValueError("No models parsed from --model_name_or_path")

    for split in splits:
        for bias_type in bias_types:
            out_dir = os.path.join(args.out_dir, split, bias_type)
            os.makedirs(out_dir, exist_ok=True)

            for k in range(args.k_min, args.k_max + 1):
                tex = latex_table_for_k(
                    k=k,
                    persistent_dir=args.persistent_dir,
                    results_root=args.results_root,
                    model_list=models,
                    debiasing=args.debiasing,
                    split=split,
                    bias_type=bias_type,
                )
                out_path = os.path.join(
                    out_dir,
                    f"STE_table_U_{k}_i-{bias_type}_s-{split}_i-{bias_type}.tex",
                )
                with open(out_path, "w") as f:
                    f.write(tex)
                print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
