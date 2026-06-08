#!/usr/bin/env python3
"""
Generate one LaTeX table per bias x split for STE results, using user-selected K values
for SAL (random), SAL (explicit), and SAL (implicit).

Compared with STE_combined.py, this script does NOT emit one table per k. Instead,
it picks exactly one k for each method and bias type, and prints a single table
containing all datasets and all models.

Expected layout:
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

# final table column order
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

BIAS_DISPLAY = {
    "name": "Name",
    "location": "Country",
    "reputation": "Reputation",
    "engagement": "Engagement",
    "gender": "Gender",
}

SPLIT_DISPLAY = {
    "joint": "Joint Split",
    "disjoint": "Disjoint Split",
    "natural_joint": "Natural Joint Split",
    "natural_disjoint": "Natural Disjoint Split",
}

SPLIT_NOTE = {
    "joint": r"\textbf{User-overlap (joint) split}: the same users may appear in both the training and test sets.",
    "disjoint": r"\textbf{User-disjoint split}: users in the training and test sets do not overlap.",
    "natural_joint": r"\textbf{Natural user-overlap split}: the same users may appear in both the training and test sets.",
    "natural_disjoint": r"\textbf{Natural user-disjoint split}: users in the training and test sets do not overlap.",
}


def short_model_name(model_name_or_path: str) -> str:
    return MODEL_ALIAS.get(model_name_or_path, model_name_or_path.split("/")[-1])


def load_json_as_list(path: str) -> List[dict]:
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError(f"Unexpected JSON type in {path}: {type(obj)}")


def mean_metrics(exps: List[dict]) -> Optional[Dict[str, float]]:
    """Returns means in percent."""
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


def parse_model_list(s: str) -> List[str]:
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
    else:
        parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


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


def get_k_for_bias_method(args: argparse.Namespace, bias_type: str, method_key: str) -> int:
    return int(getattr(args, f"k_{method_key}_{bias_type}"))


def latex_table_for_selected_ks(
    persistent_dir: str,
    results_root: str,
    model_list: List[str],
    debiasing: str,
    bias_type: str,
    split: str,
    args: argparse.Namespace,
) -> str:
    method_to_k = {
        "random_textual": get_k_for_bias_method(args, bias_type, "random"),
        "binary": get_k_for_bias_method(args, bias_type, "explicit"),
        "textual": get_k_for_bias_method(args, bias_type, "implicit"),
    }

    lines: List[str] = []
    lines.append(r"\begin{table*}[th!]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{l rr  rr  rr  rr}")
    lines.append(r"\toprule")
    lines.append(
        r"Language Model & \multicolumn{2}{c}{Baseline} & "
        r"\multicolumn{2}{c}{SAL (random)} & \multicolumn{2}{c}{SAL (explicit)} & \multicolumn{2}{c}{SAL (implicit)} \\")
    lines.append(
        r"& Main & TPR-Gap & Main & TPR-Gap & Main & TPR-Gap & Main & TPR-Gap \\")
    lines.append(r"\midrule")

    for dataset_idx, (dataset, dataset_title) in enumerate(DATASETS_ORDER):
        lines.append(rf"\multicolumn{{9}}{{l}}{{{dataset_title}}} \\")
        lines.append(r"\addlinespace[2pt]")

        for model_name_or_path in model_list:
            model_disp = short_model_name(model_name_or_path)

            by_z: Dict[str, Optional[Dict[str, float]]] = {}
            for z_attr, _colname in Z_ATTR_ORDER:
                leaf = find_leaf_dir(
                    persistent_dir=persistent_dir,
                    results_root=results_root,
                    dataset=dataset,
                    z_attr=z_attr,
                    k=method_to_k[z_attr],
                    model_name_or_path=model_name_or_path,
                    debiasing=debiasing,
                    split=split,
                    bias_type=bias_type,
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
                    cells.extend(["", ""])
                else:
                    cells.append(pretty_main(m["main_after"], base_main))
                    cells.append(pretty_tpr(m["tpr_after"], base_tpr))

            lines.append(rf"{model_disp} & " + " & ".join(cells) + r" \\")

        if dataset_idx != len(DATASETS_ORDER) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")

    bias_disp = BIAS_DISPLAY.get(bias_type, bias_type.title())
    split_disp = SPLIT_DISPLAY.get(split, split.replace("_", " ").title())
    k_random = method_to_k["random_textual"]
    k_explicit = method_to_k["binary"]
    k_implicit = method_to_k["textual"]

    lines.append(
        rf"\caption{{\textbf{{{bias_disp} {split_disp}}} debiasing on StackOverflow, Mathematics, SuperUser and English. "
        rf"Main = helpfulness accuracy; TPR-Gap = generalized disparity ($2 \times$ StdDev) of true positive rates across groups, equivalent to the difference in true positive rates in the binary case. "
        rf"SAL uses $K_{{\text{{random}}}}={k_random}$, $K_{{\text{{explicit}}}}={k_explicit}$, and $K_{{\text{{implicit}}}}={k_implicit}$. "
        rf"{SPLIT_NOTE.get(split, '')}}}"
    )
    lines.append(rf"\label{{tab:{bias_type}-{split}-selected-k}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persistent_dir", type=str, required=True)
    ap.add_argument("--model_name_or_path", type=str, required=True,
                    help='Models as "|" separated (recommended) or comma separated.')
    ap.add_argument("--debiasing", type=str, default="SAL")
    ap.add_argument("--bias_type", type=str, default="reputation",
                    help='e.g. "reputation" or "reputation|location|engagement|name|gender"')
    ap.add_argument("--split", type=str, default="disjoint",
                    help='e.g. "disjoint" or "joint|disjoint|natural_joint|natural_disjoint"')
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument("--out_dir", type=str, default="results/summary_textual_erasure/STE_tables_selected_k")

    # random defaults
    ap.add_argument("--k_random_name", type=int, default=50)
    ap.add_argument("--k_random_location", type=int, default=50)
    ap.add_argument("--k_random_reputation", type=int, default=50)
    ap.add_argument("--k_random_engagement", type=int, default=50)
    ap.add_argument("--k_random_gender", type=int, default=25)

    # explicit defaults
    ap.add_argument("--k_explicit_name", type=int, default=24)
    ap.add_argument("--k_explicit_location", type=int, default=5)
    ap.add_argument("--k_explicit_reputation", type=int, default=2)
    ap.add_argument("--k_explicit_engagement", type=int, default=2)
    ap.add_argument("--k_explicit_gender", type=int, default=2)

    # implicit defaults
    ap.add_argument("--k_implicit_name", type=int, default=50)
    ap.add_argument("--k_implicit_location", type=int, default=50)
    ap.add_argument("--k_implicit_reputation", type=int, default=50)
    ap.add_argument("--k_implicit_engagement", type=int, default=50)
    ap.add_argument("--k_implicit_gender", type=int, default=50)
    return ap


def main() -> None:
    ap = build_parser()
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

            tex = latex_table_for_selected_ks(
                persistent_dir=args.persistent_dir,
                results_root=args.results_root,
                model_list=models,
                debiasing=args.debiasing,
                bias_type=bias_type,
                split=split,
                args=args,
            )

            out_path = os.path.join(out_dir, f"STE_selected_k_table_s-{split}_i-{bias_type}.tex")
            with open(out_path, "w") as f:
                f.write(tex)
            print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()