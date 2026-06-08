#!/usr/bin/env python3
"""
Generate combined LaTeX tables for STE results (U=1..10).

Expected result layout:
{persistent_dir}/results/
  self-description-STE-{dataset}-{z_attr}/
    U_{k}/
      {model_name_or_path}/{debiasing}/{bias_type}/
        *.json   # each json is a list[dict] OR dict with metrics
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

# ----- configuration (edit if you add datasets / bias types later) -----

# % \multicolumn{10}{l}{\textcolor{black}{\logoso \textbf{ Stackoverflow}}} \\
# % \multicolumn{10}{l}{\textcolor{black}{\logoms \textbf{ Mathematics}}} \\
# % \multicolumn{10}{l}{\textcolor{black}{\logosu \textbf{ SuperUser}}} \\
# % \multicolumn{10}{l}{\textcolor{black}{\logoes \textbf{ English}}} \\

DATASETS_ORDER = [
    ("stackoverflow", r"\textcolor{black}{\logoso \textbf{Stackoverflow}}"),
    ("mathematics", r"\textcolor{black}{\logoms \textbf{Mathematics}}"),
    ("superuser", r"\textcolor{black}{\logosu \textbf{SuperUser}}"),
    ("english", r"\textcolor{black}{\logoes \textbf{English}}"),
]

# Column order in the final table
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


# ----- helpers -----

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
    """
    Returns mean of:
      main_before, main_after, tpr_before, tpr_after   (all in percent, not fraction)
    """
    if not exps:
        return None

    mb, ma, tb, ta = [], [], [], []
    for e in exps:
        if ("main_task_accuracy_before" not in e or "main_task_accuracy_after" not in e or
            "tpr_gap_before" not in e or "tpr_gap_after" not in e):
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
    # match your old behavior: equality -> plain, increase -> \uag, decrease -> \dab
    delta = after - before
    if abs(delta) < 0.05:
        return f"{after:.1f}"
    if delta > 0:
        return rf"\uag{{{abs(delta):.1f}}} {after:.1f}"
    return rf"\dab{{{abs(delta):.1f}}} {after:.1f}"


def pretty_tpr(after: float, before: float) -> str:
    # equality -> plain, increase (worse) -> \ua, decrease (better) -> \da
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


# ----- LaTeX generation -----

def latex_table_for_k(
    k: int,
    persistent_dir: str,
    results_root: str,
    model_list: List[str],
    debiasing: str,
    bias_type: str,
    split: str,
) -> str:
    """
    Build one table* for a single U=k with both datasets, all models, and all Z_attr columns.
    """

    # Header
    lines: List[str] = []
    lines.append(r"\begin{table*}[th!]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{ll rr  rr  rr  rr}")
    lines.append(r"\toprule")
    lines.append(
        r"Target & Language Model & \multicolumn{2}{c}{Baseline} & "
        r"\multicolumn{2}{c}{SAL (random)} & \multicolumn{2}{c}{SAL (explicit)} & \multicolumn{2}{c}{SAL (implicit)} \\"
    )
    lines.append(
        r"& & Main & TPR-Gap & Main & TPR-Gap & Main & TPR-Gap & Main & TPR-Gap \\"
    )
    lines.append(r"\midrule")

    # Body: dataset sections
    for dataset, dataset_title in DATASETS_ORDER:
        # section title spanning all 10 columns
        lines.append(rf"\multicolumn{{10}}{{l}}{{{dataset_title}}} \\")
        lines.append(r"\addlinespace[2pt]")

        for model_name_or_path in model_list:
            model_disp = short_model_name(model_name_or_path)

            # load metrics for each z_attr
            by_z: Dict[str, Optional[Dict[str, float]]] = {}
            for z_attr, _colname in Z_ATTR_ORDER:
                leaf = find_leaf_dir(
                    persistent_dir, results_root, dataset, z_attr, k,
                    model_name_or_path, debiasing, split, bias_type
                )
                by_z[z_attr] = load_metrics_for_setting(leaf)

            # choose a baseline from first available z_attr
            baseline_src = None
            for z_attr, _ in Z_ATTR_ORDER:
                if by_z[z_attr] is not None:
                    baseline_src = by_z[z_attr]
                    break

            if baseline_src is None:
                # no data at all for this dataset/model/k
                row = rf"EN & {model_disp} & -- & -- & -- & -- & -- & -- & -- & -- \\"
                lines.append(row)
                continue

            base_main = baseline_src["main_before"]
            base_tpr = baseline_src["tpr_before"]

            # SAL columns in required order: random, explicit, implicit
            cells: List[str] = []
            cells.append(f"{base_main:.1f}")
            cells.append(f"{base_tpr:.1f}")

            for z_attr, _colname in Z_ATTR_ORDER:
                m = by_z[z_attr]
                if m is None:
                    cells.extend(["--", "--"])
                else:
                    # compare AFTER to baseline BEFORE (as in your final table)
                    cells.append(pretty_main(m["main_after"], base_main))
                    cells.append(pretty_tpr(m["tpr_after"], base_tpr))

            row = rf"EN & {model_disp} & " + " & ".join(cells) + r" \\"
            lines.append(row)

        lines.append(r"\midrule")

    # Footer
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        rf"\caption*{{Table: Control experiment on the Stackoverflow and Mathematics dataset comparing two bias-removal settings:"
        rf" (i) erasure based on binary demographic attributes and (ii) erasure based on contextual information from self-descriptions."
        rf" \textbf{{Main}} is helpfulness prediction accuracy."
        rf" \textbf{{TPR-Gap}} is the difference in true positive rates between demographic groups."
        rf" $U$ denotes the number of eigenvectors removed by SAL (here $U={k}$).}}"
    )
    lines.append(rf"\label{{tab:STE-{bias_type}-U{k}}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines) + "\n"


def parse_model_list(s: str) -> List[str]:
    # Accept either pipe-separated or comma-separated.
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
    else:
        parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persistent_dir", type=str, required=True)
    ap.add_argument("--model_name_or_path", type=str, required=True,
                    help='Models as "|" separated (recommended) or comma separated.')
    ap.add_argument("--debiasing", type=str, default="SAL")
    ap.add_argument("--bias_type", type=str, default="reputation",
                    help='e.g. "reputation" or "reputation|location"')
    ap.add_argument("--split", type=str, default="disjoint",
                        help='e.g. "disjoint" or "joint|disjoint"')
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument("--out_dir", type=str, default="results/summary_textual_erasure/STE_tables_clean")
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
            # generate table(s)
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
                out_path = os.path.join(out_dir, f"STE_table_U_{k}_i-{bias_type}_s-{split}_i-{bias_type}.tex")
                with open(out_path, "w") as f:
                    f.write(tex)
                print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
