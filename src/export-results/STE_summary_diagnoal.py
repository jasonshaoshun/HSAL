#!/usr/bin/env python3
"""
Generate STE summary LaTeX tables.

Each table fixes one split and one model view:
  - BERT
  - Llama3.1-8B
  - Mistral-7B
  - Average (mean across the three models only)

Within each table:
  - rows cover all bias types x all datasets
  - columns show Baseline, SAL (random), SAL (explicit), SAL (implicit)
  - each SAL variant can use its own K for each bias type
  - missing results are left blank

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

BIAS_ORDER = [
    ("name", "Name"),
    ("gender", "Gender"),
    ("location", "Country"),
    ("reputation", "Reputation"),
    ("engagement", "Engagement"),
]

MODEL_ALIAS = {
    "bert-base-uncased": "BERT",
    "meta-llama/Llama-3.1-8B": "Llama3.1-8B",
    "mistralai/Mistral-7B-v0.3": "Mistral-7B",
}

SPLIT_DESC = {
    "joint": r"\textbf{User-overlap (joint) split}: the same users may appear in both the training and test sets.",
    "disjoint": r"\textbf{User-disjoint split}: users in the training and test sets are non-overlapping.",
    "natural_joint": r"\textbf{Natural user-overlap split}: the naturally imbalanced split where the same users may appear in both the training and test sets.",
    "natural_disjoint": r"\textbf{Natural user-disjoint split}: the naturally imbalanced split where users in the training and test sets are non-overlapping.",
}

DEFAULT_KS = {
    "random": {"name": 10, "gender": 25, "location": 10, "reputation": 10, "engagement": 10},
    "explicit": {"name": 2, "gender": 2, "location": 5, "reputation": 2, "engagement": 2},
    "implicit": {"name": 10, "gender": 50, "location": 10, "reputation": 10, "engagement": 10},
}


def parse_pipe_or_comma_list(s: str) -> List[str]:
    if not s:
        return []
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
    else:
        parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def short_model_name(model_name_or_path: str) -> str:
    return MODEL_ALIAS.get(model_name_or_path, model_name_or_path.split("/")[-1])


def short_split_name(split: str) -> str:
    mapping = {
        "joint": "joint",
        "disjoint": "disjoint",
        "natural_joint": "natural-joint",
        "natural_disjoint": "natural-disjoint",
    }
    return mapping.get(split, split.replace("_", "-"))


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


def average_metric_dicts(metric_dicts: List[Optional[Dict[str, float]]]) -> Optional[Dict[str, float]]:
    valid = [m for m in metric_dicts if m is not None]
    if not valid:
        return None
    return {
        "main_before": sum(m["main_before"] for m in valid) / len(valid),
        "main_after": sum(m["main_after"] for m in valid) / len(valid),
        "tpr_before": sum(m["tpr_before"] for m in valid) / len(valid),
        "tpr_after": sum(m["tpr_after"] for m in valid) / len(valid),
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


def blank_cells(n: int) -> List[str]:
    return [""] * n


def get_k_by_bias(args: argparse.Namespace, variant: str, bias_type: str) -> int:
    specific_name = f"k_{variant}_{bias_type}"
    value = getattr(args, specific_name)
    if value is not None:
        return value
    return DEFAULT_KS[variant][bias_type]


def format_k_summary(args: argparse.Namespace, variant: str) -> str:
    pieces = []
    for bias_type, bias_label in BIAS_ORDER:
        pieces.append(f"{bias_label}={get_k_by_bias(args, variant, bias_type)}")
    return ", ".join(pieces)


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


def load_metrics(
    persistent_dir: str,
    results_root: str,
    dataset: str,
    split: str,
    bias_type: str,
    model_name_or_path: str,
    debiasing: str,
    z_attr: str,
    k: int,
) -> Optional[Dict[str, float]]:
    leaf = find_leaf_dir(
        persistent_dir=persistent_dir,
        results_root=results_root,
        dataset=dataset,
        z_attr=z_attr,
        k=k,
        model_name_or_path=model_name_or_path,
        debiasing=debiasing,
        split=split,
        bias_type=bias_type,
    )
    return load_metrics_for_setting(leaf)


def gather_row_metrics_for_model(
    persistent_dir: str,
    results_root: str,
    dataset: str,
    split: str,
    bias_type: str,
    model_name_or_path: str,
    debiasing: str,
    k_random: int,
    k_explicit: int,
    k_implicit: int,
) -> Dict[str, Optional[Dict[str, float]]]:
    return {
        "random_textual": load_metrics(
            persistent_dir, results_root, dataset, split, bias_type,
            model_name_or_path, debiasing, "random_textual", k_random,
        ),
        "binary": load_metrics(
            persistent_dir, results_root, dataset, split, bias_type,
            model_name_or_path, debiasing, "binary", k_explicit,
        ),
        "textual": load_metrics(
            persistent_dir, results_root, dataset, split, bias_type,
            model_name_or_path, debiasing, "textual", k_implicit,
        ),
    }


def gather_row_metrics_average(
    persistent_dir: str,
    results_root: str,
    dataset: str,
    split: str,
    bias_type: str,
    model_list: List[str],
    debiasing: str,
    k_random: int,
    k_explicit: int,
    k_implicit: int,
) -> Dict[str, Optional[Dict[str, float]]]:
    out: Dict[str, Optional[Dict[str, float]]] = {}
    for z_attr, k in [
        ("random_textual", k_random),
        ("binary", k_explicit),
        ("textual", k_implicit),
    ]:
        per_model = [
            load_metrics(
                persistent_dir, results_root, dataset, split, bias_type,
                model_name_or_path, debiasing, z_attr, k,
            )
            for model_name_or_path in model_list
        ]
        out[z_attr] = average_metric_dicts(per_model)
    return out


def baseline_from_by_z(by_z: Dict[str, Optional[Dict[str, float]]]) -> Optional[Dict[str, float]]:
    for key in ["random_textual", "binary", "textual"]:
        if by_z.get(key) is not None:
            return by_z[key]
    return None


def cells_from_by_z(by_z: Dict[str, Optional[Dict[str, float]]]) -> List[str]:
    baseline_src = baseline_from_by_z(by_z)
    if baseline_src is None:
        return blank_cells(8)

    base_main = baseline_src["main_before"]
    base_tpr = baseline_src["tpr_before"]
    cells: List[str] = [f"{base_main:.1f}", f"{base_tpr:.1f}"]

    for z_attr in ["random_textual", "binary", "textual"]:
        m = by_z.get(z_attr)
        if m is None:
            cells.extend(["", ""])
        else:
            cells.append(pretty_main(m["main_after"], base_main))
            cells.append(pretty_tpr(m["tpr_after"], base_tpr))
    return cells


def make_caption(model_display: str, split: str, args: argparse.Namespace) -> str:
    model_phrase = (
        "Average across BERT, Llama3.1-8B, and Mistral-7B"
        if model_display == "Average"
        else model_display
    )
    body = [
        rf"\textbf{{{model_phrase} leaderboard on implicit and explicit debiasing across five types of demographic information and four different domains.}}",
        r"Main = helpfulness accuracy; TPR-Gap = generalized disparity ($2 \times$ StdDev) of true positive rates across groups (equivalent to difference for binary).",
        rf"SAL uses bias-specific choices of singular directions: random [{format_k_summary(args, 'random')}], explicit [{format_k_summary(args, 'explicit')}], and implicit [{format_k_summary(args, 'implicit')}].",
        SPLIT_DESC.get(split, ""),
    ]
    return "\n".join([x for x in body if x])


def make_label(model_display: str, split: str) -> str:
    model_key = model_display.lower().replace(".", "").replace(" ", "-")
    model_key = model_key.replace("/", "-")
    return f"tab:ste-summary-{model_key}-{short_split_name(split)}"


def latex_table(
    persistent_dir: str,
    results_root: str,
    model_list: List[str],
    model_view: str,
    debiasing: str,
    split: str,
    args: argparse.Namespace,
) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table*}[th!]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{ll rr  rr  rr  rr}")
    lines.append(r"\toprule")
    lines.append(
        r"Bias Type & Dataset & \multicolumn{2}{c}{Baseline} & "
        r"\multicolumn{2}{c}{SAL (random)} & \multicolumn{2}{c}{SAL (explicit)} & \multicolumn{2}{c}{SAL (implicit)} \\"
    )
    lines.append(
        r"& & Main & TPR-Gap & Main & TPR-Gap & Main & TPR-Gap & Main & TPR-Gap \\"
    )
    lines.append(r"\midrule")

    for bias_idx, (bias_type, bias_label) in enumerate(BIAS_ORDER):
        k_random = get_k_by_bias(args, "random", bias_type)
        k_explicit = get_k_by_bias(args, "explicit", bias_type)
        k_implicit = get_k_by_bias(args, "implicit", bias_type)

        for dataset_idx, (dataset, dataset_tex) in enumerate(DATASETS_ORDER):
            if model_view == "Average":
                by_z = gather_row_metrics_average(
                    persistent_dir=persistent_dir,
                    results_root=results_root,
                    dataset=dataset,
                    split=split,
                    bias_type=bias_type,
                    model_list=model_list,
                    debiasing=debiasing,
                    k_random=k_random,
                    k_explicit=k_explicit,
                    k_implicit=k_implicit,
                )
            else:
                by_z = gather_row_metrics_for_model(
                    persistent_dir=persistent_dir,
                    results_root=results_root,
                    dataset=dataset,
                    split=split,
                    bias_type=bias_type,
                    model_name_or_path=model_view,
                    debiasing=debiasing,
                    k_random=k_random,
                    k_explicit=k_explicit,
                    k_implicit=k_implicit,
                )

            cells = cells_from_by_z(by_z)
            if dataset_idx == 0:
                row = rf"\multirow{{4}}{{*}}{{{bias_label}}} & {dataset_tex} & " + " & ".join(cells) + r" \\"
            else:
                row = rf"& {dataset_tex} & " + " & ".join(cells) + r" \\"
            lines.append(row)
            lines.append(r"\addlinespace[4pt]")

        if bias_idx < len(BIAS_ORDER) - 1:
            lines.append(r"\midrule")
            lines.append("")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    model_display = "Average" if model_view == "Average" else short_model_name(model_view)
    lines.append(rf"\caption{{{make_caption(model_display, split, args)}}}")
    lines.append(rf"\label{{{make_label(model_display, split)}}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def add_bias_specific_k_args(ap: argparse.ArgumentParser) -> None:
    for variant in ["random", "explicit", "implicit"]:
        for bias_type, _ in BIAS_ORDER:
            ap.add_argument(
                f"--k_{variant}_{bias_type}",
                type=int,
                default=None,
                help=(
                    f"K for SAL {variant} on bias type '{bias_type}'. "
                    f"Default: {DEFAULT_KS[variant][bias_type]}."
                ),
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persistent_dir", type=str, required=True)
    ap.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased|meta-llama/Llama-3.1-8B|mistralai/Mistral-7B-v0.3",
        help='Models as "|" separated (recommended) or comma separated.',
    )
    ap.add_argument("--debiasing", type=str, default="SAL")
    ap.add_argument(
        "--split",
        type=str,
        default="joint|disjoint|natural_joint|natural_disjoint",
        help='e.g. "joint|disjoint|natural_joint|natural_disjoint"',
    )
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="results/summary_textual_erasure/STE_tables_summary",
    )
    add_bias_specific_k_args(ap)
    args = ap.parse_args()

    model_list = parse_pipe_or_comma_list(args.model_name_or_path)
    if not model_list:
        raise ValueError("No models parsed from --model_name_or_path")

    splits = parse_pipe_or_comma_list(args.split)
    if not splits:
        raise ValueError("No splits parsed from --split")

    table_views: List[str] = model_list + ["Average"]

    for split in splits:
        out_dir = os.path.join(args.out_dir, split)
        os.makedirs(out_dir, exist_ok=True)

        for model_view in table_views:
            tex = latex_table(
                persistent_dir=args.persistent_dir,
                results_root=args.results_root,
                model_list=model_list,
                model_view=model_view,
                debiasing=args.debiasing,
                split=split,
                args=args,
            )

            model_key = (
                "average"
                if model_view == "Average"
                else short_model_name(model_view).lower().replace(".", "").replace(" ", "-")
            )
            out_path = os.path.join(out_dir, f"STE_summary_{model_key}_{short_split_name(split)}.tex")
            with open(out_path, "w") as f:
                f.write(tex)
            print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()