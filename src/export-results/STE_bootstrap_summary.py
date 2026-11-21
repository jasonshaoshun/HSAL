#!/usr/bin/env python3
"""
Generate dataset-specific STE bootstrap LaTeX tables.

Each table fixes one dataset and reports the Average view used by STE_summary.py
(mean across BERT, Llama3.1-8B, and Mistral-7B). Rows follow the same bias and
method ordering as STE_summary.py. Main and TPR-Gap display the original full
test-set result; SD and 95% CI come from the post-level bootstrap distribution.

Expected result layout:
{persistent_dir}/results_bootstrap/
  STE-{dataset}-{split}-{z_attr}/
    U_{k}/
      {model_name_or_path}/{debiasing}/{bias_type}/
        *.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional


DATASETS_ORDER = [
    ("stackoverflow", "Stack Overflow", r"\textcolor{black}{\logoso \textbf{Stackoverflow}}"),
    ("mathematics", "Mathematics", r"\textcolor{black}{\logoms \textbf{Mathematics}}"),
    ("superuser", "Super User", r"\textcolor{black}{\logosu \textbf{SuperUser}}"),
    ("english", "English", r"\textcolor{black}{\logoes \textbf{English}}"),
]

BIAS_ORDER = [
    ("name", "DisplayName"),
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

METHOD_SPECS = [
    ("Baseline", None),
    ("SAL (random)", "random_textual"),
    ("SAL", "binary"),
    ("H-SAL", "textual"),
]

STAT_FIELDS = ["original", "mean", "sd", "ci_lower", "ci_upper"]
DELTA_THRESHOLD = 0.0005


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


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def average_stat_dicts(stats: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not stats:
        return None
    out: Dict[str, float] = {}
    for field in STAT_FIELDS:
        vals = [float(s[field]) for s in stats if field in s and is_number(s[field])]
        if not vals:
            return None
        out[field] = sum(vals) / len(vals)
    return out


def bootstrap_stat(exp: dict, phase: str, metric_name: str) -> Optional[Dict[str, float]]:
    bootstrap = exp.get("bootstrap")
    if not isinstance(bootstrap, dict):
        return None
    phase_dict = bootstrap.get(phase)
    if not isinstance(phase_dict, dict):
        return None
    metric = phase_dict.get(metric_name)
    if not isinstance(metric, dict):
        return None
    if not all(field in metric and is_number(metric[field]) for field in STAT_FIELDS):
        return None
    return {field: float(metric[field]) for field in STAT_FIELDS}


def mean_metrics(exps: List[dict]) -> Optional[Dict[str, Dict[str, float]]]:
    if not exps:
        return None

    main_before = []
    main_after = []
    tpr_before = []
    tpr_after = []
    for exp in exps:
        mb = bootstrap_stat(exp, "before", "main_task_accuracy")
        ma = bootstrap_stat(exp, "after", "main_task_accuracy")
        tb = bootstrap_stat(exp, "before", "tpr_gap")
        ta = bootstrap_stat(exp, "after", "tpr_gap")
        if mb is None or ma is None or tb is None or ta is None:
            continue
        main_before.append(mb)
        main_after.append(ma)
        tpr_before.append(tb)
        tpr_after.append(ta)

    mb_avg = average_stat_dicts(main_before)
    ma_avg = average_stat_dicts(main_after)
    tb_avg = average_stat_dicts(tpr_before)
    ta_avg = average_stat_dicts(tpr_after)
    if mb_avg is None or ma_avg is None or tb_avg is None or ta_avg is None:
        return None

    return {
        "main_before": mb_avg,
        "main_after": ma_avg,
        "tpr_before": tb_avg,
        "tpr_after": ta_avg,
    }


def average_metric_dicts(metric_dicts: List[Optional[Dict[str, Dict[str, float]]]]) -> Optional[Dict[str, Dict[str, float]]]:
    valid = [m for m in metric_dicts if m is not None]
    if not valid:
        return None

    out: Dict[str, Dict[str, float]] = {}
    for metric_key in ["main_before", "main_after", "tpr_before", "tpr_after"]:
        stat_avg = average_stat_dicts([m[metric_key] for m in valid])
        if stat_avg is None:
            return None
        out[metric_key] = stat_avg
    return out


def format_value(value: float, precision: int) -> str:
    if not is_number(value):
        return ""
    return f"{float(value):.{precision}f}"


def format_ci(stat: Dict[str, float], precision: int) -> str:
    lower = format_value(stat["ci_lower"], precision)
    upper = format_value(stat["ci_upper"], precision)
    if not lower or not upper:
        return ""
    return f"[{lower}, {upper}]"


def pretty_main(after: float, before: float, precision: int, include_comparisons: bool) -> str:
    value = format_value(after, precision)
    if not value or not include_comparisons:
        return value

    delta = after - before
    if abs(delta) < DELTA_THRESHOLD:
        return value
    if delta > 0:
        return rf"\uag{{{abs(delta):.{precision}f}}} {value}"
    return rf"\dab{{{abs(delta):.{precision}f}}} {value}"


def pretty_tpr(after: float, before: float, precision: int, include_comparisons: bool) -> str:
    value = format_value(after, precision)
    if not value or not include_comparisons:
        return value

    delta = after - before
    if abs(delta) < DELTA_THRESHOLD:
        return value
    if delta > 0:
        return rf"\ua{{{abs(delta):.{precision}f}}} {value}"
    return rf"\da{{{abs(delta):.{precision}f}}} {value}"


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


def load_metrics_for_setting(leaf_dir: str) -> Optional[Dict[str, Dict[str, float]]]:
    json_paths = sorted(glob.glob(os.path.join(leaf_dir, "*.json")))
    if not json_paths:
        return None
    exps: List[dict] = []
    for path in json_paths:
        exps.extend(load_json_as_list(path))
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
) -> Optional[Dict[str, Dict[str, float]]]:
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
) -> Dict[str, Optional[Dict[str, Dict[str, float]]]]:
    out: Dict[str, Optional[Dict[str, Dict[str, float]]]] = {}
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


def baseline_from_by_z(
    by_z: Dict[str, Optional[Dict[str, Dict[str, float]]]]
) -> Optional[Dict[str, Dict[str, float]]]:
    for key in ["random_textual", "binary", "textual"]:
        if by_z.get(key) is not None:
            return by_z[key]
    return None


def blank_cells(n: int) -> List[str]:
    return [""] * n


def cells_for_method(
    by_z: Dict[str, Optional[Dict[str, Dict[str, float]]]],
    method_key: Optional[str],
    precision: int,
    include_comparisons: bool,
) -> List[str]:
    baseline_src = baseline_from_by_z(by_z)
    if baseline_src is None:
        return blank_cells(6)

    base_main = baseline_src["main_before"]
    base_tpr = baseline_src["tpr_before"]

    if method_key is None:
        return [
            format_value(base_main["original"], precision),
            format_value(base_main["sd"], precision),
            format_ci(base_main, precision),
            format_value(base_tpr["original"], precision),
            format_value(base_tpr["sd"], precision),
            format_ci(base_tpr, precision),
        ]

    method_metrics = by_z.get(method_key)
    if method_metrics is None:
        return blank_cells(6)

    main = method_metrics["main_after"]
    tpr = method_metrics["tpr_after"]
    return [
        pretty_main(main["original"], base_main["original"], precision, include_comparisons),
        format_value(main["sd"], precision),
        format_ci(main, precision),
        pretty_tpr(tpr["original"], base_tpr["original"], precision, include_comparisons),
        format_value(tpr["sd"], precision),
        format_ci(tpr, precision),
    ]


def dataset_by_key(dataset_key: str) -> tuple[str, str, str]:
    for item in DATASETS_ORDER:
        if item[0] == dataset_key:
            return item
    raise ValueError(f"Unknown dataset: {dataset_key}")


def make_caption(
    dataset_plain: str,
    split: str,
    args: argparse.Namespace,
    include_comparisons: bool,
) -> str:
    comparison_text = (
        r"Comparison commands indicate changes relative to the baseline row for the same bias type."
        if include_comparisons
        else r"All entries are plain numeric values without comparison commands."
    )
    body = [
        rf"\textbf{{Average bootstrap summary for {dataset_plain} across BERT, Llama3.1-8B, and Mistral-7B.}}",
        r"Main = helpfulness accuracy; TPR-Gap = generalized disparity ($2 \times$ StdDev) of true positive rates across groups.",
        r"Main and TPR-Gap are original full-test-set results; SD and 95\% CI are from the post-level bootstrap distribution.",
        comparison_text,
        rf"SAL uses bias-specific choices of singular directions: random [{format_k_summary(args, 'random')}], explicit [{format_k_summary(args, 'explicit')}], and implicit [{format_k_summary(args, 'implicit')}].",
        SPLIT_DESC.get(split, ""),
    ]
    return "\n".join([x for x in body if x])


def make_label(dataset: str, split: str, include_comparisons: bool) -> str:
    suffix = "paper" if include_comparisons else "plain"
    return f"tab:ste-bootstrap-summary-{suffix}-{dataset}-{short_split_name(split)}"


def latex_table_for_dataset(
    persistent_dir: str,
    results_root: str,
    model_list: List[str],
    dataset: str,
    debiasing: str,
    split: str,
    args: argparse.Namespace,
    include_comparisons: bool = True,
) -> str:
    _dataset_key, dataset_plain, dataset_tex = dataset_by_key(dataset)
    dataset_title = dataset_tex if include_comparisons else rf"\textbf{{{dataset_plain}}}"

    lines: List[str] = []
    lines.append(r"\begin{table*}[th!]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{ll c c c c c c}")
    lines.append(r"\toprule")
    lines.append(rf"\multicolumn{{8}}{{c}}{{{dataset_title}}} \\")
    lines.append(r"\midrule")
    lines.append(r"Bias Type & Method & Main & SD & 95\% CI & TPR-Gap & SD & 95\% CI \\")
    lines.append(r"\midrule")

    for bias_idx, (bias_type, bias_label) in enumerate(BIAS_ORDER):
        k_random = get_k_by_bias(args, "random", bias_type)
        k_explicit = get_k_by_bias(args, "explicit", bias_type)
        k_implicit = get_k_by_bias(args, "implicit", bias_type)

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

        for method_idx, (method_label, method_key) in enumerate(METHOD_SPECS):
            row_cells = cells_for_method(
                by_z=by_z,
                method_key=method_key,
                precision=args.precision,
                include_comparisons=include_comparisons,
            )

            if method_idx == 0:
                row = rf"\multirow{{4}}{{*}}{{{bias_label}}} & {method_label} & " + " & ".join(row_cells) + r" \\"
            else:
                row = rf"& {method_label} & " + " & ".join(row_cells) + r" \\"
            lines.append(row)
            lines.append(r"\addlinespace[4pt]")

        if bias_idx < len(BIAS_ORDER) - 1:
            lines.append(r"\midrule")
            lines.append("")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(rf"\caption{{{make_caption(dataset_plain, split, args, include_comparisons)}}}")
    lines.append(rf"\label{{{make_label(dataset, split, include_comparisons)}}}")
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


def build_arg_parser() -> argparse.ArgumentParser:
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
        default="natural_disjoint",
        help='e.g. "natural_disjoint" or "joint|disjoint|natural_joint|natural_disjoint"',
    )
    ap.add_argument("--results_root", type=str, default="results_bootstrap")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="results_bootstrap/summary_textual_erasure/STE_tables_bootstrap_summary",
    )
    ap.add_argument("--precision", type=int, default=3)
    add_bias_specific_k_args(ap)
    return ap


def run_main(include_comparisons: bool = True, default_out_dir: Optional[str] = None) -> None:
    ap = build_arg_parser()
    if default_out_dir is not None:
        ap.set_defaults(out_dir=default_out_dir)
    args = ap.parse_args()

    model_list = parse_pipe_or_comma_list(args.model_name_or_path)
    if not model_list:
        raise ValueError("No models parsed from --model_name_or_path")

    splits = parse_pipe_or_comma_list(args.split)
    if not splits:
        raise ValueError("No splits parsed from --split")

    for split in splits:
        out_dir = os.path.join(args.out_dir, split)
        os.makedirs(out_dir, exist_ok=True)

        for dataset, _dataset_plain, _dataset_tex in DATASETS_ORDER:
            tex = latex_table_for_dataset(
                persistent_dir=args.persistent_dir,
                results_root=args.results_root,
                model_list=model_list,
                dataset=dataset,
                debiasing=args.debiasing,
                split=split,
                args=args,
                include_comparisons=include_comparisons,
            )
            out_path = os.path.join(
                out_dir,
                f"STE_bootstrap_summary_{dataset}_{short_split_name(split)}.tex",
            )
            with open(out_path, "w") as f:
                f.write(tex)
            print(f"[OK] wrote {out_path}")


def main() -> None:
    run_main(include_comparisons=True)


if __name__ == "__main__":
    main()
