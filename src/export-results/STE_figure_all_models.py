#!/usr/bin/env python3
"""
Plot TPR-gap vs number of singular vectors removed for STE experiments.

Expected layout:
{persistent_dir}/results/
  STE-{dataset}-{split}-{Z_attribute}/
    U_{k}/
      {model_name_or_path}/{debiasing}/{bias_type}/
        *.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # safe on clusters / headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


Z_ATTRS = ["textual", "binary", "random_textual"]
DATASETS = ["mathematics", "stackoverflow", "english", "superuser"]

# Same style for same Z_attr
Z_STYLE = {
    "textual": {"linestyle": "-",  "marker": "s"},   # dotted
    "binary": {"linestyle": "--",   "marker": "^"},   # solid
    "random_textual": {"linestyle": ":", "marker": "o" },  # dashed
}

Z_LABEL = {
    "textual": "Implicit",
    "binary": "Explicit",
    "random_textual": "Random",
}

# Explicit SAL should stop once it reaches the rank induced by the number of classes.
EXPLICIT_MAX_K_BY_BIAS = {
    "gender": 2,
    "reputation": 2,
    "engagement": 2,
    "location": 5,
    "name": 24,
}

# Custom numeric axis mapping: show true k values as tick labels, but spread out small k.
PREFERRED_K_ORDER = [0, 1, 2, 4, 5, 6, 24, 25, 45, 50]
CUSTOM_X_POSITIONS = {
    0: 0.0,
    1: 1.3,
    2: 2.5,
    4: 3.9,
    5: 4.6,
    6: 5.3,
    24: 7.5,
    25: 8.0,
    45: 9.5,
    50: 10.2,
}


MODEL_ALIAS = {
    "bert-base-uncased": "BERT",
    "meta-llama/Llama-3.1-8B": "Llama",
    "mistralai/Mistral-7B-v0.3": "Mistral",
}


def _to_percent(x: float) -> float:
    # Robust: if stored as fraction, convert to percent; if already percent, keep.
    return x * 100.0 if x <= 1.0 else x



def _load_json_list(path: str) -> List[dict]:
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError(f"Unexpected JSON type in {path}: {type(obj)}")



def _aggregate_tpr_gaps(experiments: List[dict]) -> Tuple[float, float, int]:
    """Return (mean_before_pct, mean_after_pct, n)."""
    befores, afters = [], []
    for exp in experiments:
        if "tpr_gap_before" not in exp or "tpr_gap_after" not in exp:
            continue
        befores.append(_to_percent(float(exp["tpr_gap_before"])))
        afters.append(_to_percent(float(exp["tpr_gap_after"])))

    if not befores or not afters:
        return float("nan"), float("nan"), 0

    mean_before = sum(befores) / len(befores)
    mean_after = sum(afters) / len(afters)
    return mean_before, mean_after, len(afters)



def _find_Uk_dirs(base_dir: str) -> List[Tuple[int, str]]:
    """Return list of (k, path) for directories named U_{k}."""
    out: List[Tuple[int, str]] = []
    if not os.path.isdir(base_dir):
        return out
    for name in os.listdir(base_dir):
        m = re.fullmatch(r"U_(\d+)", name)
        if not m:
            continue
        k = int(m.group(1))
        out.append((k, os.path.join(base_dir, name)))
    return sorted(out, key=lambda t: t[0])



def _short_model_name(model_name_or_path: str) -> str:
    if model_name_or_path in MODEL_ALIAS:
        return MODEL_ALIAS[model_name_or_path]
    return model_name_or_path.split("/")[-1] if "/" in model_name_or_path else model_name_or_path



def _map_k_to_custom_x(k: int) -> float:
    """
    Map true k values to a custom numeric axis that expands small k and compresses large k.
    Known values use hand-tuned positions. Unknown values fall back to monotonic positions.
    """
    if k in CUSTOM_X_POSITIONS:
        return CUSTOM_X_POSITIONS[k]

    # Fallback: preserve ordering and keep unknown ks between nearest known neighbors if possible.
    known_sorted = sorted(CUSTOM_X_POSITIONS)
    if k < known_sorted[0]:
        return CUSTOM_X_POSITIONS[known_sorted[0]] - float(known_sorted[0] - k)
    if k > known_sorted[-1]:
        return CUSTOM_X_POSITIONS[known_sorted[-1]] + 0.2 * float(k - known_sorted[-1])

    left = max(v for v in known_sorted if v < k)
    right = min(v for v in known_sorted if v > k)
    left_x = CUSTOM_X_POSITIONS[left]
    right_x = CUSTOM_X_POSITIONS[right]
    frac = (k - left) / float(right - left)
    return left_x + frac * (right_x - left_x)



def collect_curve(
    persistent_dir: str,
    dataset: str,
    z_attr: str,
    model_name_or_path: str,
    debiasing: str,
    split: str,
    bias_type: str,
    results_root: str = "results",
    max_k: Optional[int] = None,
) -> Tuple[List[int], List[float]]:
    """
    Returns x (true k values including 0) and y (tpr-gap values),
    where y[0] is baseline tpr_gap_before (k=0).
    """
    base = os.path.join(persistent_dir, results_root, f"STE-{dataset}-{split}-{z_attr}")
    uk_dirs = _find_Uk_dirs(base)
    if not uk_dirs:
        print(f"[WARN] No U_k dirs under: {base}")
        return [], []

    k_to_after: Dict[int, float] = {}
    baseline_before: Optional[float] = None

    for k, uk_path in uk_dirs:
        if max_k is not None and k > max_k:
            continue

        leaf = os.path.join(uk_path, model_name_or_path, debiasing, bias_type)
        json_paths = sorted(glob.glob(os.path.join(leaf, "*.json")))
        if not json_paths:
            continue

        exps: List[dict] = []
        for jp in json_paths:
            exps.extend(_load_json_list(jp))

        mean_before, mean_after, n = _aggregate_tpr_gaps(exps)
        if n == 0:
            continue

        if baseline_before is None and not (mean_before != mean_before):  # not NaN
            baseline_before = mean_before

        k_to_after[k] = mean_after

    if baseline_before is None or not k_to_after:
        return [], []

    ks = sorted(k_to_after.keys())
    xs = [0] + ks
    ys = [baseline_before] + [k_to_after[k] for k in ks]
    return xs, ys



def _sorted_tick_values(present_ks: List[int]) -> List[int]:
    present = set(present_ks)
    ordered = [k for k in PREFERRED_K_ORDER if k in present]
    extras = sorted(k for k in present if k not in PREFERRED_K_ORDER)
    return ordered + extras



def plot_dataset_all_models(
    persistent_dir: str,
    dataset: str,
    model_list: List[str],
    debiasing: str,
    bias_type: str,
    split: str,
    out_dir: str,
    results_root: str = "results",
) -> None:
    fig, ax = plt.subplots()

    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_cycle:
        default_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    model_to_color: Dict[str, str] = {}
    for i, m in enumerate(model_list):
        model_to_color[m] = default_cycle[i % len(default_cycle)]

    any_line = False
    all_true_ks: List[int] = []

    for model_name_or_path in model_list:
        color = model_to_color[model_name_or_path]
        model_label = _short_model_name(model_name_or_path)

        for z_attr in Z_ATTRS:
            max_k: Optional[int] = None
            if z_attr == "binary":
                max_k = EXPLICIT_MAX_K_BY_BIAS.get(bias_type)

            true_xs, ys = collect_curve(
                persistent_dir=persistent_dir,
                dataset=dataset,
                z_attr=z_attr,
                model_name_or_path=model_name_or_path,
                debiasing=debiasing,
                split=split,
                bias_type=bias_type,
                results_root=results_root,
                max_k=max_k,
            )
            if not true_xs:
                continue

            plot_xs = [_map_k_to_custom_x(x) for x in true_xs]
            all_true_ks.extend(true_xs)
            any_line = True
            style = Z_STYLE[z_attr]
            ax.plot(
                plot_xs,
                ys,
                label=f"{model_label} | {z_attr}",
                color=color,
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
                markersize=4,
            )

            pairs = ", ".join([f"{x}:{y:.2f}" for x, y in zip(true_xs, ys)])
            print(f"[{dataset}] {model_name_or_path} {z_attr} -> {pairs}")

    if not any_line:
        print(f"[WARN] Nothing to plot for dataset={dataset}.")
        plt.close(fig)
        return

    ax.set_xlabel("k")
    ax.set_ylabel("TPR-gap")
    ax.grid(True, alpha=0.3)

    tick_values = _sorted_tick_values(all_true_ks)
    tick_positions = [_map_k_to_custom_x(k) for k in tick_values]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(k) for k in tick_values])
    if tick_positions:
        ax.set_xlim(min(tick_positions) - 0.3, max(tick_positions) + 0.3)

    model_handles = [
        Line2D([0], [0], color=model_to_color[m], lw=3, label=_short_model_name(m))
        for m in model_list
    ]
    legend1 = ax.legend(handles=model_handles, title="Model (color)", loc="upper right", frameon=True)
    ax.add_artist(legend1)

    z_handles = [
        Line2D(
            [0], [0],
            color="black",
            lw=2,
            linestyle=Z_STYLE[z]["linestyle"],
            marker=Z_STYLE[z]["marker"],
            markersize=4,
            label=Z_LABEL.get(z, z),
        )
        for z in Z_ATTRS
    ]
    ax.legend(handles=z_handles, title="Bias", loc="lower left", frameon=True)

    out_dir = os.path.join(out_dir, split, bias_type)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"STE_tpr_gap_vs_U_{dataset}_s-{split}_i-{bias_type}_all_models.png")
    pdf_path = os.path.join(out_dir, f"STE_tpr_gap_vs_U_{dataset}_s-{split}_i-{bias_type}_all_models.pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent_dir", type=str, required=True)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help='Pipe-separated list, e.g. "bert-base-uncased|meta-llama/Llama-3.1-8B|mistralai/Mistral-7B-v0.3"',
    )
    parser.add_argument("--debiasing", type=str, default="SAL")
    parser.add_argument("--bias_type", type=str, default="reputation",
                    help='e.g. "reputation" or "reputation|location"')
    parser.add_argument("--split", type=str, default="disjoint",
                        help='e.g. "disjoint" or "joint|disjoint"')
    parser.add_argument("--out_dir", type=str, default="results/figures_textual_erasure")
    parser.add_argument("--results_root", type=str, default="results")
    args = parser.parse_args()

    bias_types = [x.strip() for x in args.bias_type.split("|") if x.strip()]
    splits = [x.strip() for x in args.split.split("|") if x.strip()]
    language_model_list = [m.strip() for m in args.model_name_or_path.split("|") if m.strip()]
    if not language_model_list:
        raise ValueError("No models parsed from --model_name_or_path")

    for split in splits:
        for bias_type in bias_types:
            for dataset in DATASETS:
                plot_dataset_all_models(
                    persistent_dir=args.persistent_dir,
                    dataset=dataset,
                    model_list=language_model_list,
                    debiasing=args.debiasing,
                    split=split,
                    bias_type=bias_type,
                    out_dir=args.out_dir,
                    results_root=args.results_root,
                )


if __name__ == "__main__":
    main()
