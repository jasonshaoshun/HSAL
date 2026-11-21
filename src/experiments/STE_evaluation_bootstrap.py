import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add current directory to path for imports, matching STE_evaluation.py.
current_dir = os.getcwd()
sys.path.append(current_dir)

from src.experiments.STE_evaluation import get_tpr_stats, print_tpr_stats, train_clf
from src.util import generate_experiment_id


parser = argparse.ArgumentParser(description="Evaluate multilingual SEFair with post-level bootstrap CIs.")
parser.add_argument("--persistent_dir", action="store", type=str)
parser.add_argument("--experiment", action="store", type=str)
parser.add_argument("--model", action="store", type=str)
parser.add_argument("--model_name_or_path", action="store", type=str, default="bert-base-multilingual-cased")
parser.add_argument("--X_attribute", action="store", choices=["input", "self-description"], type=str, default="input")
parser.add_argument("--debiasing", action="store", type=str)
parser.add_argument("--projection_matrix", action="store", type=str)
parser.add_argument("--bias_type", action="store", type=str)
parser.add_argument("--npz_path", action="store", type=str)
parser.add_argument("--results_save_path", action="store", type=str)
parser.add_argument("--num_eigenvectors_to_remove", action="store", type=int, default=1)
parser.add_argument("--num_eigenvectors_to_remove_from_global_U", action="store", type=int)
parser.add_argument("--bootstrap_samples", action="store", type=int, default=200)
parser.add_argument("--bootstrap_seed", action="store", type=int, default=0)
parser.add_argument("--bootstrap_max_attempts", action="store", type=int, default=100000)


def generate_valid_bootstrap_indices(
    y_true: np.ndarray,
    z_true: np.ndarray,
    num_samples: int,
    seed: int,
    max_attempts: int,
) -> Tuple[List[np.ndarray], int]:
    """Sample post indices until every replicate preserves original (Y, Z) cells."""
    if num_samples < 2:
        raise ValueError("--bootstrap_samples must be at least 2 when using ddof=1.")
    if max_attempts < num_samples:
        raise ValueError("--bootstrap_max_attempts must be at least --bootstrap_samples.")

    n_posts = int(len(y_true))
    if n_posts == 0:
        raise ValueError("Cannot bootstrap an empty test set.")

    original_pairs = set(zip(y_true.astype(int).tolist(), z_true.astype(int).tolist()))
    rng = np.random.default_rng(seed)

    samples: List[np.ndarray] = []
    attempts = 0
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        indices = rng.integers(0, n_posts, size=n_posts)
        sampled_pairs = set(zip(y_true[indices].astype(int).tolist(), z_true[indices].astype(int).tolist()))
        if original_pairs.issubset(sampled_pairs):
            samples.append(indices)

    if len(samples) != num_samples:
        raise RuntimeError(
            "Unable to collect "
            f"{num_samples} valid bootstrap samples after {max_attempts} attempts. "
            f"Collected {len(samples)}. Original (Y, Z) cells: {sorted(original_pairs)}."
        )

    return samples, attempts


def summarize_bootstrap(original: float, bootstrap_values: List[float]) -> Dict[str, float]:
    values = np.asarray(bootstrap_values, dtype=float)
    ci_lower, ci_upper = np.percentile(values, [2.5, 97.5])
    return {
        "original": float(original),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def summarize_missing(original: float) -> Dict[str, float]:
    return {
        "original": float(original),
        "mean": float(np.nan),
        "sd": float(np.nan),
        "ci_lower": float(np.nan),
        "ci_upper": float(np.nan),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    z_true: np.ndarray,
    y_pred_before: np.ndarray,
    y_pred_after: Optional[np.ndarray],
    original_accuracy_before: float,
    original_tpr_gap_before: float,
    original_accuracy_after: float,
    original_tpr_gap_after: float,
    num_samples: int,
    seed: int,
    max_attempts: int,
) -> Dict[str, Any]:
    bootstrap_indices, attempts = generate_valid_bootstrap_indices(
        y_true=y_true,
        z_true=z_true,
        num_samples=num_samples,
        seed=seed,
        max_attempts=max_attempts,
    )

    before_accuracy_values: List[float] = []
    before_tpr_gap_values: List[float] = []
    after_accuracy_values: List[float] = []
    after_tpr_gap_values: List[float] = []

    for indices in bootstrap_indices:
        y_sample = y_true[indices]
        z_sample = z_true[indices]

        before_pred_sample = y_pred_before[indices]
        before_accuracy_values.append(float((before_pred_sample == y_sample).mean()))
        before_tpr_gap_values.append(float(get_tpr_stats(before_pred_sample, y_sample, z_sample)["tpr_gap_rms"]))

        if y_pred_after is not None:
            after_pred_sample = y_pred_after[indices]
            after_accuracy_values.append(float((after_pred_sample == y_sample).mean()))
            after_tpr_gap_values.append(float(get_tpr_stats(after_pred_sample, y_sample, z_sample)["tpr_gap_rms"]))

    if y_pred_after is None:
        after_accuracy = summarize_missing(original_accuracy_after)
        after_tpr_gap = summarize_missing(original_tpr_gap_after)
    else:
        after_accuracy = summarize_bootstrap(original_accuracy_after, after_accuracy_values)
        after_tpr_gap = summarize_bootstrap(original_tpr_gap_after, after_tpr_gap_values)

    return {
        "num_samples": int(num_samples),
        "seed": int(seed),
        "max_attempts": int(max_attempts),
        "attempts": int(attempts),
        "before": {
            "main_task_accuracy": summarize_bootstrap(original_accuracy_before, before_accuracy_values),
            "tpr_gap": summarize_bootstrap(original_tpr_gap_before, before_tpr_gap_values),
        },
        "after": {
            "main_task_accuracy": after_accuracy,
            "tpr_gap": after_tpr_gap,
        },
    }


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name=args.experiment,
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        debiasing_method=args.debiasing,
        bias_type=args.bias_type,
    )

    print(f"Running Bootstrap Eval on: {args.npz_path}")
    data = np.load(args.npz_path)

    if args.X_attribute == "input":
        x_train = data["x_train_text_cls"]
        x_test = data["x_test_text_cls"]
    elif args.X_attribute == "self-description":
        x_train = data["x_train_aboutme_cls"]
        x_test = data["x_test_aboutme_cls"]
    else:
        raise ValueError(f"Unknown X_attribute: {args.X_attribute}")

    y_train_help = data["y_train_helpfulness"].astype(int)
    y_test_help = data["y_test_helpfulness"].astype(int)

    if args.bias_type == "location":
        print("Loading Z as LOCATION")
        z_train = data["y_train_location"].astype(int)
        z_test = data["y_test_location"].astype(int)
    elif args.bias_type == "engagement":
        print("Loading Z as ENGAGEMENT")
        z_train = data["y_train_engagement"].astype(int)
        z_test = data["y_test_engagement"].astype(int)
    elif args.bias_type == "reputation":
        print("Loading Z as REPUTATION")
        z_train = data["y_train_reputation"].astype(int)
        z_test = data["y_test_reputation"].astype(int)
    elif args.bias_type == "name":
        print("Loading Z as NAME")
        z_train = data["y_train_name_cluster"].astype(int)
        z_test = data["y_test_name_cluster"].astype(int)
    elif args.bias_type == "gender":
        print("Loading Z as GENDER")
        z_train = data["y_train_gender"].astype(int)
        z_test = data["y_test_gender"].astype(int)
    else:
        raise ValueError(f"Unknown bias_type: {args.bias_type}")

    # ========================== BIASED EVALUATION ==========================
    clf_help, _, help_acc = train_clf(x_train, y_train_help, x_test, y_test_help)
    y_pred_help = clf_help.predict(x_test)

    tpr_stats_before = get_tpr_stats(y_pred_help, y_test_help, z_test)
    tpr_gap_before = tpr_stats_before["tpr_gap_rms"]
    weighted_tpr_gap_before = tpr_stats_before["weighted_tpr_gap_rms"]

    _, _, z_acc = train_clf(x_train, z_train, x_test, z_test)

    print_tpr_stats("BIASED", tpr_stats_before)
    print(
        f"BIASED -> HelpAcc: {help_acc:.8f}, BiasPredAcc: {z_acc:.8f}, "
        f"Gap(RMS): {tpr_gap_before:.8f}, WeightedGap(RMS): {weighted_tpr_gap_before:.8f}"
    )

    # ========================== DEBIASED EVALUATION ==========================
    help_acc_after = np.nan
    z_acc_after = np.nan
    tpr_gap_after = np.nan
    weighted_tpr_gap_after = np.nan
    tpr_stats_after = None
    y_pred_help_af = None

    if args.projection_matrix and os.path.exists(args.projection_matrix):
        print(f"Loading Projection: {args.projection_matrix}")
        P = np.load(args.projection_matrix)["proj_accumulated"]

        x_train_deb = P.dot(x_train.T).T
        x_test_deb = P.dot(x_test.T).T

        clf_help_af, _, help_acc_after = train_clf(x_train_deb, y_train_help, x_test_deb, y_test_help)
        y_pred_help_af = clf_help_af.predict(x_test_deb)

        tpr_stats_after = get_tpr_stats(y_pred_help_af, y_test_help, z_test)
        tpr_gap_after = tpr_stats_after["tpr_gap_rms"]
        weighted_tpr_gap_after = tpr_stats_after["weighted_tpr_gap_rms"]

        _, _, z_acc_after = train_clf(x_train_deb, z_train, x_test_deb, z_test)

        print_tpr_stats("DEBIASED", tpr_stats_after)
        print(
            f"DEBIASED -> HelpAcc: {help_acc_after:.8f}, BiasPredAcc: {z_acc_after:.8f}, "
            f"Gap(RMS): {tpr_gap_after:.8f}, WeightedGap(RMS): {weighted_tpr_gap_after:.8f}"
        )

    bootstrap = bootstrap_metrics(
        y_true=y_test_help,
        z_true=z_test,
        y_pred_before=y_pred_help,
        y_pred_after=y_pred_help_af,
        original_accuracy_before=help_acc,
        original_tpr_gap_before=tpr_gap_before,
        original_accuracy_after=help_acc_after,
        original_tpr_gap_after=tpr_gap_after,
        num_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        max_attempts=args.bootstrap_max_attempts,
    )

    result = [{
        "experiment_id": experiment_id,
        "experiment": args.experiment,
        "debiasing": args.debiasing,
        "model": args.model,
        "main_task_accuracy_before": help_acc,
        "main_task_accuracy_after": help_acc_after,
        "tpr_gap_before": tpr_gap_before,
        "tpr_gap_after": tpr_gap_after,
        "weighted_tpr_gap_before": weighted_tpr_gap_before,
        "weighted_tpr_gap_after": weighted_tpr_gap_after,
        "bias_prediction_accuracy_before": z_acc,
        "bias_prediction_accuracy_after": z_acc_after,
        "bias_type": args.bias_type,
        "num_eigenvectors_removed": args.num_eigenvectors_to_remove,
        "tpr_stats_before": tpr_stats_before,
        "tpr_stats_after": tpr_stats_after,
        "bootstrap": bootstrap,
    }]

    os.makedirs(f"{args.results_save_path}", exist_ok=True)
    with open(f"{args.results_save_path}/{experiment_id}.json", "w") as f:
        json.dump(result, f, indent=4)
