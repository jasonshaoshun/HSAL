import numpy as np
import random
from typing import Dict, Any
import argparse
from sklearn.linear_model import LogisticRegression
import json
import os
import sys

# Add current directory to path for imports
current_dir = os.getcwd()
sys.path.append(current_dir)
from src.util import generate_experiment_id


parser = argparse.ArgumentParser(description="Evaluate multilingual SEFair.")
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


def train_clf(x_train, y_train, x_test, y_test):
    """
    Trains a Logistic Regression classifier and returns train/test accuracy.
    """
    random.seed(0)
    np.random.seed(0)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=0)

    clf.fit(x_train, y_train)
    score_test = float(clf.score(x_test, y_test))
    score_train = float(clf.score(x_train, y_train))

    return clf, score_train, score_test


def get_tpr_stats(y_pred: np.ndarray, y_true: np.ndarray, z_true: np.ndarray) -> Dict[str, Any]:
    """
    Returns detailed per-class/group TPRs plus:
      - tpr_gap_rms: unweighted RMS of per-class gaps
      - weighted_tpr_gap_rms: class-support-weighted RMS of per-class gaps

    Per-class gap = 2 * std(TPRs across Z groups), which matches |A-B|
    in the binary-Z case.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    z_true = z_true.astype(int)

    unique_y = np.unique(y_true)
    unique_z = np.unique(z_true)

    per_class_gap = {}
    per_class_support = {}
    per_class_tprs = {}

    gaps = []
    weights = []

    for y_c in unique_y:
        tprs_for_this_class = []
        per_class_tprs[str(int(y_c))] = {}
        support_this_class = int((y_true == y_c).sum())
        per_class_support[str(int(y_c))] = support_this_class

        for g in unique_z:
            mask_true = (y_true == y_c) & (z_true == g)
            denom = int(mask_true.sum())

            if denom == 0:
                per_class_tprs[str(int(y_c))][str(int(g))] = {
                    "support": 0,
                    "tpr": None,
                }
                continue

            tpr = float((y_pred[mask_true] == y_c).mean())
            per_class_tprs[str(int(y_c))][str(int(g))] = {
                "support": denom,
                "tpr": tpr,
            }
            tprs_for_this_class.append(tpr)

        if len(tprs_for_this_class) > 0:
            gap = float(2 * np.std(tprs_for_this_class))
        else:
            gap = 0.0

        per_class_gap[str(int(y_c))] = gap
        gaps.append(gap)
        weights.append(support_this_class)

    if len(gaps) == 0:
        tpr_gap_rms = 0.0
        weighted_tpr_gap_rms = 0.0
    else:
        gaps_arr = np.asarray(gaps, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr = weights_arr / weights_arr.sum()
        tpr_gap_rms = float(np.sqrt(np.mean(gaps_arr ** 2)))
        weighted_tpr_gap_rms = float(np.sqrt(np.sum(weights_arr * (gaps_arr ** 2))))

    return {
        "per_class_tprs": per_class_tprs,
        "per_class_gap": per_class_gap,
        "per_class_support": per_class_support,
        "tpr_gap_rms": tpr_gap_rms,
        "weighted_tpr_gap_rms": weighted_tpr_gap_rms,
    }


def print_tpr_stats(label: str, stats: Dict[str, Any]) -> None:
    print(f"=== {label} TPR details ===")
    for y_c in sorted(stats["per_class_tprs"].keys(), key=int):
        print(
            f"Y={y_c} | support={stats['per_class_support'][y_c]} | "
            f"gap={stats['per_class_gap'][y_c]:.8f}"
        )
        for z_g in sorted(stats["per_class_tprs"][y_c].keys(), key=int):
            item = stats["per_class_tprs"][y_c][z_g]
            tpr_str = "None" if item["tpr"] is None else f"{item['tpr']:.8f}"
            print(f"  Z={z_g}: support={item['support']}, tpr={tpr_str}")
    print(
        f"{label} -> tpr_gap_rms={stats['tpr_gap_rms']:.8f}, "
        f"weighted_tpr_gap_rms={stats['weighted_tpr_gap_rms']:.8f}"
    )


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name=args.experiment,
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        debiasing_method=args.debiasing,
        bias_type=args.bias_type
    )

    print(f"Running Eval on: {args.npz_path}")
    data = np.load(args.npz_path)

    if args.X_attribute == "input":
        x_train = data['x_train_text_cls']
        x_test = data['x_test_text_cls']
    elif args.X_attribute == "self-description":
        x_train = data['x_train_aboutme_cls']
        x_test = data['x_test_aboutme_cls']
    else:
        raise ValueError(f"Unknown X_attribute: {args.X_attribute}")

    y_train_help = data['y_train_helpfulness'].astype(int)
    y_test_help = data['y_test_helpfulness'].astype(int)

    if args.bias_type == "location":
        print("Loading Z as LOCATION")
        z_train = data['y_train_location'].astype(int)
        z_test = data['y_test_location'].astype(int)
    elif args.bias_type == "engagement":
        print("Loading Z as ENGAGEMENT")
        z_train = data['y_train_engagement'].astype(int)
        z_test = data['y_test_engagement'].astype(int)
    elif args.bias_type == "reputation":
        print("Loading Z as REPUTATION")
        z_train = data['y_train_reputation'].astype(int)
        z_test = data['y_test_reputation'].astype(int)
    elif args.bias_type == "name":
        print("Loading Z as NAME")
        z_train = data['y_train_name_cluster'].astype(int)
        z_test = data['y_test_name_cluster'].astype(int)
    elif args.bias_type == "gender":
        print("Loading Z as GENDER")
        z_train = data['y_train_gender'].astype(int)
        z_test = data['y_test_gender'].astype(int)
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

    if args.projection_matrix and os.path.exists(args.projection_matrix):
        print(f"Loading Projection: {args.projection_matrix}")
        P = np.load(args.projection_matrix)['proj_accumulated']

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
    }]

    os.makedirs(f"{args.results_save_path}", exist_ok=True)
    with open(f"{args.results_save_path}/{experiment_id}.json", "w") as f:
        json.dump(result, f, indent=4)
