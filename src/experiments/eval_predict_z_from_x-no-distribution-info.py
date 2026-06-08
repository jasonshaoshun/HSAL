import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

current_dir = os.getcwd()
sys.path.append(current_dir)

try:
    from src.util import generate_experiment_id
except Exception:
    def generate_experiment_id(name, model, model_name_or_path, debiasing_method, bias_type):
        safe_model_name = str(model_name_or_path).replace('/', '-')
        return f"{name}-{model}-{safe_model_name}-{debiasing_method}-{bias_type}"


parser = argparse.ArgumentParser(
    description="Predict z from x with user deduplication and dev-set model selection over logistic regression settings."
)
parser.add_argument("--persistent_dir", type=str, default="")
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--split", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--bias_type", type=str, required=True, choices=["reputation", "location", "engagement", "name"])
parser.add_argument("--npz_path", type=str, required=True)
parser.add_argument("--results_save_path", type=str, required=True)
parser.add_argument("--dev_ratio", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=0)


def _find_first_existing(data: np.lib.npyio.NpzFile, candidates: List[str]) -> str:
    for key in candidates:
        if key in data:
            return key
    raise KeyError(f"None of these keys found in npz: {candidates}. Available keys: {list(data.keys())}")


def load_arrays(data: np.lib.npyio.NpzFile, bias_type: str) -> Dict[str, Any]:
    x_train_key = _find_first_existing(data, ["x_train_aboutme_cls", "x_train_cls", "x_train"])
    x_test_key = _find_first_existing(data, ["x_test_aboutme_cls", "x_test_cls", "x_test"])
    uid_train_key = _find_first_existing(data, ["x_train_userid", "userid_train", "user_id_train", "train_userid"])
    uid_test_key = _find_first_existing(data, ["x_test_userid", "userid_test", "user_id_test", "test_userid"])
    z_train_key = _find_first_existing(data, [f"y_train_{bias_type}", f"z_train_{bias_type}", "z_train"])
    z_test_key = _find_first_existing(data, [f"y_test_{bias_type}", f"z_test_{bias_type}", "z_test"])

    return {
        "x_train": np.asarray(data[x_train_key]),
        "x_test": np.asarray(data[x_test_key]),
        "uid_train": np.asarray(data[uid_train_key]),
        "uid_test": np.asarray(data[uid_test_key]),
        "z_train": np.asarray(data[z_train_key]).astype(int),
        "z_test": np.asarray(data[z_test_key]).astype(int),
        "keys": {
            "x_train": x_train_key,
            "x_test": x_test_key,
            "uid_train": uid_train_key,
            "uid_test": uid_test_key,
            "z_train": z_train_key,
            "z_test": z_test_key,
        },
    }


def deduplicate_keep_first(x: np.ndarray, y: np.ndarray, user_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seen = set()
    keep: List[int] = []
    for idx, uid in enumerate(user_ids.tolist()):
        if uid not in seen:
            seen.add(uid)
            keep.append(idx)
    keep_idx = np.asarray(keep, dtype=int)
    return x[keep_idx], y[keep_idx], user_ids[keep_idx], keep_idx


def stratified_train_dev_split(
    x: np.ndarray,
    y: np.ndarray,
    user_ids: np.ndarray,
    dev_ratio: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    unique_classes, counts = np.unique(y, return_counts=True)
    can_stratify = len(unique_classes) >= 2 and np.all(counts >= 2)
    stratify = y if can_stratify else None
    x_tr, x_dev, y_tr, y_dev, uid_tr, uid_dev = train_test_split(
        x,
        y,
        user_ids,
        test_size=dev_ratio,
        random_state=random_state,
        stratify=stratify,
    )
    return x_tr, x_dev, y_tr, y_dev, uid_tr, uid_dev, can_stratify


def build_logreg_pipeline(random_state: int, c_value: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                C=c_value,
                max_iter=4000,
                solver="lbfgs",
                random_state=random_state,
            ),
        ),
    ])


def get_logreg_grid() -> List[Dict[str, Any]]:
    return [
        {"C": 0.01},
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0},
        {"C": 100.0},
    ]


def counts_by_label(y: np.ndarray) -> Dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def evaluate_logreg_grid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    tried: List[Dict[str, Any]] = []
    best_result: Dict[str, Any] | None = None

    for params in get_logreg_grid():
        clf = build_logreg_pipeline(random_state=random_state, c_value=float(params["C"]))
        clf.fit(x_train, y_train)

        result = {
            "classifier": "logistic_regression",
            "params": params,
            "train_accuracy": float(clf.score(x_train, y_train)),
            "dev_accuracy": float(clf.score(x_dev, y_dev)),
            "test_accuracy": float(clf.score(x_test, y_test)),
        }
        tried.append(result)

        if best_result is None or result["dev_accuracy"] > best_result["dev_accuracy"]:
            best_result = result

    assert best_result is not None
    return best_result, tried


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name=args.experiment,
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        debiasing_method="predict_z_from_x_logreg",
        bias_type=args.bias_type,
    )

    print(f"Loading NPZ: {args.npz_path}")
    data = np.load(args.npz_path, allow_pickle=True)
    arrays = load_arrays(data, args.bias_type)

    x_train_raw = arrays["x_train"]
    x_test_raw = arrays["x_test"]
    z_train_raw = arrays["z_train"]
    z_test_raw = arrays["z_test"]
    uid_train_raw = arrays["uid_train"]
    uid_test_raw = arrays["uid_test"]

    x_train, z_train, uid_train, keep_train = deduplicate_keep_first(x_train_raw, z_train_raw, uid_train_raw)
    x_test, z_test, uid_test, keep_test = deduplicate_keep_first(x_test_raw, z_test_raw, uid_test_raw)

    overlap_after_dedup = sorted(set(uid_train.tolist()) & set(uid_test.tolist()))
    if overlap_after_dedup:
        print(f"[WARN] Found {len(overlap_after_dedup)} overlapping user ids after deduplication.")

    x_tr, x_dev, y_tr, y_dev, uid_tr, uid_dev, used_stratify = stratified_train_dev_split(
        x_train,
        z_train,
        uid_train,
        dev_ratio=args.dev_ratio,
        random_state=args.random_state,
    )

    best_result, tried_results = evaluate_logreg_grid(
        x_train=x_tr,
        y_train=y_tr,
        x_dev=x_dev,
        y_dev=y_dev,
        x_test=x_test,
        y_test=z_test,
        random_state=args.random_state,
    )

    result = {
        "experiment_id": experiment_id,
        "experiment": args.experiment,
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "model_name_or_path": args.model_name_or_path,
        "bias_type": args.bias_type,
        "npz_path": args.npz_path,
        "feature_key": arrays["keys"]["x_train"],
        "user_id_keys": {
            "train": arrays["keys"]["uid_train"],
            "test": arrays["keys"]["uid_test"],
        },
        "label_keys": {
            "train": arrays["keys"]["z_train"],
            "test": arrays["keys"]["z_test"],
        },
        "deduplication": {
            "strategy": "keep_first_by_userid_within_each_split",
            "train_raw_n": int(len(x_train_raw)),
            "test_raw_n": int(len(x_test_raw)),
            "train_dedup_n": int(len(x_train)),
            "test_dedup_n": int(len(x_test)),
            "train_removed_n": int(len(x_train_raw) - len(x_train)),
            "test_removed_n": int(len(x_test_raw) - len(x_test)),
            "train_overlap_with_test_after_dedup_n": int(len(overlap_after_dedup)),
        },
        "split_sizes": {
            "n_train": int(len(x_tr)),
            "n_dev": int(len(x_dev)),
            "n_test": int(len(x_test)),
            "used_stratify": bool(used_stratify),
        },
        "label_distributions": {
            "train": counts_by_label(y_tr),
            "dev": counts_by_label(y_dev),
            "test": counts_by_label(z_test),
        },
        "selection_metric": "maximize_validation_accuracy",
        "best_result": best_result,
        "all_validation_runs": tried_results,
    }

    os.makedirs(args.results_save_path, exist_ok=True)
    output_path = os.path.join(args.results_save_path, f"{experiment_id}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Best logistic-regression setting:")
    print(json.dumps(best_result, indent=2))
    print(f"Saved result to: {output_path}")
