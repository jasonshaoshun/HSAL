import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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


DEFAULT_THRESHOLDS = {
    "english": {"engagement": 4, "reputation": 92133},
    "mathematics": {"engagement": 10872, "reputation": 127912},
    "stackoverflow": {"engagement": 6321, "reputation": 22483},
    "superuser": {"engagement": 4436, "reputation": 6763},
}


parser = argparse.ArgumentParser(
    description=(
        "Predict z from x_aboutme_cls using raw.npz only: load raw arrays with allow_pickle=True, "
        "deduplicate globally by x_userid, split unique users into train/dev/test, then tune logistic regression on dev."
    )
)
parser.add_argument("--persistent_dir", type=str, default="")
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--split", type=str, default="raw_unique")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--bias_type", type=str, required=True, choices=["reputation", "location", "engagement"])
parser.add_argument("--npz_path", type=str, required=True)
parser.add_argument("--results_save_path", type=str, required=True)
parser.add_argument("--dev_ratio", type=float, default=0.2)
parser.add_argument("--test_ratio", type=float, default=0.3)
parser.add_argument("--random_state", type=int, default=0)
parser.add_argument("--reputation_threshold", type=int, default=None)
parser.add_argument("--engagement_threshold", type=int, default=None)


def resolve_thresholds(dataset: str, reputation_threshold: int | None, engagement_threshold: int | None) -> Tuple[int, int]:
    dataset_key = str(dataset).strip().lower()
    if dataset_key in DEFAULT_THRESHOLDS:
        defaults = DEFAULT_THRESHOLDS[dataset_key]
        rep = int(defaults["reputation"] if reputation_threshold is None else reputation_threshold)
        eng = int(defaults["engagement"] if engagement_threshold is None else engagement_threshold)
        return rep, eng

    if reputation_threshold is None or engagement_threshold is None:
        raise ValueError(
            f"No default thresholds known for dataset={dataset!r}. "
            f"Please pass both --reputation_threshold and --engagement_threshold explicitly."
        )

    return int(reputation_threshold), int(engagement_threshold)


def _require_keys(data: np.lib.npyio.NpzFile, required_keys: List[str]) -> None:
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(
            f"raw.npz is missing required keys: {missing}. Available keys: {list(data.keys())}"
        )


def _normalize_user_ids(arr: Any) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(str)


def _normalize_features(arr: Any) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected x_aboutme_cls to be 2D (n,d) or 1D, got shape {arr.shape}")
    try:
        return arr.astype(np.float32)
    except (TypeError, ValueError):
        return np.vstack([np.asarray(row, dtype=np.float32) for row in arr])


def _normalize_location_labels(arr: Any) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(int)

    unique_vals, inverse = np.unique(arr.astype(str), return_inverse=True)
    print(
        "Location labels are not integer typed in raw.npz; remapping to consecutive ints: "
        + json.dumps({str(v): int(i) for i, v in enumerate(unique_vals.tolist())}, ensure_ascii=False)
    )
    return inverse.astype(int)


def load_raw_arrays(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    required_keys = [
        "x_aboutme_cls",
        "x_userid",
        "y_reputation",
        "y_location",
        "y_engagement",
    ]
    _require_keys(data, required_keys)

    x = _normalize_features(data["x_aboutme_cls"])
    user_ids = _normalize_user_ids(data["x_userid"])
    y_reputation_raw = np.asarray(data["y_reputation"]).reshape(-1)
    y_location_raw = _normalize_location_labels(data["y_location"])
    y_engagement_raw = np.asarray(data["y_engagement"]).reshape(-1)

    lengths = {
        "x_aboutme_cls": int(len(x)),
        "x_userid": int(len(user_ids)),
        "y_reputation": int(len(y_reputation_raw)),
        "y_location": int(len(y_location_raw)),
        "y_engagement": int(len(y_engagement_raw)),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Length mismatch across raw arrays: {lengths}")

    return {
        "x": x,
        "userid": user_ids,
        "reputation_raw": y_reputation_raw,
        "location_raw": y_location_raw,
        "engagement_raw": y_engagement_raw,
        "keys": {
            "x": "x_aboutme_cls",
            "userid": "x_userid",
            "reputation": "y_reputation",
            "location": "y_location",
            "engagement": "y_engagement",
        },
    }


def make_labels(arrays: Dict[str, Any], bias_type: str, reputation_threshold: int, engagement_threshold: int) -> np.ndarray:
    if bias_type == "reputation":
        raw = np.asarray(arrays["reputation_raw"]).reshape(-1)
        return (raw >= reputation_threshold).astype(int)
    if bias_type == "engagement":
        raw = np.asarray(arrays["engagement_raw"]).reshape(-1)
        return (raw >= engagement_threshold).astype(int)
    if bias_type == "location":
        return np.asarray(arrays["location_raw"]).reshape(-1).astype(int)
    raise ValueError(f"Unsupported bias_type: {bias_type}")


def counts_by_label(y: np.ndarray) -> Dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def deduplicate_keep_first_global(
    x: np.ndarray,
    y: np.ndarray,
    user_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seen = set()
    keep: List[int] = []
    for idx, uid in enumerate(user_ids.tolist()):
        if uid not in seen:
            seen.add(uid)
            keep.append(idx)
    keep_idx = np.asarray(keep, dtype=int)
    return x[keep_idx], y[keep_idx], user_ids[keep_idx], keep_idx


def _stratify_or_none(y: np.ndarray) -> Tuple[Any, bool]:
    values, counts = np.unique(y, return_counts=True)
    can_stratify = len(values) >= 2 and np.all(counts >= 2)
    return (y if can_stratify else None), can_stratify


def split_unique_users(
    x: np.ndarray,
    y: np.ndarray,
    user_ids: np.ndarray,
    test_ratio: float,
    dev_ratio: float,
    random_state: int,
) -> Dict[str, Any]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0,1), got {test_ratio}")
    if not (0.0 < dev_ratio < 1.0):
        raise ValueError(f"dev_ratio must be in (0,1), got {dev_ratio}")
    if test_ratio + dev_ratio >= 1.0:
        raise ValueError(f"Need test_ratio + dev_ratio < 1.0, got {test_ratio + dev_ratio}")

    stratify_all, used_stratify_all = _stratify_or_none(y)
    x_train_dev, x_test, y_train_dev, y_test, uid_train_dev, uid_test = train_test_split(
        x,
        y,
        user_ids,
        test_size=test_ratio,
        random_state=random_state,
        stratify=stratify_all,
    )

    dev_ratio_within_train_dev = dev_ratio / (1.0 - test_ratio)
    stratify_train_dev, used_stratify_train_dev = _stratify_or_none(y_train_dev)
    x_train, x_dev, y_train, y_dev, uid_train, uid_dev = train_test_split(
        x_train_dev,
        y_train_dev,
        uid_train_dev,
        test_size=dev_ratio_within_train_dev,
        random_state=random_state,
        stratify=stratify_train_dev,
    )

    return {
        "x_train": x_train,
        "x_dev": x_dev,
        "x_test": x_test,
        "y_train": y_train,
        "y_dev": y_dev,
        "y_test": y_test,
        "uid_train": uid_train,
        "uid_dev": uid_dev,
        "uid_test": uid_test,
        "used_stratify_unique_to_test": used_stratify_all,
        "used_stratify_train_dev": used_stratify_train_dev,
        "dev_ratio_within_train_dev": dev_ratio_within_train_dev,
    }


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
                multi_class="auto",
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


def prediction_summary(model: Pipeline, x: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    y_pred = model.predict(x)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "predicted_label_counts": counts_by_label(y_pred),
        "confusion_matrix": {
            "labels": [int(v) for v in labels.tolist()],
            "matrix": cm.astype(int).tolist(),
            "rows": "true",
            "cols": "pred",
        },
    }


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
    best_payload: Dict[str, Any] | None = None

    for params in get_logreg_grid():
        clf = build_logreg_pipeline(random_state=random_state, c_value=float(params["C"]))
        clf.fit(x_train, y_train)

        train_summary = prediction_summary(clf, x_train, y_train)
        dev_summary = prediction_summary(clf, x_dev, y_dev)
        test_summary = prediction_summary(clf, x_test, y_test)

        result = {
            "classifier": "logistic_regression",
            "params": params,
            "train_accuracy": train_summary["accuracy"],
            "dev_accuracy": dev_summary["accuracy"],
            "test_accuracy": test_summary["accuracy"],
            "train_prediction_summary": train_summary,
            "dev_prediction_summary": dev_summary,
            "test_prediction_summary": test_summary,
        }
        tried.append(result)

        if best_payload is None or result["dev_accuracy"] > best_payload["best_result"]["dev_accuracy"]:
            best_payload = {
                "best_result": result,
            }

    assert best_payload is not None
    return best_payload["best_result"], tried


def print_label_distribution(prefix: str, y: np.ndarray) -> None:
    print(f"{prefix} label distribution: {json.dumps(counts_by_label(y), sort_keys=True)}")


def print_prediction_block(tag: str, summary: Dict[str, Any]) -> None:
    print(f"[logreg] predicted label counts ({tag}): {json.dumps(summary['predicted_label_counts'], sort_keys=True)}")
    cm = summary["confusion_matrix"]
    print(
        f"[logreg] confusion matrix ({tag}) [rows=true, cols=pred, labels={cm['labels']}]: {json.dumps(cm['matrix'])}"
    )


if __name__ == "__main__":
    args = parser.parse_args()

    reputation_threshold, engagement_threshold = resolve_thresholds(
        dataset=args.dataset,
        reputation_threshold=args.reputation_threshold,
        engagement_threshold=args.engagement_threshold,
    )

    experiment_id = generate_experiment_id(
        name=args.experiment,
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        debiasing_method="predict_z_from_x_logreg_raw_unique",
        bias_type=args.bias_type,
    )

    print(f"Loading raw NPZ with allow_pickle=True: {args.npz_path}")
    print(
        f"Using thresholds: dataset={args.dataset}, reputation_threshold={reputation_threshold}, "
        f"engagement_threshold={engagement_threshold}"
    )
    data = np.load(args.npz_path, allow_pickle=True)
    arrays = load_raw_arrays(data)

    x_raw = arrays["x"]
    uid_raw = arrays["userid"]
    y_raw = make_labels(
        arrays=arrays,
        bias_type=args.bias_type,
        reputation_threshold=reputation_threshold,
        engagement_threshold=engagement_threshold,
    )

    print(f"Raw samples before deduplication: {len(x_raw)}")
    print_label_distribution("Raw before deduplication", y_raw)

    x_unique, y_unique, uid_unique, keep_idx = deduplicate_keep_first_global(x_raw, y_raw, uid_raw)

    print(f"Unique-user samples after deduplication: {len(x_unique)}")
    print(f"Removed duplicated-user samples: {len(x_raw) - len(x_unique)}")
    print_label_distribution("Unique-user after deduplication", y_unique)

    split = split_unique_users(
        x=x_unique,
        y=y_unique,
        user_ids=uid_unique,
        test_ratio=args.test_ratio,
        dev_ratio=args.dev_ratio,
        random_state=args.random_state,
    )

    x_train = split["x_train"]
    x_dev = split["x_dev"]
    x_test = split["x_test"]
    y_train = split["y_train"]
    y_dev = split["y_dev"]
    y_test = split["y_test"]
    uid_train = split["uid_train"]
    uid_dev = split["uid_dev"]
    uid_test = split["uid_test"]

    overlap_train_dev = int(len(set(uid_train.tolist()) & set(uid_dev.tolist())))
    overlap_train_test = int(len(set(uid_train.tolist()) & set(uid_test.tolist())))
    overlap_dev_test = int(len(set(uid_dev.tolist()) & set(uid_test.tolist())))
    if overlap_train_dev or overlap_train_test or overlap_dev_test:
        raise RuntimeError(
            f"Unexpected split overlap: train/dev={overlap_train_dev}, train/test={overlap_train_test}, dev/test={overlap_dev_test}"
        )

    print(f"Final split sizes: train={len(x_train)}, dev={len(x_dev)}, test={len(x_test)}")
    print_label_distribution("Train split", y_train)
    print_label_distribution("Dev split", y_dev)
    print_label_distribution("Test split", y_test)

    best_result, tried_results = evaluate_logreg_grid(
        x_train=x_train,
        y_train=y_train,
        x_dev=x_dev,
        y_dev=y_dev,
        x_test=x_test,
        y_test=y_test,
        random_state=args.random_state,
    )

    print("Best logistic-regression setting:")
    print(json.dumps({
        "classifier": best_result["classifier"],
        "params": best_result["params"],
        "train_accuracy": best_result["train_accuracy"],
        "dev_accuracy": best_result["dev_accuracy"],
        "test_accuracy": best_result["test_accuracy"],
    }, indent=2))
    print_prediction_block("train", best_result["train_prediction_summary"])
    print_prediction_block("dev", best_result["dev_prediction_summary"])
    print_prediction_block("test", best_result["test_prediction_summary"])

    result = {
        "experiment_id": experiment_id,
        "experiment": args.experiment,
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "model_name_or_path": args.model_name_or_path,
        "bias_type": args.bias_type,
        "npz_path": args.npz_path,
        "feature_key": arrays["keys"]["x"],
        "user_id_key": arrays["keys"]["userid"],
        "raw_label_keys": {
            "reputation": arrays["keys"]["reputation"],
            "location": arrays["keys"]["location"],
            "engagement": arrays["keys"]["engagement"],
        },
        "label_construction": {
            "bias_type": args.bias_type,
            "dataset": args.dataset,
            "reputation_threshold": int(reputation_threshold),
            "engagement_threshold": int(engagement_threshold),
            "location_note": "location labels taken directly from y_location in raw.npz",
        },
        "deduplication": {
            "strategy": "keep_first_by_userid_globally_from_raw_pool",
            "raw_n": int(len(x_raw)),
            "unique_user_n": int(len(x_unique)),
            "removed_duplicate_rows_n": int(len(x_raw) - len(x_unique)),
            "kept_indices_preview": keep_idx[:10].astype(int).tolist(),
        },
        "split_sizes": {
            "n_train": int(len(x_train)),
            "n_dev": int(len(x_dev)),
            "n_test": int(len(x_test)),
            "test_ratio": float(args.test_ratio),
            "dev_ratio": float(args.dev_ratio),
            "used_stratify_unique_to_test": bool(split["used_stratify_unique_to_test"]),
            "used_stratify_train_dev": bool(split["used_stratify_train_dev"]),
        },
        "label_distributions": {
            "raw_before_deduplication": counts_by_label(y_raw),
            "unique_after_deduplication": counts_by_label(y_unique),
            "train": counts_by_label(y_train),
            "dev": counts_by_label(y_dev),
            "test": counts_by_label(y_test),
        },
        "user_id_overlap": {
            "train_dev": overlap_train_dev,
            "train_test": overlap_train_test,
            "dev_test": overlap_dev_test,
        },
        "selection_metric": "maximize_validation_accuracy",
        "best_result": best_result,
        "all_validation_runs": tried_results,
    }

    os.makedirs(args.results_save_path, exist_ok=True)
    output_path = os.path.join(args.results_save_path, f"{experiment_id}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved result to: {output_path}")
