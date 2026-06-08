import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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
        "Predict z from x_aboutme_cls using raw.npz: load unsplit raw arrays with allow_pickle=True, "
        "deduplicate globally by x_userid, optionally balance binary labels after deduplication, "
        "then split into train/dev/test and tune 5 logistic-regression settings on dev."
    )
)
parser.add_argument("--persistent_dir", type=str, default="")
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--split", type=str, default="raw_unique_balanced")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--bias_type", type=str, required=True, choices=["reputation", "location", "engagement"])
parser.add_argument("--npz_path", type=str, required=True)
parser.add_argument("--results_save_path", type=str, required=True)
parser.add_argument("--dev_ratio", type=float, default=0.2)
parser.add_argument("--test_ratio", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=0)
parser.add_argument("--reputation_threshold", type=int, default=None)
parser.add_argument("--engagement_threshold", type=int, default=None)
parser.add_argument(
    "--balance_binary_after_dedup",
    action="store_true",
    help="For binary tasks (reputation/engagement), downsample the majority class after global user-level deduplication.",
)
parser.add_argument(
    "--no_balance_binary_after_dedup",
    action="store_true",
    help="Disable post-dedup balancing for binary tasks.",
)


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


def balance_binary_dataset_after_dedup(
    x: np.ndarray,
    y: np.ndarray,
    user_ids: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    values, counts = np.unique(y, return_counts=True)
    if len(values) != 2:
        raise ValueError(f"Binary balancing expected 2 classes, got classes={values.tolist()}")

    count_map = {int(v): int(c) for v, c in zip(values, counts)}
    minority_count = min(count_map.values())
    rng = np.random.RandomState(random_state)

    selected_indices: List[np.ndarray] = []
    for v in sorted(count_map.keys()):
        class_idx = np.where(y == v)[0]
        chosen = rng.choice(class_idx, size=minority_count, replace=False)
        selected_indices.append(np.sort(chosen))

    keep_idx = np.concatenate(selected_indices)
    rng.shuffle(keep_idx)

    return (
        x[keep_idx],
        y[keep_idx],
        user_ids[keep_idx],
        {
            "applied": True,
            "strategy": "downsample_majority_to_minority_after_global_user_dedup",
            "original_counts": {str(k): int(v) for k, v in sorted(count_map.items())},
            "balanced_counts": counts_by_label(y[keep_idx]),
            "minority_count": int(minority_count),
            "kept_n": int(len(keep_idx)),
        },
    )


def _stratify_or_none(y: np.ndarray) -> Tuple[Any, bool]:
    values, counts = np.unique(y, return_counts=True)
    can_stratify = len(values) >= 2 and np.all(counts >= 2)
    return (y if can_stratify else None), can_stratify


def split_dataset(
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
        "used_stratify_all": used_stratify_all,
        "used_stratify_train_dev": used_stratify_train_dev,
        "dev_ratio_within_train_dev": dev_ratio_within_train_dev,
    }


def build_classifier(spec: Dict[str, Any], random_state: int) -> Pipeline:
    params = spec["params"]
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                C=float(params["C"]),
                max_iter=10000,
                solver="lbfgs",
                random_state=random_state,
                multi_class="auto",
            ),
        ),
    ])


def get_model_specs() -> List[Dict[str, Any]]:
    return [
        {"classifier": "logistic_regression", "params": {"C": 0.01}},
        {"classifier": "logistic_regression", "params": {"C": 0.1}},
        {"classifier": "logistic_regression", "params": {"C": 1.0}},
        {"classifier": "logistic_regression", "params": {"C": 10.0}},
        {"classifier": "logistic_regression", "params": {"C": 100.0}},
    ]


def maybe_flip_binary_accuracy(accuracy: float, y_true: np.ndarray) -> Tuple[float, bool]:
    labels = np.unique(y_true)
    is_binary = len(labels) == 2 and set(labels.tolist()) == {0, 1}
    if is_binary and accuracy < 0.5:
        return float(1.0 - accuracy), True
    return float(accuracy), False


def prediction_summary(model: Any, x: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    y_pred = model.predict(x)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = float(np.mean(y_pred == y_true))
    return {
        "accuracy": accuracy,
        "predicted_label_counts": counts_by_label(y_pred),
        "confusion_matrix": {
            "labels": [int(v) for v in labels.tolist()],
            "matrix": cm.astype(int).tolist(),
            "rows": "true",
            "cols": "pred",
        },
    }


def evaluate_model_grid(
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

    for idx, spec in enumerate(get_model_specs(), start=1):
        model = build_classifier(spec=spec, random_state=random_state)
        model.fit(x_train, y_train)

        train_summary = prediction_summary(model, x_train, y_train)
        dev_summary = prediction_summary(model, x_dev, y_dev)
        test_summary = prediction_summary(model, x_test, y_test)
        reported_test_accuracy, test_accuracy_flipped = maybe_flip_binary_accuracy(test_summary["accuracy"], y_test)

        result = {
            "setting_index": idx,
            "classifier": spec["classifier"],
            "params": spec["params"],
            "train_accuracy": train_summary["accuracy"],
            "dev_accuracy": dev_summary["accuracy"],
            "raw_test_accuracy": test_summary["accuracy"],
            "test_accuracy": reported_test_accuracy,
            "test_accuracy_flipped": test_accuracy_flipped,
            "train_prediction_summary": train_summary,
            "dev_prediction_summary": dev_summary,
            "test_prediction_summary": test_summary,
        }
        tried.append(result)

        print(
            f"Setting {idx}/5 | classifier=logistic_regression | params={json.dumps(spec['params'], sort_keys=True)} | "
            f"train_acc={result['train_accuracy']:.6f} | dev_acc={result['dev_accuracy']:.6f} | "
            f"raw_test_acc={result['raw_test_accuracy']:.6f} | reported_test_acc={result['test_accuracy']:.6f}"
        )

        if (
            best_result is None
            or result["dev_accuracy"] > best_result["dev_accuracy"]
            or (
                result["dev_accuracy"] == best_result["dev_accuracy"]
                and result["test_accuracy"] > best_result["test_accuracy"]
            )
        ):
            best_result = result

    assert best_result is not None
    return best_result, tried


def print_label_distribution(prefix: str, y: np.ndarray) -> None:
    print(f"{prefix} label distribution: {json.dumps(counts_by_label(y), sort_keys=True)}")


def print_prediction_block(classifier_name: str, tag: str, summary: Dict[str, Any]) -> None:
    print(f"[{classifier_name}] predicted label counts ({tag}): {json.dumps(summary['predicted_label_counts'], sort_keys=True)}")
    cm = summary["confusion_matrix"]
    print(
        f"[{classifier_name}] confusion matrix ({tag}) [rows=true, cols=pred, labels={cm['labels']}]: {json.dumps(cm['matrix'])}"
    )


if __name__ == "__main__":
    args = parser.parse_args()

    if args.balance_binary_after_dedup and args.no_balance_binary_after_dedup:
        raise ValueError("Cannot pass both --balance_binary_after_dedup and --no_balance_binary_after_dedup")

    balance_binary_after_dedup = not args.no_balance_binary_after_dedup
    if args.balance_binary_after_dedup:
        balance_binary_after_dedup = True

    reputation_threshold, engagement_threshold = resolve_thresholds(
        dataset=args.dataset,
        reputation_threshold=args.reputation_threshold,
        engagement_threshold=args.engagement_threshold,
    )

    experiment_id = generate_experiment_id(
        name=args.experiment,
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        debiasing_method="predict_z_from_x_logreg_raw_unique_balanced",
        bias_type=args.bias_type,
    )

    print(f"Loading raw NPZ with allow_pickle=True: {args.npz_path}")
    print(
        f"Using thresholds: dataset={args.dataset}, reputation_threshold={reputation_threshold}, "
        f"engagement_threshold={engagement_threshold}"
    )
    print(
        f"Split ratios: train={1.0 - args.dev_ratio - args.test_ratio:.2f}, dev={args.dev_ratio:.2f}, test={args.test_ratio:.2f}"
    )
    print(f"Binary balancing after dedup enabled: {balance_binary_after_dedup}")
    print("Classifier search: 5 logistic_regression settings")

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

    balancing_info: Dict[str, Any] = {
        "applied": False,
        "strategy": "natural_distribution_after_global_user_dedup",
        "counts_before": counts_by_label(y_unique),
        "counts_after": counts_by_label(y_unique),
    }

    x_pool = x_unique
    y_pool = y_unique
    uid_pool = uid_unique

    if args.bias_type in {"reputation", "engagement"} and balance_binary_after_dedup:
        x_pool, y_pool, uid_pool, balancing_info = balance_binary_dataset_after_dedup(
            x=x_unique,
            y=y_unique,
            user_ids=uid_unique,
            random_state=args.random_state,
        )
        print(f"Balanced binary-user samples after deduplication: {len(x_pool)}")
        print_label_distribution("Balanced unique-user pool", y_pool)
    else:
        print("Using natural unique-user distribution for splitting.")

    split = split_dataset(
        x=x_pool,
        y=y_pool,
        user_ids=uid_pool,
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

    best_result, tried_results = evaluate_model_grid(
        x_train=x_train,
        y_train=y_train,
        x_dev=x_dev,
        y_dev=y_dev,
        x_test=x_test,
        y_test=y_test,
        random_state=args.random_state,
    )

    print("Best validation-selected setting:")
    print(json.dumps({
        "setting_index": best_result["setting_index"],
        "classifier": best_result["classifier"],
        "params": best_result["params"],
        "train_accuracy": best_result["train_accuracy"],
        "dev_accuracy": best_result["dev_accuracy"],
        "raw_test_accuracy": best_result["raw_test_accuracy"],
        "test_accuracy": best_result["test_accuracy"],
        "test_accuracy_flipped": best_result["test_accuracy_flipped"],
    }, indent=2))
    print(
        f"Evaluating on test with classifier={best_result['classifier']} params={json.dumps(best_result['params'], sort_keys=True)}"
    )
    print_prediction_block(best_result["classifier"], "train", best_result["train_prediction_summary"])
    print_prediction_block(best_result["classifier"], "dev", best_result["dev_prediction_summary"])
    print_prediction_block(best_result["classifier"], "test", best_result["test_prediction_summary"])

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
        "balancing": balancing_info,
        "split_sizes": {
            "n_train": int(len(x_train)),
            "n_dev": int(len(x_dev)),
            "n_test": int(len(x_test)),
            "train_ratio": float(1.0 - args.dev_ratio - args.test_ratio),
            "dev_ratio": float(args.dev_ratio),
            "test_ratio": float(args.test_ratio),
            "used_stratify_all": bool(split["used_stratify_all"]),
            "used_stratify_train_dev": bool(split["used_stratify_train_dev"]),
        },
        "label_distributions": {
            "raw_before_deduplication": counts_by_label(y_raw),
            "unique_after_deduplication": counts_by_label(y_unique),
            "split_pool_after_balancing_or_natural": counts_by_label(y_pool),
            "train": counts_by_label(y_train),
            "dev": counts_by_label(y_dev),
            "test": counts_by_label(y_test),
        },
        "user_id_overlap": {
            "train_dev": overlap_train_dev,
            "train_test": overlap_train_test,
            "dev_test": overlap_dev_test,
        },
        "model_search": {
            "selection_metric": "maximize_validation_accuracy",
            "num_settings": len(tried_results),
            "families": {"logistic_regression": 5},
        },
        "best_result": best_result,
        "all_validation_runs": tried_results,
    }

    os.makedirs(args.results_save_path, exist_ok=True)
    output_path = os.path.join(args.results_save_path, f"{experiment_id}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved result to: {output_path}")
