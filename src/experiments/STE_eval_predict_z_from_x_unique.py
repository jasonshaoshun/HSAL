import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def generate_experiment_id(
    name: str,
    model: str,
    model_name_or_path: str,
    debiasing_method: str,
    bias_type: str,
) -> str:
    safe_model_name = str(model_name_or_path).replace("/", "-")
    return f"{name}-{model}-{safe_model_name}-{debiasing_method}-{bias_type}"


parser = argparse.ArgumentParser(
    description=(
        "Predict z from x_train/val/test_aboutme_cls using the existing train/val/test splits "
        "stored in the NPZ file. Deduplicate to unique users within each split before training."
    )
)
parser.add_argument("--persistent_dir", type=str, default="")
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument(
    "--bias_type",
    type=str,
    required=True,
    choices=["reputation", "location", "engagement", "gender", "name"],
)
parser.add_argument("--npz_path", type=str, required=True)
parser.add_argument("--results_save_path", type=str, required=True)


def _require_keys(data: np.lib.npyio.NpzFile, required_keys: List[str]) -> None:
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(
            f"NPZ file is missing required keys: {missing}. Available keys: {list(data.keys())}"
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
        raise ValueError(f"Expected feature array to be 2D (n,d) or 1D, got shape {arr.shape}")
    try:
        return arr.astype(np.float32)
    except (TypeError, ValueError):
        return np.vstack([np.asarray(row, dtype=np.float32) for row in arr])


def _normalize_integer_like_labels(arr: Any, label_name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(int)

    unique_vals, inverse = np.unique(arr.astype(str), return_inverse=True)
    print(
        f"{label_name} labels are not integer typed in npz; remapping to consecutive ints: "
        + json.dumps({str(v): int(i) for i, v in enumerate(unique_vals.tolist())}, ensure_ascii=False)
    )
    return inverse.astype(int)


def _check_same_length(prefix: str, x: np.ndarray, y: np.ndarray, uid: np.ndarray) -> None:
    lengths = {
        f"x_{prefix}_aboutme_cls": int(len(x)),
        f"y_{prefix}": int(len(y)),
        f"x_{prefix}_userid": int(len(uid)),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Length mismatch in {prefix} split: {lengths}")


def counts_by_label(y: np.ndarray) -> Dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def deduplicate_keep_first_within_split(
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


def load_split_arrays(data: np.lib.npyio.NpzFile, bias_type: str) -> Dict[str, Any]:
    required_keys = [
        "x_train_aboutme_cls",
        "x_val_aboutme_cls",
        "x_test_aboutme_cls",
        "x_train_userid",
        "x_val_userid",
        "x_test_userid",
    ]
    _require_keys(data, required_keys)

    x_train_raw = _normalize_features(data["x_train_aboutme_cls"])
    x_val_raw = _normalize_features(data["x_val_aboutme_cls"])
    x_test_raw = _normalize_features(data["x_test_aboutme_cls"])

    uid_train_raw = _normalize_user_ids(data["x_train_userid"])
    uid_val_raw = _normalize_user_ids(data["x_val_userid"])
    uid_test_raw = _normalize_user_ids(data["x_test_userid"])

    if bias_type == "reputation":
        y_train_raw = _normalize_integer_like_labels(data["y_train_reputation"], "reputation")
        y_val_raw = _normalize_integer_like_labels(data["y_val_reputation"], "reputation")
        y_test_raw = _normalize_integer_like_labels(data["y_test_reputation"], "reputation")
        label_keys = {
            "train": "y_train_reputation",
            "val": "y_val_reputation",
            "test": "y_test_reputation",
        }
    elif bias_type == "engagement":
        y_train_raw = _normalize_integer_like_labels(data["y_train_engagement"], "engagement")
        y_val_raw = _normalize_integer_like_labels(data["y_val_engagement"], "engagement")
        y_test_raw = _normalize_integer_like_labels(data["y_test_engagement"], "engagement")
        label_keys = {
            "train": "y_train_engagement",
            "val": "y_val_engagement",
            "test": "y_test_engagement",
        }
    elif bias_type == "gender":
        y_train_raw = _normalize_integer_like_labels(data["y_train_gender"], "gender")
        y_val_raw = _normalize_integer_like_labels(data["y_val_gender"], "gender")
        y_test_raw = _normalize_integer_like_labels(data["y_test_gender"], "gender")
        label_keys = {
            "train": "y_train_gender",
            "val": "y_val_gender",
            "test": "y_test_gender",
        }
    elif bias_type == "location":
        y_train_raw = _normalize_integer_like_labels(data["y_train_location"], "location")
        y_val_raw = _normalize_integer_like_labels(data["y_val_location"], "location")
        y_test_raw = _normalize_integer_like_labels(data["y_test_location"], "location")
        label_keys = {
            "train": "y_train_location",
            "val": "y_val_location",
            "test": "y_test_location",
        }
    elif bias_type == "name":
        y_train_raw = _normalize_integer_like_labels(data["y_train_name_cluster"], "name")
        y_val_raw = _normalize_integer_like_labels(data["y_val_name_cluster"], "name")
        y_test_raw = _normalize_integer_like_labels(data["y_test_name_cluster"], "name")
        label_keys = {
            "train": "y_train_name_cluster",
            "val": "y_val_name_cluster",
            "test": "y_test_name_cluster",
        }
    else:
        raise ValueError(f"Unsupported bias_type: {bias_type}")

    _check_same_length("train_raw", x_train_raw, y_train_raw, uid_train_raw)
    _check_same_length("val_raw", x_val_raw, y_val_raw, uid_val_raw)
    _check_same_length("test_raw", x_test_raw, y_test_raw, uid_test_raw)

    x_train, y_train, uid_train, keep_train = deduplicate_keep_first_within_split(
        x_train_raw, y_train_raw, uid_train_raw
    )
    x_val, y_val, uid_val, keep_val = deduplicate_keep_first_within_split(
        x_val_raw, y_val_raw, uid_val_raw
    )
    x_test, y_test, uid_test, keep_test = deduplicate_keep_first_within_split(
        x_test_raw, y_test_raw, uid_test_raw
    )

    return {
        "x_train_raw": x_train_raw,
        "x_val_raw": x_val_raw,
        "x_test_raw": x_test_raw,
        "y_train_raw": y_train_raw,
        "y_val_raw": y_val_raw,
        "y_test_raw": y_test_raw,
        "uid_train_raw": uid_train_raw,
        "uid_val_raw": uid_val_raw,
        "uid_test_raw": uid_test_raw,
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "uid_train": uid_train,
        "uid_val": uid_val,
        "uid_test": uid_test,
        "keep_train": keep_train,
        "keep_val": keep_val,
        "keep_test": keep_test,
        "keys": {
            "x_train": "x_train_aboutme_cls",
            "x_val": "x_val_aboutme_cls",
            "x_test": "x_test_aboutme_cls",
            "uid_train": "x_train_userid",
            "uid_val": "x_val_userid",
            "uid_test": "x_test_userid",
            "label_train": label_keys["train"],
            "label_val": label_keys["val"],
            "label_test": label_keys["test"],
        },
    }


def build_logreg_pipeline(c_value: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                C=c_value,
                max_iter=4000,
                solver="lbfgs",
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
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    tried: List[Dict[str, Any]] = []
    best_payload: Dict[str, Any] | None = None

    for params in get_logreg_grid():
        clf = build_logreg_pipeline(c_value=float(params["C"]))
        clf.fit(x_train, y_train)

        train_summary = prediction_summary(clf, x_train, y_train)
        val_summary = prediction_summary(clf, x_val, y_val)
        test_summary = prediction_summary(clf, x_test, y_test)

        result = {
            "classifier": "logistic_regression",
            "params": params,
            "train_accuracy": train_summary["accuracy"],
            "val_accuracy": val_summary["accuracy"],
            "test_accuracy": test_summary["accuracy"],
            "train_prediction_summary": train_summary,
            "val_prediction_summary": val_summary,
            "test_prediction_summary": test_summary,
        }
        tried.append(result)

        if best_payload is None or result["val_accuracy"] > best_payload["best_result"]["val_accuracy"]:
            best_payload = {"best_result": result}

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

    experiment_id = generate_experiment_id(
        name=args.experiment,
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        debiasing_method="predict_z_from_x_logreg_existing_split_unique_users",
        bias_type=args.bias_type,
    )

    print(f"Loading NPZ with allow_pickle=True: {args.npz_path}")
    data = np.load(args.npz_path, allow_pickle=True)
    arrays = load_split_arrays(data, bias_type=args.bias_type)

    x_train = arrays["x_train"]
    x_val = arrays["x_val"]
    x_test = arrays["x_test"]

    y_train = arrays["y_train"]
    y_val = arrays["y_val"]
    y_test = arrays["y_test"]

    uid_train = arrays["uid_train"]
    uid_val = arrays["uid_val"]
    uid_test = arrays["uid_test"]

    overlap_train_val = int(len(set(uid_train.tolist()) & set(uid_val.tolist())))
    overlap_train_test = int(len(set(uid_train.tolist()) & set(uid_test.tolist())))
    overlap_val_test = int(len(set(uid_val.tolist()) & set(uid_test.tolist())))

    print(
        f"Raw split sizes before within-split deduplication: "
        f"train={len(arrays['x_train_raw'])}, val={len(arrays['x_val_raw'])}, test={len(arrays['x_test_raw'])}"
    )
    print_label_distribution("Train split raw", arrays["y_train_raw"])
    print_label_distribution("Val split raw", arrays["y_val_raw"])
    print_label_distribution("Test split raw", arrays["y_test_raw"])

    print(
        f"Unique-user split sizes after within-split deduplication: "
        f"train={len(x_train)}, val={len(x_val)}, test={len(x_test)}"
    )
    print(
        f"Removed duplicate rows within split: "
        f"train={len(arrays['x_train_raw']) - len(x_train)}, "
        f"val={len(arrays['x_val_raw']) - len(x_val)}, "
        f"test={len(arrays['x_test_raw']) - len(x_test)}"
    )
    print_label_distribution("Train split unique-user", y_train)
    print_label_distribution("Val split unique-user", y_val)
    print_label_distribution("Test split unique-user", y_test)

    print(
        f"User overlap across provided splits after deduplication: "
        f"train/val={overlap_train_val}, train/test={overlap_train_test}, val/test={overlap_val_test}"
    )

    best_result, tried_results = evaluate_logreg_grid(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    print("Best logistic-regression setting:")
    print(json.dumps({
        "classifier": best_result["classifier"],
        "params": best_result["params"],
        "train_accuracy": best_result["train_accuracy"],
        "val_accuracy": best_result["val_accuracy"],
        "test_accuracy": best_result["test_accuracy"],
    }, indent=2))
    print_prediction_block("train", best_result["train_prediction_summary"])
    print_prediction_block("val", best_result["val_prediction_summary"])
    print_prediction_block("test", best_result["test_prediction_summary"])

    result = {
        "experiment_id": experiment_id,
        "experiment": args.experiment,
        "dataset": args.dataset,
        "model": args.model,
        "model_name_or_path": args.model_name_or_path,
        "bias_type": args.bias_type,
        "npz_path": args.npz_path,
        "feature_keys": {
            "train": arrays["keys"]["x_train"],
            "val": arrays["keys"]["x_val"],
            "test": arrays["keys"]["x_test"],
        },
        "user_id_keys": {
            "train": arrays["keys"]["uid_train"],
            "val": arrays["keys"]["uid_val"],
            "test": arrays["keys"]["uid_test"],
        },
        "label_keys": {
            "train": arrays["keys"]["label_train"],
            "val": arrays["keys"]["label_val"],
            "test": arrays["keys"]["label_test"],
        },
        "deduplication": {
            "strategy": "keep_first_by_userid_within_each_existing_split",
            "raw_n_train": int(len(arrays["x_train_raw"])),
            "raw_n_val": int(len(arrays["x_val_raw"])),
            "raw_n_test": int(len(arrays["x_test_raw"])),
            "unique_user_n_train": int(len(x_train)),
            "unique_user_n_val": int(len(x_val)),
            "unique_user_n_test": int(len(x_test)),
            "removed_duplicate_rows_train": int(len(arrays["x_train_raw"]) - len(x_train)),
            "removed_duplicate_rows_val": int(len(arrays["x_val_raw"]) - len(x_val)),
            "removed_duplicate_rows_test": int(len(arrays["x_test_raw"]) - len(x_test)),
            "kept_indices_preview": {
                "train": arrays["keep_train"][:10].astype(int).tolist(),
                "val": arrays["keep_val"][:10].astype(int).tolist(),
                "test": arrays["keep_test"][:10].astype(int).tolist(),
            },
        },
        "label_construction": {
            "bias_type": args.bias_type,
            "dataset": args.dataset,
            "note": "labels are taken directly from the existing train/val/test arrays in the NPZ file; no thresholding is applied",
        },
        "split_sizes": {
            "n_train": int(len(x_train)),
            "n_val": int(len(x_val)),
            "n_test": int(len(x_test)),
        },
        "label_distributions": {
            "train_raw": counts_by_label(arrays["y_train_raw"]),
            "val_raw": counts_by_label(arrays["y_val_raw"]),
            "test_raw": counts_by_label(arrays["y_test_raw"]),
            "train": counts_by_label(y_train),
            "val": counts_by_label(y_val),
            "test": counts_by_label(y_test),
        },
        "user_id_overlap": {
            "train_val": overlap_train_val,
            "train_test": overlap_train_test,
            "val_test": overlap_val_test,
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