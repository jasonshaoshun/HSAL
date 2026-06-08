#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize

NAME_PCA_DIM = 100
SECOND_SPLIT_SEED_OFFSET = 100000

TARGET_COUNTRIES_BY_DATASET = {
    "mathematics": ["USA", "FRA", "CAN", "DEU", "GBR"],
    "stackoverflow": ["USA", "IND", "DEU", "GBR", "CAN"],
    "english": ["GBR", "USA", "CAN", "IND", "ITA"],
    "superuser": ["USA", "GBR", "CAN", "DEU", "IND"],
}

def compute_name_cluster_labels(x_name_cls: np.ndarray, n_clusters: int) -> np.ndarray:
    if n_clusters is None or int(n_clusters) <= 0:
        raise ValueError("--n_clusters must be a positive int.")
    X = x_name_cls.astype(np.float32, copy=False)
    X = normalize(X, norm="l2")
    if X.shape[1] > NAME_PCA_DIM:
        ncomp = int(min(NAME_PCA_DIM, X.shape[0], X.shape[1]))
        X = PCA(n_components=ncomp, random_state=0).fit_transform(X)
    labels = KMeans(n_clusters=int(n_clusters), random_state=0, n_init=20).fit_predict(X).astype(np.int64)
    return labels

def zbalanced_indices(pool_idx: np.ndarray, z_bin: np.ndarray, seed: int) -> np.ndarray:
    idx0 = pool_idx[z_bin[pool_idx] == 0]
    idx1 = pool_idx[z_bin[pool_idx] == 1]
    if idx0.size == 0 or idx1.size == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    if idx0.size <= idx1.size:
        keep0 = idx0
        keep1 = rng.choice(idx1, size=idx0.size, replace=False)
    else:
        keep1 = idx1
        keep0 = rng.choice(idx0, size=idx1.size, replace=False)
    out = np.concatenate([keep0, keep1]).astype(int)
    rng.shuffle(out)
    return out

def compute_filter_mask(
    n: int,
    bias_type: str,
    dataset_name: str,
    y_gender: np.ndarray | None,
    location: np.ndarray,
    location_label_values: np.ndarray | None,
) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    if bias_type == "gender":
        if y_gender is None:
            raise KeyError("bias_type='gender' requires y_gender in the input npz.")
        mask &= np.isin(y_gender, [0, 1])
    elif bias_type == "location":
        if location_label_values is None:
            raise KeyError("bias_type='location' requires location_label_values in the input npz.")
        target = TARGET_COUNTRIES_BY_DATASET[str(dataset_name).lower()]
        labels = np.asarray(location_label_values).astype(str)
        target_codes = np.array([i for i, lab in enumerate(labels.tolist()) if lab in target], dtype=int)
        mask &= np.isin(location, target_codes)
    return mask

def remap_location_to_target_order(
    location_values: np.ndarray,
    dataset_name: str,
    location_label_values: np.ndarray | None,
):
    if location_label_values is None:
        raise KeyError("location remapping requires location_label_values in the input npz.")
    target_locations = TARGET_COUNTRIES_BY_DATASET[str(dataset_name).lower()]
    labels = np.asarray(location_label_values).astype(str)
    orig_code_to_label = {int(i): str(label) for i, label in enumerate(labels.tolist())}
    label_to_new_code = {location: i + 1 for i, location in enumerate(target_locations)}
    remapped = np.array(
        [label_to_new_code[orig_code_to_label[int(code)]] for code in np.asarray(location_values).tolist()],
        dtype=int,
    )
    saved_location_label_values = np.array(["UNKNOWN"] + target_locations, dtype=object)
    return remapped, saved_location_label_values

def choose_best_ratio_matched_split_3way(
    idx_all: np.ndarray,
    groups: np.ndarray,
    z_bin: np.ndarray,
    val_size: float,
    test_size: float,
    seed_start: int,
    split_tries: int,
):
    val_rel = val_size / (1.0 - test_size)
    global_ratio = float(z_bin[idx_all].mean())
    best = None
    for seed in range(seed_start, seed_start + split_tries):
        val_split_seed = int(seed + SECOND_SPLIT_SEED_OFFSET)
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_rel, test_rel = next(gss_test.split(idx_all, groups=groups[idx_all]))
        trainval_pool = idx_all[trainval_rel]
        test_pool = idx_all[test_rel]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=val_split_seed)
        tr_rel, va_rel = next(gss_val.split(trainval_pool, groups=groups[trainval_pool]))
        train_pool = trainval_pool[tr_rel]
        val_pool = trainval_pool[va_rel]
        if min(len(train_pool), len(val_pool), len(test_pool)) == 0:
            continue
        tr_ratio = float(z_bin[train_pool].mean())
        va_ratio = float(z_bin[val_pool].mean())
        te_ratio = float(z_bin[test_pool].mean())
        ratio_diff = abs(tr_ratio - global_ratio) + abs(va_ratio - global_ratio) + abs(te_ratio - global_ratio)
        cand = (-ratio_diff, min(len(train_pool), len(val_pool), len(test_pool)), seed, train_pool, val_pool, test_pool, global_ratio, tr_ratio, va_ratio, te_ratio)
        if best is None or cand > best:
            best = cand
    if best is None:
        raise ValueError("Could not find a valid ratio-matched split.")
    _, _, seed, train_pool, val_pool, test_pool, global_ratio, tr_ratio, va_ratio, te_ratio = best
    return train_pool, val_pool, test_pool, global_ratio, tr_ratio, va_ratio, te_ratio

def save_npz(
    outpath,
    idx_train, idx_val, idx_test,
    x_text_avg, x_text_cls, x_aboutme_avg, x_aboutme_cls, x_name_avg, x_name_cls,
    location, x_userid_groups, y_bin, z_bin, y_score, z_score, q, name_cluster,
    y_gender, location_label_values, bias_type, dataset_name, z_key,
):
    saved_location_label_values = location_label_values
    if bias_type == "location":
        train_location_out, saved_location_label_values = remap_location_to_target_order(location[idx_train], dataset_name, location_label_values)
        val_location_out, _ = remap_location_to_target_order(location[idx_val], dataset_name, location_label_values)
        test_location_out, _ = remap_location_to_target_order(location[idx_test], dataset_name, location_label_values)
    else:
        train_location_out = location[idx_train].astype(int)
        val_location_out = location[idx_val].astype(int)
        test_location_out = location[idx_test].astype(int)

    common = {
        "x_train_text_cls": x_text_cls[idx_train],
        "x_train_text_avg": x_text_avg[idx_train],
        "x_train_aboutme_cls": x_aboutme_cls[idx_train],
        "x_train_aboutme_avg": x_aboutme_avg[idx_train],
        "y_train_helpfulness": y_bin[idx_train].astype(int),
        "y_train_location": train_location_out.astype(int),
        "y_train_joint": q[idx_train].astype(int),
        "y_train_helpfulness_raw": y_score[idx_train].astype(float),
        "x_train_userid": x_userid_groups[idx_train].astype(str),
        "x_train_name_cls": x_name_cls[idx_train],
        "x_train_name_avg": x_name_avg[idx_train],
        "y_train_name_cluster": name_cluster[idx_train].astype(int),

        "x_val_text_cls": x_text_cls[idx_val],
        "x_val_text_avg": x_text_avg[idx_val],
        "x_val_aboutme_cls": x_aboutme_cls[idx_val],
        "x_val_aboutme_avg": x_aboutme_avg[idx_val],
        "y_val_helpfulness": y_bin[idx_val].astype(int),
        "y_val_location": val_location_out.astype(int),
        "y_val_joint": q[idx_val].astype(int),
        "y_val_helpfulness_raw": y_score[idx_val].astype(float),
        "x_val_userid": x_userid_groups[idx_val].astype(str),
        "x_val_name_cls": x_name_cls[idx_val],
        "x_val_name_avg": x_name_avg[idx_val],
        "y_val_name_cluster": name_cluster[idx_val].astype(int),

        "x_test_text_cls": x_text_cls[idx_test],
        "x_test_text_avg": x_text_avg[idx_test],
        "x_test_aboutme_cls": x_aboutme_cls[idx_test],
        "x_test_aboutme_avg": x_aboutme_avg[idx_test],
        "y_test_helpfulness": y_bin[idx_test].astype(int),
        "y_test_location": test_location_out.astype(int),
        "y_test_joint": q[idx_test].astype(int),
        "y_test_helpfulness_raw": y_score[idx_test].astype(float),
        "x_test_userid": x_userid_groups[idx_test].astype(str),
        "x_test_name_cls": x_name_cls[idx_test],
        "x_test_name_avg": x_name_avg[idx_test],
        "y_test_name_cluster": name_cluster[idx_test].astype(int),
        "location_label_values": np.asarray(location_label_values if bias_type != "location" else saved_location_label_values),
    }

    if y_gender is not None:
        common.update({
            "y_train_gender": y_gender[idx_train].astype(int),
            "y_val_gender": y_gender[idx_val].astype(int),
            "y_test_gender": y_gender[idx_test].astype(int),
        })

    if z_key == "y_reputation":
        common.update({
            "y_train_reputation": z_bin[idx_train].astype(int),
            "y_train_reputation_raw": z_score[idx_train].astype(float),
            "y_val_reputation": z_bin[idx_val].astype(int),
            "y_val_reputation_raw": z_score[idx_val].astype(float),
            "y_test_reputation": z_bin[idx_test].astype(int),
            "y_test_reputation_raw": z_score[idx_test].astype(float),
        })
    elif z_key == "y_engagement":
        common.update({
            "y_train_engagement": z_bin[idx_train].astype(int),
            "y_train_engagement_raw": z_score[idx_train].astype(float),
            "y_val_engagement": z_bin[idx_val].astype(int),
            "y_val_engagement_raw": z_score[idx_val].astype(float),
            "y_test_engagement": z_bin[idx_test].astype(int),
            "y_test_engagement_raw": z_score[idx_test].astype(float),
        })
    else:
        raise ValueError(f"Unsupported z_key={z_key}")

    np.savez(outpath, **common)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", required=True)
    ap.add_argument("--output_npz", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--y_thr", type=int, required=True)
    ap.add_argument("--z_thr", type=int, required=True)
    ap.add_argument("--y_key", default="y_helpfulness")
    ap.add_argument("--z_key", default="y_reputation", choices=["y_reputation", "y_engagement"])
    ap.add_argument("--bias_type", required=True, choices=["reputation", "engagement", "name", "gender", "location"])
    ap.add_argument("--dataset_name", required=True, choices=["mathematics", "stackoverflow", "english", "superuser"])
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, required=True)
    ap.add_argument("--split_tries", type=int, default=50)
    ap.add_argument("--downsample_seed", type=int, default=0)
    ap.add_argument("--val_downsample_seed", type=int, default=None)
    ap.add_argument("--test_downsample_seed", type=int, default=None)
    ap.add_argument("--n_clusters", type=int, required=True)
    args = ap.parse_args()

    data = np.load(args.input_npz, allow_pickle=True)
    x_text_avg = data["x_text_avg"]; x_text_cls = data["x_text_cls"]
    x_aboutme_avg = data["x_aboutme_avg"]; x_aboutme_cls = data["x_aboutme_cls"]
    x_name_avg = data["x_name_avg"]; x_name_cls = data["x_name_cls"]
    location = data["y_location"]
    x_userid_groups = np.asarray(data["x_userid"]).astype(str)
    y_gender = data["y_gender"].astype(int) if "y_gender" in data else None
    location_label_values = data["location_label_values"] if "location_label_values" in data else None
    y_score = data[args.y_key].astype(float)
    z_score = data[args.z_key].astype(float)
    y_bin = (y_score >= int(args.y_thr)).astype(int)
    z_bin = (z_score >= int(args.z_thr)).astype(int)
    q = (z_bin * 2 + y_bin).astype(int)
    name_cluster = compute_name_cluster_labels(x_name_cls, args.n_clusters)

    idx_all = np.arange(len(y_bin))
    if args.bias_type in {"gender", "location"}:
        mask = compute_filter_mask(len(y_bin), args.bias_type, args.dataset_name, y_gender, location, location_label_values)
        idx_all = idx_all[mask]
        if idx_all.size == 0:
            raise ValueError("No rows remain after initial gender/location filtering.")
        idx_train, idx_val, idx_test, global_ratio, tr_ratio, va_ratio, te_ratio = choose_best_ratio_matched_split_3way(
            idx_all, x_userid_groups, z_bin, args.val_size, args.test_size, args.split_seed, args.split_tries
        )
        print(f"Matched Z ratios | global={global_ratio:.4f} train={tr_ratio:.4f} val={va_ratio:.4f} test={te_ratio:.4f}")
    else:
        val_split_seed = int(args.split_seed + SECOND_SPLIT_SEED_OFFSET)
        if args.val_downsample_seed is None:
            args.val_downsample_seed = int(args.downsample_seed) + 5000
        if args.test_downsample_seed is None:
            args.test_downsample_seed = int(args.downsample_seed) + 10000
        val_rel = args.val_size / (1.0 - args.test_size)
        gss_test = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.split_seed)
        trainval_pool, test_pool = next(gss_test.split(idx_all, groups=x_userid_groups))
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=val_split_seed)
        tr_rel, va_rel = next(gss_val.split(trainval_pool, groups=x_userid_groups[trainval_pool]))
        train_pool = trainval_pool[tr_rel]
        val_pool = trainval_pool[va_rel]
        idx_train = zbalanced_indices(train_pool, z_bin, args.downsample_seed)
        idx_val = zbalanced_indices(val_pool, z_bin, args.val_downsample_seed)
        idx_test = zbalanced_indices(test_pool, z_bin, args.test_downsample_seed)
        if idx_train.size == 0 or idx_val.size == 0 or idx_test.size == 0:
            raise ValueError("Cannot form Z-balanced subset in one of train/val/test pools.")

    if args.output_npz is None:
        if args.output_dir is None or args.name is None:
            raise ValueError("Provide either --output_npz or (--output_dir and --name).")
        os.makedirs(args.output_dir, exist_ok=True)
        outpath = os.path.join(args.output_dir, f"{args.name}.npz")
    else:
        outpath = args.output_npz
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    save_npz(outpath, idx_train, idx_val, idx_test, x_text_avg, x_text_cls, x_aboutme_avg, x_aboutme_cls, x_name_avg, x_name_cls, location, x_userid_groups, y_bin, z_bin, y_score, z_score, q, name_cluster, y_gender, location_label_values, args.bias_type, args.dataset_name, args.z_key)
    print(f"Saved {outpath} | train={len(idx_train)} val={len(idx_val)} test={len(idx_test)} | bias_type={args.bias_type} dataset_name={args.dataset_name}")

if __name__ == "__main__":
    main()
