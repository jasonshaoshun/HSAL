#!/usr/bin/env python3
"""
analyze_kmeans_natural_disjoint_v2.py

USER-DISJOINT + Z-balanced subset:
- group split by raw x_userid (split_seed)
- Z-balance within each pool:
    train uses downsample_seed
    test uses test_downsample_seed (default downsample_seed+10000)
- x_name_cls only, PCA dim fixed to 100

Fixes:
- raw userid used for grouping
- test_downsample_seed default matches make scripts
"""

from __future__ import annotations
import argparse, os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize

RANDOM_STATE = 0
NAME_PCA_DIM = 100
SILHOUETTE_SAMPLE_SIZE = 20000
LR_MAX_ITER = 1000


def parse_candidate_ks(s: str) -> list[int]:
    if s is None or s.strip() == "":
        return [2, 4, 8, 16, 32]
    return [int(x) for x in s.split(",") if x.strip()]


def quadrant_label(z_bin: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    return (z_bin.astype(int) * 2 + y_bin.astype(int)).astype(int)


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
    idx = np.concatenate([keep0, keep1]).astype(int)
    rng.shuffle(idx)
    return idx


def prepare_name_features(x_name_cls: np.ndarray) -> np.ndarray:
    X = x_name_cls.astype(np.float32, copy=False)
    X = normalize(X, norm="l2")
    if X.shape[1] > NAME_PCA_DIM:
        ncomp = int(min(NAME_PCA_DIM, X.shape[0], X.shape[1]))
        X = PCA(n_components=ncomp, random_state=RANDOM_STATE).fit_transform(X)
    return X


def kmeans_labels_and_silhouette(X: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    labels = KMeans(n_clusters=int(k), random_state=RANDOM_STATE, n_init=20).fit_predict(X).astype(np.int64)
    n = X.shape[0]
    if k < 2 or k >= n or np.unique(labels).size < 2:
        return labels, float("nan")
    if SILHOUETTE_SAMPLE_SIZE is not None and n > SILHOUETTE_SAMPLE_SIZE:
        s = silhouette_score(X, labels, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=RANDOM_STATE)
    else:
        s = silhouette_score(X, labels)
    return labels, float(s)


def fit_lr_binary(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float, np.ndarray]:
    clf = LogisticRegression(max_iter=LR_MAX_ITER, n_jobs=-1, random_state=RANDOM_STATE)
    clf.fit(x_train, y_train)
    return float(clf.score(x_train, y_train)), float(clf.score(x_test, y_test)), clf.predict(x_test)


def fit_lr_multiclass_acc(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    clf = LogisticRegression(max_iter=LR_MAX_ITER, n_jobs=-1, random_state=RANDOM_STATE)
    clf.fit(x_train, y_train)
    return float(clf.score(x_train, y_train)), float(clf.score(x_test, y_test))


def tpr_gap_multigroup(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> tuple[float, float, float, int, int]:
    gaps = []
    gap_y0 = float("nan")
    gap_y1 = float("nan")
    used0 = 0
    used1 = 0

    for y_c in (0, 1):
        tprs = []
        for g in np.unique(group):
            mask = (y_true == y_c) & (group == g)
            if int(mask.sum()) == 0:
                continue
            tprs.append(float((y_pred[mask] == y_c).mean()))
        if len(tprs) >= 2:
            gap = float(np.std(tprs))
            if y_c == 0:
                gap_y0 = gap
                used0 = len(tprs)
            else:
                gap_y1 = gap
                used1 = len(tprs)
            gaps.append(gap)

    tpr_gap = float(np.sqrt(np.mean(np.square(gaps)))) if gaps else float("nan")
    return tpr_gap, gap_y0, gap_y1, used0, used1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", required=True)
    ap.add_argument("--out_xlsx", required=True)
    ap.add_argument("--y_key", default="y_helpfulness")
    ap.add_argument("--z_key", required=True, choices=["y_reputation", "y_engagement"])
    ap.add_argument("--y_thr", type=float, required=True)
    ap.add_argument("--z_thr", type=float, required=True)
    ap.add_argument("--split_seed", "--seed", dest="split_seed", type=int, required=True)
    ap.add_argument("--downsample_seed", type=int, default=0)
    ap.add_argument("--test_downsample_seed", type=int, default=None, help="Default: downsample_seed+10000")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--candidate_ks", default="")
    args = ap.parse_args()

    if args.test_downsample_seed is None:
        args.test_downsample_seed = int(args.downsample_seed) + 10000

    os.makedirs(os.path.dirname(args.out_xlsx) or ".", exist_ok=True)

    data = np.load(args.input_npz, allow_pickle=True)
    for k in ["x_text_cls", "x_name_cls", "x_userid", args.y_key, args.z_key]:
        if k not in data:
            raise KeyError(f"Missing key {k}")

    x_text_cls = data["x_text_cls"]
    x_name_cls = data["x_name_cls"]
    x_userid_groups = data["x_userid"]
    y_score = data[args.y_key].astype(float)
    z_score = data[args.z_key].astype(float)

    y_bin = (y_score >= args.y_thr).astype(int)
    z_bin = (z_score >= args.z_thr).astype(int)

    idx_all = np.arange(len(y_bin))
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.split_seed)
    tr_pool, te_pool = next(gss.split(idx_all, groups=x_userid_groups))

    idx_train = zbalanced_indices(tr_pool, z_bin, args.downsample_seed)
    idx_test = zbalanced_indices(te_pool, z_bin, args.test_downsample_seed)
    if idx_train.size == 0 or idx_test.size == 0:
        raise ValueError("Cannot form Z-balanced subset in train/test pools: one Z group empty.")

    print(f"Total {idx_train.size} train and {idx_test.size} test.")

    main_tr_acc, main_te_acc, y_pred_test = fit_lr_binary(
        x_text_cls[idx_train], y_bin[idx_train],
        x_text_cls[idx_test], y_bin[idx_test],
    )

    Xname = prepare_name_features(x_name_cls)
    ks = parse_candidate_ks(args.candidate_ks)

    rows = []
    for k in ks:
        name_cluster, sil = kmeans_labels_and_silhouette(Xname, k)

        prot_tr_acc, prot_te_acc = fit_lr_multiclass_acc(
            x_name_cls[idx_train], name_cluster[idx_train],
            x_name_cls[idx_test], name_cluster[idx_test],
        )

        tpr_gap, gap_y0, gap_y1, used0, used1 = tpr_gap_multigroup(
            y_true=y_bin[idx_test],
            y_pred=y_pred_test,
            group=name_cluster[idx_test],
        )
        
        print(f"k={k}: silhouette={sil:.4f}, tpr_gap={tpr_gap:.4f}, prot_te_acc={prot_te_acc:.4f}, prot_tr_acc={prot_tr_acc:.4f}, main_tr_acc={main_tr_acc:.4f}, main_te_acc={main_te_acc:.4f}, gap_y0={gap_y0:.4f} (used {used0} groups), gap_y1={gap_y1:.4f} (used {used1} groups)")

        rows.append(dict(
            k=int(k),
            silhouette=float(sil),
            tpr_gap=float(tpr_gap),
            protected_test_acc=float(prot_te_acc),
            protected_train_acc=float(prot_tr_acc),
            main_train_acc=float(main_tr_acc),
            main_test_acc=float(main_te_acc),

            gap_y0=float(gap_y0),
            gap_y1=float(gap_y1),
            n_groups_used_y0=int(used0),
            n_groups_used_y1=int(used1),

            N_train=int(idx_train.size),
            N_test=int(idx_test.size),
            achieved_test_ratio=float(idx_test.size / (idx_train.size + idx_test.size)),
            y_thr=float(args.y_thr),
            z_thr=float(args.z_thr),
            split_seed=int(args.split_seed),
            downsample_seed=int(args.downsample_seed),
            test_downsample_seed=int(args.test_downsample_seed),
        ))

    df = pd.DataFrame(rows)
    leading = ["k", "silhouette", "tpr_gap", "protected_test_acc", "protected_train_acc", "main_train_acc", "main_test_acc"]
    df = df[leading + [c for c in df.columns if c not in leading]]
    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="kmeans", index=False)

    print(f"Wrote {len(df)} rows to {args.out_xlsx}")


if __name__ == "__main__":
    main()