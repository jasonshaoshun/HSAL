#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit


PCTS_19 = list(range(5, 100, 5))
# PCTS_19 = list(range(5, 100, 100))

SECOND_SPLIT_SEED_OFFSET = 100000
VAL_DOWNSAMPLE_SEED_OFFSET = 5000
TEST_DOWNSAMPLE_SEED_OFFSET = 10000


def make_percentile_candidates(values: np.ndarray, percentiles: List[int]) -> List[Tuple[int, int]]:
    """
    Convert percentile thresholds to integer thresholds using ceil().
    For integer-valued scores, (score >= raw_float_threshold) is equivalent to
    (score >= ceil(raw_float_threshold)).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    thrs = np.ceil(np.percentile(values, percentiles)).astype(int)
    out, seen = [], set()
    for thr, pct in zip(thrs.tolist(), percentiles):
        thr = int(thr)
        if thr not in seen:
            out.append((thr, int(pct)))
            seen.add(thr)
    return out


def quadrant_label(z_bin: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    return (2 * z_bin.astype(int) + y_bin.astype(int)).astype(int)


def counts_by_quadrant(q: np.ndarray) -> np.ndarray:
    return np.bincount(q.astype(int), minlength=4).astype(int)


def counts_z(z: np.ndarray) -> Tuple[int, int]:
    n1 = int(z.sum())
    return int(len(z) - n1), n1


def unique_user_counts_by_z(indices: np.ndarray, user_id: np.ndarray, z_bin: np.ndarray) -> Dict[str, int]:
    if len(indices) == 0:
        return {"users_total": 0, "users_z0": 0, "users_z1": 0}
    uid = np.asarray(user_id[indices]).astype(str)
    z_here = np.asarray(z_bin[indices]).astype(int)

    users_z0 = np.unique(uid[z_here == 0]).size
    users_z1 = np.unique(uid[z_here == 1]).size
    users_total = np.unique(uid).size
    return {
        "users_total": int(users_total),
        "users_z0": int(users_z0),
        "users_z1": int(users_z1),
    }


def zbalanced_indices(pool_idx: np.ndarray, z_bin: np.ndarray, seed: int) -> np.ndarray:
    idx0 = pool_idx[z_bin[pool_idx] == 0]
    idx1 = pool_idx[z_bin[pool_idx] == 1]
    if len(idx0) == 0 or len(idx1) == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    if len(idx0) <= len(idx1):
        keep0 = idx0
        keep1 = rng.choice(idx1, size=len(idx0), replace=False)
    else:
        keep1 = idx1
        keep0 = rng.choice(idx0, size=len(idx1), replace=False)
    out = np.concatenate([keep0, keep1]).astype(int)
    rng.shuffle(out)
    return out


def choose_best_split_3way(
    groups: np.ndarray,
    z_bin: np.ndarray,
    val_size: float,
    test_size: float,
    seed_start: int,
    split_tries: int,
) -> Optional[Dict[str, Any]]:
    if not (0 < val_size < 1) or not (0 < test_size < 1) or (val_size + test_size >= 1):
        raise ValueError("Require 0 < val_size < 1, 0 < test_size < 1, and val_size + test_size < 1.")

    train_size = 1.0 - val_size - test_size
    val_rel = val_size / (1.0 - test_size)

    idx_all = np.arange(len(groups))
    best = None

    for seed in range(seed_start, seed_start + split_tries):
        test_split_seed = int(seed)
        val_split_seed = int(seed + SECOND_SPLIT_SEED_OFFSET)

        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=test_split_seed)
        trainval_pool, test_pool = next(gss_test.split(idx_all, groups=groups))

        trainval_groups = groups[trainval_pool]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=val_split_seed)
        tr_rel, va_rel = next(gss_val.split(trainval_pool, groups=trainval_groups))

        train_pool = trainval_pool[tr_rel]
        val_pool = trainval_pool[va_rel]

        tr0, tr1 = counts_z(z_bin[train_pool])
        va0, va1 = counts_z(z_bin[val_pool])
        te0, te1 = counts_z(z_bin[test_pool])

        if min(tr0, tr1, va0, va1, te0, te1) <= 0:
            continue

        train_bal = 2 * min(tr0, tr1)
        val_bal = 2 * min(va0, va1)
        test_bal = 2 * min(te0, te1)
        total_bal = train_bal + val_bal + test_bal

        if total_bal <= 0:
            continue

        train_ratio = train_bal / total_bal
        val_ratio = val_bal / total_bal
        test_ratio = test_bal / total_bal

        ratio_diff = (
            abs(train_ratio - train_size)
            + abs(val_ratio - val_size)
            + abs(test_ratio - test_size)
        )

        cand = (
            total_bal,
            -ratio_diff,
            seed,
            train_pool,
            val_pool,
            test_pool,
            train_ratio,
            val_ratio,
            test_ratio,
            val_split_seed,
        )

        if best is None or cand > best:
            best = cand

    if best is None:
        return None

    (
        _,
        _,
        split_seed_used,
        train_pool,
        val_pool,
        test_pool,
        train_ratio_bal,
        val_ratio_bal,
        test_ratio_bal,
        val_split_seed_used,
    ) = best

    return {
        "split_seed_used": int(split_seed_used),
        "val_split_seed_used": int(val_split_seed_used),
        "train_pool": train_pool,
        "val_pool": val_pool,
        "test_pool": test_pool,
        "train_ratio_bal": float(train_ratio_bal),
        "val_ratio_bal": float(val_ratio_bal),
        "test_ratio_bal": float(test_ratio_bal),
    }


def fit_lr(x_train, y_train, x_eval, y_eval, seed: int, lr_n_jobs: int = 1):
    clf = LogisticRegression(max_iter=1000, n_jobs=lr_n_jobs, random_state=seed)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)
    return float(clf.score(x_train, y_train)), float(clf.score(x_eval, y_eval)), y_pred


def fit_lr_val_test(x_train, y_train, x_val, y_val, x_test, y_test, seed: int, lr_n_jobs: int = 1):
    """
    Same LR model as fit_lr, but fit once and evaluate on both val/test.
    This avoids fitting the identical train set twice for the same target.
    """
    clf = LogisticRegression(max_iter=1000, n_jobs=lr_n_jobs, random_state=seed)
    clf.fit(x_train, y_train)

    y_val_pred = clf.predict(x_val)
    y_test_pred = clf.predict(x_test)

    train_acc = float(clf.score(x_train, y_train))
    val_acc = float((y_val_pred == y_val).mean())
    test_acc = float((y_test_pred == y_test).mean())

    return train_acc, val_acc, test_acc, y_val_pred, y_test_pred


def tpr_details(prefix: str, y_true, y_pred, z_true):
    out = {}
    gaps = []
    for y_c in [0, 1]:
        vals = {}
        for z_g in [0, 1]:
            mask = (y_true == y_c) & (z_true == z_g)
            n = int(mask.sum())
            out[f'{prefix}_n_y{y_c}_z{z_g}'] = n
            out[f'{prefix}_tpr_y{y_c}_z{z_g}'] = np.nan if n == 0 else float((y_pred[mask] == y_c).mean())
            if n > 0:
                vals[z_g] = out[f'{prefix}_tpr_y{y_c}_z{z_g}']
        out[f'{prefix}_tpr_gap_y{y_c}'] = np.nan if len(vals) < 2 else float(abs(vals[1] - vals[0]))
        if len(vals) == 2:
            gaps.append(out[f'{prefix}_tpr_gap_y{y_c}'])
    out[f'{prefix}_tpr_gap'] = np.nan if not gaps else float(np.sqrt(np.mean(np.square(gaps))))
    return out

def class_recall_details(prefix: str, y_true, y_pred):
    out = {}
    for y_c in [0, 1]:
        mask = (y_true == y_c)
        tp = int(((y_pred == y_c) & mask).sum())
        fn = int(((y_pred != y_c) & mask).sum())
        denom = tp + fn

        out[f"{prefix}_tp_y{y_c}"] = tp
        out[f"{prefix}_fn_y{y_c}"] = fn
        out[f"{prefix}_recall_y{y_c}"] = np.nan if denom == 0 else float(tp / denom)
    return out


def process_pair(
    pair_count: int,
    z_thr: int,
    z_pct: int,
    y_thr: int,
    y_pct: int,
    x: np.ndarray,
    y_score: np.ndarray,
    z_score: np.ndarray,
    user_id: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    z_bin = (z_score >= int(z_thr)).astype(int)
    y_bin = (y_score >= int(y_thr)).astype(int)
    q = quadrant_label(z_bin, y_bin)
    avail = counts_by_quadrant(q)

    row = {
        'z_key': args.z_key,
        'y_key': args.y_key,
        'z_thr': int(z_thr),
        'y_thr': int(y_thr),
        'z_pct': int(z_pct),
        'y_pct': int(y_pct),
        'train_size_used': float(1.0 - args.val_size - args.test_size),
        'val_size_used': float(args.val_size),
        'test_size_used': float(args.test_size),
        'train_downsample_seed_used': int(args.downsample_seed),
        'val_downsample_seed_used': int(args.downsample_seed + VAL_DOWNSAMPLE_SEED_OFFSET),
        'test_downsample_seed_used': int(args.downsample_seed + TEST_DOWNSAMPLE_SEED_OFFSET),
        'A00': int(avail[0]),
        'A01': int(avail[1]),
        'A10': int(avail[2]),
        'A11': int(avail[3]),
    }

    if z_bin.min() == z_bin.max() or y_bin.min() == y_bin.max() or (avail == 0).any():
        row.update({
            'flags': 'degenerate_or_missing',
            'N_train': 0, 'N_val': 0, 'N_test': 0, 'N_subset': 0,
            'users_train_total': 0, 'users_train_z0': 0, 'users_train_z1': 0,
            'users_val_total': 0, 'users_val_z0': 0, 'users_val_z1': 0,
            'users_test_total': 0, 'users_test_z0': 0, 'users_test_z1': 0,
        })
        return row

    best = choose_best_split_3way(
        groups=user_id,
        z_bin=z_bin,
        val_size=args.val_size,
        test_size=args.test_size,
        seed_start=args.split_seed_start,
        split_tries=args.split_tries,
    )
    if best is None:
        row.update({
            'flags': 'no_valid_group_split',
            'N_train': 0, 'N_val': 0, 'N_test': 0, 'N_subset': 0,
            'users_train_total': 0, 'users_train_z0': 0, 'users_train_z1': 0,
            'users_val_total': 0, 'users_val_z0': 0, 'users_val_z1': 0,
            'users_test_total': 0, 'users_test_z0': 0, 'users_test_z1': 0,
        })
        return row

    idx_train = zbalanced_indices(best['train_pool'], z_bin, args.downsample_seed)
    idx_val = zbalanced_indices(best['val_pool'], z_bin, args.downsample_seed + VAL_DOWNSAMPLE_SEED_OFFSET)
    idx_test = zbalanced_indices(best['test_pool'], z_bin, args.downsample_seed + TEST_DOWNSAMPLE_SEED_OFFSET)

    if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
        row.update(best)
        row.update({
            'flags': 'no_zbalanced_subset',
            'N_train': 0, 'N_val': 0, 'N_test': 0, 'N_subset': 0,
            'users_train_total': 0, 'users_train_z0': 0, 'users_train_z1': 0,
            'users_val_total': 0, 'users_val_z0': 0, 'users_val_z1': 0,
            'users_test_total': 0, 'users_test_z0': 0, 'users_test_z1': 0,
        })
        row.pop('train_pool', None)
        row.pop('val_pool', None)
        row.pop('test_pool', None)
        return row

    tr_counts = counts_by_quadrant(q[idx_train])
    va_counts = counts_by_quadrant(q[idx_val])
    te_counts = counts_by_quadrant(q[idx_test])

    main_train_acc, main_val_acc, main_test_acc, y_val_pred, y_test_pred = fit_lr_val_test(
        x[idx_train], y_bin[idx_train],
        x[idx_val], y_bin[idx_val],
        x[idx_test], y_bin[idx_test],
        args.probe_seed,
        lr_n_jobs=args.lr_n_jobs,
    )

    prot_train_acc, prot_val_acc, prot_test_acc, _, _ = fit_lr_val_test(
        x[idx_train], z_bin[idx_train],
        x[idx_val], z_bin[idx_val],
        x[idx_test], z_bin[idx_test],
        args.probe_seed,
        lr_n_jobs=args.lr_n_jobs,
    )

    train_user_counts = unique_user_counts_by_z(idx_train, user_id, z_bin)
    val_user_counts = unique_user_counts_by_z(idx_val, user_id, z_bin)
    test_user_counts = unique_user_counts_by_z(idx_test, user_id, z_bin)

    row.update(best)
    row.update({
        'N_train': int(len(idx_train)),
        'N_val': int(len(idx_val)),
        'N_test': int(len(idx_test)),
        'N_subset': int(len(idx_train) + len(idx_val) + len(idx_test)),
        'tr00': int(tr_counts[0]), 'tr01': int(tr_counts[1]), 'tr10': int(tr_counts[2]), 'tr11': int(tr_counts[3]),
        'va00': int(va_counts[0]), 'va01': int(va_counts[1]), 'va10': int(va_counts[2]), 'va11': int(va_counts[3]),
        'te00': int(te_counts[0]), 'te01': int(te_counts[1]), 'te10': int(te_counts[2]), 'te11': int(te_counts[3]),
        'users_train_total': train_user_counts['users_total'],
        'users_train_z0': train_user_counts['users_z0'],
        'users_train_z1': train_user_counts['users_z1'],
        'users_val_total': val_user_counts['users_total'],
        'users_val_z0': val_user_counts['users_z0'],
        'users_val_z1': val_user_counts['users_z1'],
        'users_test_total': test_user_counts['users_total'],
        'users_test_z0': test_user_counts['users_z0'],
        'users_test_z1': test_user_counts['users_z1'],
        'main_train_acc': float(main_train_acc),
        'main_val_acc': float(main_val_acc),
        'main_test_acc': float(main_test_acc),
        'protected_train_acc': float(prot_train_acc),
        'protected_val_acc': float(prot_val_acc),
        'protected_test_acc': float(prot_test_acc),
        'flags': '',
    })
    row.update(tpr_details('val', y_bin[idx_val], y_val_pred, z_bin[idx_val]))
    row.update(tpr_details('te', y_bin[idx_test], y_test_pred, z_bin[idx_test]))

    row.update(class_recall_details('val', y_bin[idx_val], y_val_pred))
    row.update(class_recall_details('te', y_bin[idx_test], y_test_pred))

    row.pop('train_pool', None)
    row.pop('val_pool', None)
    row.pop('test_pool', None)

    print(
        f"Processed pair {pair_count}: "
        f"z_pct={int(z_pct)}, z_thr={int(z_thr)}, "
        f"y_pct={int(y_pct)}, y_thr={int(y_thr)}",
        flush=True,
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_npz', required=True)
    ap.add_argument('--out_xlsx', required=True)
    ap.add_argument('--z_key', default='y_reputation')
    ap.add_argument('--y_key', default='y_helpfulness')
    ap.add_argument('--val_size', type=float, default=0.1)
    ap.add_argument('--test_size', type=float, default=0.2)
    ap.add_argument('--split_seed_start', type=int, default=0)
    ap.add_argument('--split_tries', type=int, default=50)
    ap.add_argument('--downsample_seed', type=int, default=0)
    ap.add_argument('--probe_seed', type=int, default=0)
    ap.add_argument('--max_pairs', type=int, default=10_000)
    ap.add_argument('--parallel_jobs', type=int, default=1)
    ap.add_argument('--lr_n_jobs', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_xlsx) or '.', exist_ok=True)

    data = np.load(args.input_npz, allow_pickle=True)
    x = data['x_text_cls']
    y_score = data[args.y_key].astype(float)
    z_score = data[args.z_key].astype(float)
    user_id = np.asarray(data['x_userid']).astype(str)

    pairs = []
    pair_count = 0
    z_candidates = make_percentile_candidates(z_score, PCTS_19)
    y_candidates = make_percentile_candidates(y_score, PCTS_19)

    for z_thr, z_pct in z_candidates:
        for y_thr, y_pct in y_candidates:
            pair_count += 1
            if pair_count > args.max_pairs:
                break
            pairs.append((pair_count, z_thr, z_pct, y_thr, y_pct))
        if pair_count > args.max_pairs:
            break

    print(
        f"Prepared {len(pairs)} threshold pairs; "
        f"parallel_jobs={args.parallel_jobs}, lr_n_jobs={args.lr_n_jobs}",
        flush=True,
    )

    if args.parallel_jobs == 1:
        rows = [
            process_pair(pair_id, z_thr, z_pct, y_thr, y_pct, x, y_score, z_score, user_id, args)
            for pair_id, z_thr, z_pct, y_thr, y_pct in pairs
        ]
    else:
        rows = Parallel(n_jobs=args.parallel_jobs, backend='loky', verbose=10)(
            delayed(process_pair)(pair_id, z_thr, z_pct, y_thr, y_pct, x, y_score, z_score, user_id, args)
            for pair_id, z_thr, z_pct, y_thr, y_pct in pairs
        )

    df = pd.DataFrame(rows)

    preferred_order = [
        'z_thr', 'y_thr',
        'te_tpr_gap',
        'main_val_acc', 'val_recall_y0', 'val_recall_y1',
        'main_test_acc', 'te_recall_y0', 'te_recall_y1',
        'protected_test_acc',
        'split_seed_used', 'val_split_seed_used',
        'N_train', 'N_val', 'N_test',
        'users_train_z0', 'users_train_z1',
        'users_val_z0', 'users_val_z1',
        'users_test_z0', 'users_test_z1',
    ]
    ordered_cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    df = df.loc[:, ordered_cols]

    df.to_excel(args.out_xlsx, index=False)
    print(f'saved {len(rows)} rows to {args.out_xlsx}')


if __name__ == '__main__':
    main()
