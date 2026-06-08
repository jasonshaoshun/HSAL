#!/usr/bin/env python3
"""
Encode unique display names, cluster all names, and annotate only a hand-prepared list.

Supported workflow:
1) filter   -> create english_only_unique_names_filtered.jsonl from the raw unique-name jsonl
2) encode   -> encode ALL names in the chosen input jsonl
3) cluster  -> cluster ALL encoded names, but only annotate names from prepared_annotation_names.csv
4) all      -> filter + encode + cluster in one run
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

RANDOM_STATE = 0
NAME_PCA_DIM = 100
FIG_PCA_DIM = 2

DEFAULT_ANNOTATIONS = [
    ("Anglo / English-coded", "Kate Gregory"),
    ("Anglo / English-coded", "Charlotte Farley"),
    ("Anglo / English-coded", "Rachel"),
    ("Anglo / English-coded", "Peter Taylor"),
    ("Anglo / English-coded", "Richard Gadsden"),
    ("Anglo / English-coded", "Brian M. Scott"),
    ("South Asian-coded", "Vaibhav Garg"),
    ("South Asian-coded", "Ankur Banerjee"),
    ("South Asian-coded", "Vamsi Emani"),
    ("South Asian-coded", "Akshay"),
    ("East Asian-coded", "Tsuyoshi Ito"),
    ("East Asian-coded", "Yoichi Oishi"),
    ("Middle Eastern / Persian / Arabic-coded", "Omar Kooheji"),
    ("Middle Eastern / Persian / Arabic-coded", "Saeed Neamati"),
    ("Continental / Eastern / Northern European-coded", "Claudiu"),
    ("Continental / Eastern / Northern European-coded", "Marcin"),
    ("Backup", "Monica Cellio"),
    ("Backup", "Ellen Spertus"),
    ("Backup", "Jeanne Pindar"),
    ("Backup", "Mehper C. Palavuzlar"),
    ("Backup", "Felipe Sabino"),
    ("Backup", "Christoffer Hammarström"),
    ("Backup", "Stein G. Strindhaug"),
    ("Backup", "Hannele"),
    ("Backup", "Frantisek"),
    ("Backup", "Bogdan"),
    ("Backup", "Jimi Oke"),
    ("Backup", "Paul Lammertsma"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, default=None,
                    help="Input jsonl. For filter/encode/all this should point to the name jsonl.")
    ap.add_argument("--mode", type=str, choices=["filter", "encode", "cluster", "all"], default="all")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_name_or_path", type=str, default=None)
    ap.add_argument("--embeddings_npz", type=str, default=None)
    ap.add_argument("--names_csv", type=str, default=None,
                    help="CSV from encode step containing names in exact embedding row order.")
    ap.add_argument("--text_col", type=str, default="DisplayName")
    ap.add_argument("--n_clusters", type=int, default=12)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--annotation_list_csv", type=str, default=None,
                    help="CSV listing names to annotate. If absent, a template is created in output_dir.")
    ap.add_argument("--annotation_fontsize", type=int, default=10)
    ap.add_argument("--scatter_alpha", type=float, default=0.75)
    ap.add_argument("--point_size", type=float, default=18.0)

    # filtering controls
    ap.add_argument("--min_name_len", type=int, default=2)
    ap.add_argument("--max_name_len", type=int, default=40)
    ap.add_argument("--max_digits", type=int, default=2)
    ap.add_argument("--max_symbol_ratio", type=float, default=0.35)
    ap.add_argument("--drop_handle_patterns", action="store_true", default=True)
    return ap.parse_args()


def read_jsonl(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def write_jsonl(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def normalize_display_name(x: str) -> str:
    if x is None:
        return ""
    return " ".join(str(x).strip().split())


def make_default_annotation_df() -> pd.DataFrame:
    rows = []
    for category, name in DEFAULT_ANNOTATIONS:
        rows.append({
            "enabled": 1,
            "category": category,
            "DisplayName": name,
            "dx": 4.0,
            "dy": 4.0,
        })
    return pd.DataFrame(rows)


def ensure_annotation_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        make_default_annotation_df().to_csv(path, index=False)


def load_annotation_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"enabled", "category", "DisplayName"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotation CSV missing required columns: {sorted(missing)}")
    if "dx" not in df.columns:
        df["dx"] = 4.0
    if "dy" not in df.columns:
        df["dy"] = 4.0
    df["DisplayNameClean"] = df["DisplayName"].map(normalize_display_name)
    df["enabled"] = df["enabled"].fillna(1).astype(int)
    df["dx"] = df["dx"].fillna(4.0).astype(float)
    df["dy"] = df["dy"].fillna(4.0).astype(float)
    return df


def is_obvious_bad_name(name: str, args: argparse.Namespace) -> tuple[bool, str]:
    s = normalize_display_name(name)
    if not s:
        return True, "empty"
    if len(s) < args.min_name_len:
        return True, "too_short"
    if len(s) > args.max_name_len:
        return True, "too_long"
    lower = s.lower()
    if "@" in s:
        return True, "email_like"
    if "http" in lower or "www." in lower:
        return True, "url_like"
    if not re.search(r"[A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF]", s):
        return True, "no_letters"
    if re.fullmatch(r"[\W_]+", s):
        return True, "symbols_only"
    if args.drop_handle_patterns and re.fullmatch(r"user[_\- ]?\d+", lower):
        return True, "auto_user_handle"
    if args.drop_handle_patterns and lower in {
        "anonymous", "anon", "n/a", "na", "none", "null", "deleted user", "goodbye stack exchange"
    }:
        return True, "placeholder"
    digit_count = sum(ch.isdigit() for ch in s)
    if digit_count > args.max_digits:
        return True, "too_many_digits"
    symbol_count = sum((not ch.isalnum()) and (not ch.isspace()) for ch in s)
    if len(s) > 0 and (symbol_count / len(s)) > args.max_symbol_ratio:
        return True, "too_many_symbols"
    return False, ""


def filter_names(df: pd.DataFrame, text_col: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if text_col not in df.columns:
        raise ValueError(f"Column {text_col!r} not found in input jsonl")

    out = df.copy()
    out[text_col] = out[text_col].astype(str)
    out["DisplayNameClean"] = out[text_col].map(normalize_display_name)
    keep_flags = []
    drop_reasons = []
    for s in out["DisplayNameClean"].tolist():
        bad, reason = is_obvious_bad_name(s, args)
        keep_flags.append(not bad)
        drop_reasons.append(reason)
    out["keep"] = keep_flags
    out["drop_reason"] = drop_reasons

    # deduplicate by cleaned name, preserve first occurrence
    out = out.drop_duplicates(subset=["DisplayNameClean"]).reset_index(drop=True)
    kept = out[out["keep"]].copy().reset_index(drop=True)
    dropped = out[~out["keep"]].copy().reset_index(drop=True)
    return out, kept, dropped


def load_model_and_tokenizer(model_name_or_path: str, device: str | None = None):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    lower_name = model_name_or_path.lower()
    is_decoder_only = ("llama" in lower_name) or ("mistral" in lower_name)

    if is_decoder_only:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto")
    else:
        model = AutoModel.from_pretrained(model_name_or_path, torch_dtype="auto")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def encode_texts(model, tokenizer, texts: Iterable[str], model_name_or_path: str, max_length: int = 64, device: str = "cpu"):
    is_decoder_only = ("llama" in model_name_or_path.lower()) or ("mistral" in model_name_or_path.lower())
    avg_list, cls_list = [], []

    with torch.no_grad():
        for text in tqdm(texts, desc="Encoding names"):
            text = "" if text is None else str(text)
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1)
            avg_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            if is_decoder_only:
                last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)
                cls_emb = hidden[0, last_idx.item(), :].unsqueeze(0)
            else:
                cls_emb = hidden[:, 0, :]

            avg_list.append(avg_emb.detach().to(torch.float32).cpu())
            cls_list.append(cls_emb.detach().to(torch.float32).cpu())

            del enc, input_ids, attention_mask, outputs, hidden, mask, avg_emb, cls_emb
            if device == "cuda":
                torch.cuda.empty_cache()

    avg_np = torch.cat(avg_list, dim=0).numpy() if avg_list else np.empty((0, 0), dtype=np.float32)
    cls_np = torch.cat(cls_list, dim=0).numpy() if cls_list else np.empty((0, 0), dtype=np.float32)
    return avg_np, cls_np


def prepare_name_features(x_name_cls: np.ndarray) -> np.ndarray:
    X = x_name_cls.astype(np.float32, copy=False)
    X = normalize(X, norm="l2")
    if X.shape[1] > NAME_PCA_DIM:
        ncomp = int(min(NAME_PCA_DIM, X.shape[0], X.shape[1]))
        X = PCA(n_components=ncomp, random_state=RANDOM_STATE).fit_transform(X)
    return X


def build_cluster_summary(names_df: pd.DataFrame, X_features: np.ndarray, labels: np.ndarray, top_k: int = 6) -> pd.DataFrame:
    rows = []
    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        cluster_X = X_features[idx]
        centroid = cluster_X.mean(axis=0, keepdims=True)
        dists = ((cluster_X - centroid) ** 2).sum(axis=1)
        rep_idx = idx[np.argsort(dists)[:top_k]]
        rep_names = names_df.iloc[rep_idx]["DisplayNameClean"].tolist()
        rows.append({
            "cluster": int(c),
            "cluster_size": int(len(idx)),
            "representative_names": " | ".join(rep_names),
        })
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def annotate_selected_names(ax, names_df: pd.DataFrame, X_plot: np.ndarray, annotation_df: pd.DataFrame, fontsize: int):
    found_rows = []
    missing_rows = []

    index_by_name: dict[str, list[int]] = {}
    for i, name in enumerate(names_df["DisplayNameClean"].tolist()):
        index_by_name.setdefault(name, []).append(i)

    for _, row in annotation_df.iterrows():
        if int(row["enabled"]) != 1:
            continue
        target = row["DisplayNameClean"]
        category = row["category"]
        matches = index_by_name.get(target, [])
        if not matches:
            missing_rows.append({
                "category": category,
                "DisplayName": row["DisplayName"],
                "DisplayNameClean": target,
                "reason": "name_not_found_in_names_df",
            })
            continue

        j = matches[0]
        x, y = float(X_plot[j, 0]), float(X_plot[j, 1])
        dx, dy = float(row["dx"]), float(row["dy"])
        txt = ax.annotate(
            row["DisplayName"],
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            fontweight="bold",
            color="black",
            alpha=0.98,
            zorder=8,
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
        found_rows.append({
            "category": category,
            "DisplayName": row["DisplayName"],
            "DisplayNameClean": target,
            "match_index": int(j),
            "pca_x": x,
            "pca_y": y,
            "dx": dx,
            "dy": dy,
            "cluster": int(names_df.iloc[j]["cluster"]) if "cluster" in names_df.columns else None,
        })

    return pd.DataFrame(found_rows), pd.DataFrame(missing_rows)


def plot_clusters(
    names_df: pd.DataFrame,
    X_plot: np.ndarray,
    labels: np.ndarray,
    output_png: str,
    title: str,
    annotation_df: pd.DataFrame,
    fontsize: int,
    point_size: float,
    scatter_alpha: float,
):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, s=point_size, alpha=scatter_alpha)

    found_df, missing_df = annotate_selected_names(ax, names_df, X_plot, annotation_df, fontsize)

    ax.set_title(title)
    ax.set_xlabel("PCA-1", fontsize=36)
    ax.set_ylabel("PCA-2", fontsize=36)
    plt.tight_layout()
    plt.savefig(output_png, dpi=220, bbox_inches="tight")
    output_pdf = str(Path(output_png).with_suffix(".pdf"))
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    return found_df, missing_df


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    filtered_jsonl_path = outdir / "english_only_unique_names_filtered.jsonl"
    dropped_jsonl_path = outdir / "english_only_unique_names_dropped.jsonl"
    audit_csv_path = outdir / "english_only_unique_names_filter_audit.csv"
    names_csv_path = outdir / "names_used_for_encoding.csv"
    annotation_csv_path = Path(args.annotation_list_csv) if args.annotation_list_csv else (outdir / "prepared_annotation_names.csv")
    ensure_annotation_csv(annotation_csv_path)

    if args.mode in {"filter", "all"}:
        if args.input_jsonl is None:
            raise ValueError("--input_jsonl is required for filter/all")
        raw_df = read_jsonl(args.input_jsonl)
        analyzed, kept, dropped = filter_names(raw_df, args.text_col, args)
        analyzed.to_csv(audit_csv_path, index=False)
        write_jsonl(kept[[args.text_col, "DisplayNameClean"]], filtered_jsonl_path)
        write_jsonl(dropped[[args.text_col, "DisplayNameClean", "drop_reason"]], dropped_jsonl_path)
        summary = {
            "n_total": int(len(analyzed)),
            "n_kept": int(len(kept)),
            "n_dropped": int(len(dropped)),
            "drop_reason_counts": analyzed.loc[analyzed["drop_reason"] != "", "drop_reason"].value_counts().to_dict(),
        }
        with open(outdir / "filter_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.mode in {"encode", "all"}:
        if args.input_jsonl is None:
            raise ValueError("--input_jsonl is required for encode/all")
        if not args.model_name_or_path:
            raise ValueError("--model_name_or_path is required for encode/all")
        names_df = read_jsonl(args.input_jsonl)
        if args.text_col not in names_df.columns:
            raise ValueError(f"Column {args.text_col!r} not found in input jsonl")
        names_df["DisplayNameClean"] = names_df[args.text_col].map(normalize_display_name)
        texts = names_df["DisplayNameClean"].tolist()
        tokenizer, model, device = load_model_and_tokenizer(args.model_name_or_path, args.device)
        avg_np, cls_np = encode_texts(model, tokenizer, texts, args.model_name_or_path, max_length=args.max_length, device=device)
        np.savez_compressed(outdir / "name_embeddings.npz", x_name_avg=avg_np, x_name_cls=cls_np)
        names_df[[args.text_col, "DisplayNameClean"]].to_csv(names_csv_path, index=False)

    if args.mode in {"cluster", "all"}:
        emb_path = Path(args.embeddings_npz) if args.embeddings_npz else (outdir / "name_embeddings.npz")
        if not emb_path.exists():
            raise ValueError("Need --embeddings_npz or an existing name_embeddings.npz in output_dir for cluster/all")

        names_source = Path(args.names_csv) if args.names_csv else names_csv_path
        if not names_source.exists():
            raise ValueError("Need --names_csv or names_used_for_encoding.csv from the encode step")

        names_df = pd.read_csv(names_source)
        if "DisplayNameClean" not in names_df.columns:
            col = args.text_col if args.text_col in names_df.columns else names_df.columns[0]
            names_df["DisplayNameClean"] = names_df[col].map(normalize_display_name)

        emb = np.load(emb_path)
        x_name_cls = emb["x_name_cls"]
        if len(names_df) != len(x_name_cls):
            raise ValueError(f"Row mismatch: names_csv has {len(names_df)} rows but embeddings have {len(x_name_cls)} rows")

        X = prepare_name_features(x_name_cls)
        labels = KMeans(n_clusters=args.n_clusters, random_state=RANDOM_STATE, n_init=20).fit_predict(X).astype(np.int64)
        X_plot = PCA(n_components=FIG_PCA_DIM, random_state=RANDOM_STATE).fit_transform(X)

        names_df = names_df.copy()
        names_df["cluster"] = labels
        names_df["pca_x"] = X_plot[:, 0]
        names_df["pca_y"] = X_plot[:, 1]
        names_df.to_csv(outdir / "cluster_assignments.csv", index=False)

        metrics = {
            "n_names": int(len(names_df)),
            "n_clusters": int(args.n_clusters),
            "silhouette_score": float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else None,
            "davies_bouldin_score": float(davies_bouldin_score(X, labels)) if len(np.unique(labels)) > 1 else None,
            "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)) if len(np.unique(labels)) > 1 else None,
        }
        with open(outdir / "cluster_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        summary_df = build_cluster_summary(names_df, X, labels, top_k=6)
        summary_df.to_csv(outdir / "cluster_summary.csv", index=False)

        annotation_df = load_annotation_df(annotation_csv_path)
        title_stub = Path(args.model_name_or_path).name if args.model_name_or_path else emb_path.parent.name
        found_df, missing_df = plot_clusters(
            names_df=names_df,
            X_plot=X_plot,
            labels=labels,
            output_png=str(outdir / "cluster_plot.png"),
            # title=f"Name clusters: {title_stub}",
            title='',
            annotation_df=annotation_df,
            fontsize=args.annotation_fontsize,
            point_size=args.point_size,
            scatter_alpha=args.scatter_alpha,
        )
        found_df.to_csv(outdir / "annotation_names_found.csv", index=False)
        missing_df.to_csv(outdir / "annotation_names_missing.csv", index=False)


if __name__ == "__main__":
    main()
