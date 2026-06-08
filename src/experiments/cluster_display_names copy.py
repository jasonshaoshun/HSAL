#!/usr/bin/env python3
"""
Build a filtered unique-name dataset, encode names with a HF model, run PCA+KMeans,
and draw a labelled clustering figure.

Examples
--------
# 1) Filter only
python cluster_display_names.py \
  --input_jsonl english_only_unique_names.jsonl \
  --mode filter \
  --output_dir name_cluster_outputs

# 2) Encode filtered names with one model
python cluster_display_names.py \
  --input_jsonl name_cluster_outputs/english_only_unique_names_filtered.jsonl \
  --mode encode \
  --model_name_or_path bert-base-uncased \
  --output_dir name_cluster_outputs/bert

# 3) Cluster + plot from saved embeddings (uses x_name_cls)
python cluster_display_names.py \
  --mode cluster \
  --embeddings_npz name_cluster_outputs/bert/name_embeddings.npz \
  --output_dir name_cluster_outputs/bert \
  --n_clusters 12
"""

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

RANDOM_STATE = 0
NAME_PCA_DIM = 100
FIG_PCA_DIM = 2

PLACEHOLDER_PATTERNS = [
    re.compile(r'^\s*anonymous\s*$', re.I),
    re.compile(r'^\s*goodbye stack exchange\s*$', re.I),
    re.compile(r'^\s*deleted user\s*$', re.I),
    re.compile(r'^\s*community\s*$', re.I),
    re.compile(r'^\s*guest\s*$', re.I),
    re.compile(r'^\s*admin(istrator)?\s*$', re.I),
    re.compile(r'^\s*moderator\s*$', re.I),
    re.compile(r'^\s*anon\d*user\s*$', re.I),
    re.compile(r'^\s*user\d+\s*$', re.I),
]
EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
URL_RE = re.compile(r'https?://|www\.', re.I)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_jsonl', type=str, default=None)
    ap.add_argument('--mode', type=str, choices=['filter', 'encode', 'cluster', 'all'], default='all')
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--model_name_or_path', type=str, default=None)
    ap.add_argument('--embeddings_npz', type=str, default=None)
    ap.add_argument('--text_col', type=str, default='DisplayName')
    ap.add_argument('--n_clusters', type=int, default=12)
    ap.add_argument('--max_length', type=int, default=64)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--annotation_per_cluster', type=int, default=6)
    ap.add_argument('--min_chars', type=int, default=2)
    ap.add_argument('--max_chars', type=int, default=40)
    ap.add_argument('--digit_ratio_threshold', type=float, default=0.20)
    ap.add_argument('--symbol_letter_ratio_threshold', type=float, default=0.60)
    return ap.parse_args()


def read_jsonl(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def write_jsonl(df: pd.DataFrame, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for row in df.to_dict(orient='records'):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def script_bucket(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return 'UNKNOWN'
    for key in ['LATIN', 'CYRILLIC', 'GREEK', 'ARMENIAN', 'HEBREW', 'ARABIC', 'CJK', 'HIRAGANA', 'KATAKANA', 'HANGUL', 'DEVANAGARI']:
        if key in name:
            return key
    return 'OTHER'


def analyze_name(raw_name: str,
                 min_chars: int = 2,
                 max_chars: int = 40,
                 digit_ratio_threshold: float = 0.20,
                 symbol_letter_ratio_threshold: float = 0.60) -> dict:
    raw = '' if raw_name is None else str(raw_name)
    s = ' '.join(raw.strip().split())

    total = len(s)
    letters = sum(ch.isalpha() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    spaces = sum(ch.isspace() for ch in s)
    punct = total - letters - digits - spaces
    letter_ratio = (letters / total) if total else 0.0
    digit_ratio = (digits / total) if total else 0.0
    scripts = sorted({script_bucket(ch) for ch in s if ch.isalpha()})

    reasons = []
    flags = []

    if s == '':
        reasons.append('empty')
    if any(p.match(s) for p in PLACEHOLDER_PATTERNS):
        reasons.append('placeholder_or_auto_username')
    if EMAIL_RE.search(s):
        reasons.append('contains_email')
    if URL_RE.search(s):
        reasons.append('contains_url')
    if total < min_chars:
        reasons.append('too_short')
    if total > max_chars:
        reasons.append('too_long')
    if letters == 0:
        reasons.append('no_letters')
    if digits >= 3 and digit_ratio > digit_ratio_threshold:
        reasons.append('digit_heavy')
    if punct >= 3 and letter_ratio < symbol_letter_ratio_threshold:
        reasons.append('symbol_heavy')

    # soft flags: informative but not dropped by default
    if digits > 0:
        flags.append('has_digits')
    if len(scripts) > 1:
        flags.append('mixed_script')
    if ' ' not in s:
        flags.append('single_token')
    if spaces == 0 and s.isalpha() and any(ch.isupper() for ch in s[1:]):
        flags.append('camelcase_handle_style')

    return {
        'DisplayName': raw,
        'DisplayNameClean': s,
        'n_chars': total,
        'n_letters': letters,
        'n_digits': digits,
        'n_spaces': spaces,
        'n_punct': punct,
        'letter_ratio': round(letter_ratio, 4),
        'digit_ratio': round(digit_ratio, 4),
        'script_types': ','.join(scripts),
        'drop_reason': '|'.join(reasons),
        'flags': '|'.join(flags),
        'keep_for_clustering': len(reasons) == 0,
    }


def filter_names(df: pd.DataFrame, text_col: str, args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if text_col not in df.columns:
        raise ValueError(f'Missing column: {text_col}')
    analyzed = pd.DataFrame([
        analyze_name(x,
                     min_chars=args.min_chars,
                     max_chars=args.max_chars,
                     digit_ratio_threshold=args.digit_ratio_threshold,
                     symbol_letter_ratio_threshold=args.symbol_letter_ratio_threshold)
        for x in df[text_col].tolist()
    ])
    kept = analyzed[analyzed['keep_for_clustering']].copy().reset_index(drop=True)
    dropped = analyzed[~analyzed['keep_for_clustering']].copy().reset_index(drop=True)
    return analyzed, kept, dropped


def load_model_and_tokenizer(model_name_or_path: str, device: str | None = None):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    lower_name = model_name_or_path.lower()
    is_decoder_only = ('llama' in lower_name) or ('mistral' in lower_name)

    if is_decoder_only:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype='auto')
    else:
        model = AutoModel.from_pretrained(model_name_or_path, torch_dtype='auto')

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, device, is_decoder_only


def encode_texts(model, tokenizer, texts, model_name_or_path: str, max_length: int = 64, device: str = 'cpu'):
    is_decoder_only = ('llama' in model_name_or_path.lower()) or ('mistral' in model_name_or_path.lower())
    avg_list, cls_list = [], []

    with torch.no_grad():
        for text in tqdm(texts, desc='Encoding names'):
            text = '' if text is None else str(text)
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc.get('attention_mask', torch.ones_like(input_ids)).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]  # [1, L, H]

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
            if device == 'cuda':
                torch.cuda.empty_cache()

    avg_np = torch.cat(avg_list, dim=0).numpy() if avg_list else np.empty((0, 0), dtype=np.float32)
    cls_np = torch.cat(cls_list, dim=0).numpy() if cls_list else np.empty((0, 0), dtype=np.float32)
    return avg_np, cls_np


def prepare_name_features(x_name_cls: np.ndarray) -> np.ndarray:
    X = x_name_cls.astype(np.float32, copy=False)
    X = normalize(X, norm='l2')
    if X.shape[1] > NAME_PCA_DIM:
        ncomp = int(min(NAME_PCA_DIM, X.shape[0], X.shape[1]))
        X = PCA(n_components=ncomp, random_state=RANDOM_STATE).fit_transform(X)
    return X


def build_cluster_summary(names_df: pd.DataFrame, X_features: np.ndarray, X_plot: np.ndarray, labels: np.ndarray, annotation_per_cluster: int = 6) -> pd.DataFrame:
    rows = []
    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        cluster_X = X_features[idx]
        centroid = cluster_X.mean(axis=0, keepdims=True)
        dists = ((cluster_X - centroid) ** 2).sum(axis=1)
        order = np.argsort(dists)
        rep_idx = idx[order[:annotation_per_cluster]]
        rep_names = names_df.iloc[rep_idx]['DisplayNameClean'].tolist()
        rows.append({
            'cluster': int(c),
            'cluster_size': int(len(idx)),
            'representative_names': ' | '.join(rep_names),
        })
    return pd.DataFrame(rows).sort_values('cluster').reset_index(drop=True)


def plot_clusters(names_df: pd.DataFrame, X_plot: np.ndarray, labels: np.ndarray, summary_df: pd.DataFrame, output_png: str, title: str, annotation_per_cluster: int = 6):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, s=18, alpha=0.75)

    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        centroid = X_plot[idx].mean(axis=0)
        ax.scatter([centroid[0]], [centroid[1]], s=110, marker='x')

        cluster_pts = X_plot[idx]
        dists = ((cluster_pts - centroid) ** 2).sum(axis=1)
        order = np.argsort(dists)
        rep_idx = idx[order[:annotation_per_cluster]]
        for j in rep_idx:
            name = names_df.iloc[j]['DisplayNameClean']
            x, y = X_plot[j, 0], X_plot[j, 1]
            txt = ax.annotate(
                name,
                (x, y),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                color='black',
                alpha=0.98,
                zorder=6,
            )
            txt.set_path_effects([
                pe.withStroke(linewidth=2.5, foreground='white')
            ])

    ax.set_title(title)
    ax.set_xlabel('PCA-1')
    ax.set_ylabel('PCA-2')
    plt.tight_layout()
    plt.savefig(output_png, dpi=220, bbox_inches='tight')
    output_pdf = str(Path(output_png).with_suffix('.pdf'))
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    filtered_jsonl_path = outdir / 'english_only_unique_names_filtered.jsonl'
    all_csv_path = outdir / 'english_only_unique_names_filter_audit.csv'
    dropped_jsonl_path = outdir / 'english_only_unique_names_dropped.jsonl'

    if args.mode in {'filter', 'all', 'encode'}:
        if args.input_jsonl is None:
            raise ValueError('--input_jsonl is required for filter/encode/all')
        df = read_jsonl(args.input_jsonl)

    if args.mode in {'filter', 'all'}:
        analyzed, kept, dropped = filter_names(df, args.text_col, args)
        analyzed.to_csv(all_csv_path, index=False)
        write_jsonl(kept[['DisplayName', 'DisplayNameClean', 'flags', 'script_types']], filtered_jsonl_path)
        write_jsonl(dropped[['DisplayName', 'DisplayNameClean', 'drop_reason', 'flags', 'script_types']], dropped_jsonl_path)

        summary = {
            'n_total': int(len(analyzed)),
            'n_kept': int(len(kept)),
            'n_dropped': int(len(dropped)),
            'drop_reason_counts': analyzed.loc[analyzed['drop_reason'] != '', 'drop_reason'].value_counts().to_dict(),
        }
        with open(outdir / 'filter_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.mode in {'encode', 'all'}:
        input_for_encoding = filtered_jsonl_path if filtered_jsonl_path.exists() else Path(args.input_jsonl)
        names_df = read_jsonl(str(input_for_encoding))
        if 'DisplayNameClean' in names_df.columns:
            texts = names_df['DisplayNameClean'].tolist()
        else:
            texts = names_df[args.text_col].astype(str).tolist()
            names_df['DisplayNameClean'] = texts

        if not args.model_name_or_path:
            raise ValueError('--model_name_or_path is required for encode/all')
        tokenizer, model, device, _ = load_model_and_tokenizer(args.model_name_or_path, args.device)
        avg_np, cls_np = encode_texts(model, tokenizer, texts, args.model_name_or_path, max_length=args.max_length, device=device)
        np.savez_compressed(outdir / 'name_embeddings.npz', x_name_avg=avg_np, x_name_cls=cls_np)
        names_df.to_csv(outdir / 'filtered_names_used_for_encoding.csv', index=False)

    if args.mode in {'cluster', 'all'}:
        if args.embeddings_npz is None and not (outdir / 'name_embeddings.npz').exists():
            raise ValueError('Need --embeddings_npz or an existing name_embeddings.npz in output_dir for cluster/all')
        emb_path = Path(args.embeddings_npz) if args.embeddings_npz else (outdir / 'name_embeddings.npz')
        data_path = filtered_jsonl_path if filtered_jsonl_path.exists() else None
        if data_path is None or not data_path.exists():
            raise ValueError('Filtered jsonl was not found in output_dir; rerun with --mode filter or --mode all first.')

        names_df = read_jsonl(str(data_path))
        emb = np.load(emb_path)
        x_name_cls = emb['x_name_cls']
        X = prepare_name_features(x_name_cls)
        labels = KMeans(n_clusters=args.n_clusters, random_state=RANDOM_STATE, n_init=20).fit_predict(X).astype(np.int64)
        X_plot = PCA(n_components=FIG_PCA_DIM, random_state=RANDOM_STATE).fit_transform(X)

        names_df = names_df.copy()
        names_df['cluster'] = labels
        names_df['pca_x'] = X_plot[:, 0]
        names_df['pca_y'] = X_plot[:, 1]
        names_df.to_csv(outdir / 'cluster_assignments.csv', index=False)

        metrics = {
            'n_names': int(len(names_df)),
            'n_clusters': int(args.n_clusters),
            'silhouette_score': float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else None,
            'davies_bouldin_score': float(davies_bouldin_score(X, labels)) if len(np.unique(labels)) > 1 else None,
            'calinski_harabasz_score': float(calinski_harabasz_score(X, labels)) if len(np.unique(labels)) > 1 else None,
        }
        with open(outdir / 'cluster_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        summary_df = build_cluster_summary(names_df, X, X_plot, labels, annotation_per_cluster=args.annotation_per_cluster)
        summary_df.to_csv(outdir / 'cluster_summary.csv', index=False)
        plot_clusters(
            names_df=names_df,
            X_plot=X_plot,
            labels=labels,
            summary_df=summary_df,
            output_png=str(outdir / 'cluster_plot.png'),
            title=f'Name clusters: {Path(args.model_name_or_path).name if args.model_name_or_path else emb_path.parent.name}',
            annotation_per_cluster=args.annotation_per_cluster,
        )


if __name__ == '__main__':
    main()
