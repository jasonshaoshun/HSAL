import argparse
import os
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

BIAS_TO_KEY = {
    "location": "location",
    "engagement": "engagement",
    "reputation": "reputation",
    "name": "name_cluster",
    "gender": "gender",
}

SPLIT_ALIASES = {
    "train": ["train"],
    "validation": ["validation", "valid", "val", "dev"],
    "test": ["test"],
}

DATASET_LATEX = {
    "stackoverflow": r"\logoso StackOverflow",
    "mathematics": r"\logoms Mathematics",
    "superuser": r"\logosu SuperUser",
    "english": r"\logoes English",
}

BIAS_LATEX = {
    "reputation": "Reputation",
    "gender": "Gender",
    "location": "Location",
    "engagement": "Engagement",
}


def parse_csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def find_z_key(data, split_name, bias_type):
    suffix = BIAS_TO_KEY[bias_type]
    tried = []
    for alias in SPLIT_ALIASES[split_name]:
        candidates = [
            f"y_{alias}_{suffix}",
            f"z_{alias}_{suffix}",
            f"{alias}_{suffix}",
            f"y_{alias}_{bias_type}",
            f"z_{alias}_{bias_type}",
            f"{alias}_{bias_type}",
        ]
        for key in candidates:
            tried.append(key)
            if key in data.files:
                return key
    return None


def count_array(arr, drop_negative=False):
    arr = np.asarray(arr).astype(int)
    if drop_negative:
        arr = arr[arr >= 0]
    c = Counter(arr.tolist())
    return {int(k): int(v) for k, v in sorted(c.items())}


def format_counts(counts, class_orders=None):
    if counts is None:
        return "--"
    if class_orders is None:
        keys = sorted(counts.keys())
    else:
        keys = class_orders
    return " / ".join(str(int(counts.get(k, 0))) for k in keys)


def load_counts(npz_path, bias_type, drop_negative=False):
    data = np.load(npz_path)
    out = {}
    used_keys = {}
    for split_name in ["train", "validation", "test"]:
        key = find_z_key(data, split_name, bias_type)
        if key is None:
            out[split_name] = None
            used_keys[split_name] = None
        else:
            out[split_name] = count_array(data[key], drop_negative=drop_negative)
            used_keys[split_name] = key
    return out, used_keys, list(data.files)


def choose_path(data_root, experiment, dataset, bias_type, model_name_or_path, split_file):
    return Path(data_root) / experiment / dataset / bias_type / model_name_or_path / f"{split_file}.npz"


def merge_class_orders(stats, bias_types):
    orders = {}
    for b in bias_types:
        labels = set()
        for ds_stats in stats.values():
            if b not in ds_stats:
                continue
            for split_counts in ds_stats[b].values():
                if split_counts is not None:
                    labels.update(split_counts.keys())
        orders[b] = sorted(labels)
    return orders


def make_latex_table(stats, datasets, bias_types, caption, label, class_orders):
    col_spec = "ll" + "c" * len(bias_types)
    header = "Dataset & Split & " + " & ".join(BIAS_LATEX.get(b, b.title()) for b in bias_types) + r" \\\n"
    lines = []
    lines.append(r"\begin{table*}[th!]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{0.8\textwidth}{!}{")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(header.rstrip())
    lines.append(r"\midrule")
    for di, dataset in enumerate(datasets):
        ds_name = DATASET_LATEX.get(dataset, dataset)
        for si, split_name in enumerate(["train", "validation", "test"]):
            split_display = {"train": "Train", "validation": "Validation", "test": "Test"}[split_name]
            cells = []
            for b in bias_types:
                counts = stats.get(dataset, {}).get(b, {}).get(split_name)
                cells.append(format_counts(counts, class_orders.get(b)))
            if si == 0:
                prefix = rf"\multirow{{3}}{{*}}{{{ds_name}}} & {split_display}"
            else:
                prefix = rf"& {split_display}"
            lines.append(prefix + " & " + " & ".join(cells) + r" \\")
        if di != len(datasets) - 1:
            lines.append("")
            lines.append(r"\addlinespace[3pt]")
            lines.append(r"\midrule")
            lines.append("")
    lines.append("")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Print protected-attribute distribution and produce LaTeX stats table.")
    p.add_argument("--data_root", default="data")
    p.add_argument("--experiment", default="STE")
    p.add_argument("--datasets", default="stackoverflow,mathematics,superuser,english")
    p.add_argument("--bias_types", default="reputation,gender,location,engagement,name")
    p.add_argument("--model_name_or_paths", required=True, help="Comma-separated model_name_or_path values to check, e.g. bert-base-uncased,meta-llama/Llama-3.1-8B")
    p.add_argument("--split_file", default="natural_disjoint", help="NPZ split filename without .npz")
    p.add_argument("--drop_negative", action="store_true", help="Drop labels <0, useful if -1 means unknown/missing.")
    p.add_argument("--output_dir", default="tables/data_distribution")
    p.add_argument("--caption", default=r"\textbf{Number of users in each split by protected-attribute class under the natural disjoint split.} For binary attributes, entries are reported as class counts in the form $Z=0$ / $Z=1$. For location, entries are reported as counts for each country class.")
    p.add_argument("--label", default="tab:user-counts-natural-disjoint")
    args = p.parse_args()

    datasets = parse_csv(args.datasets)
    bias_types = parse_csv(args.bias_types)
    model_names = parse_csv(args.model_name_or_paths)
    os.makedirs(args.output_dir, exist_ok=True)

    stats = defaultdict(dict)
    provenance = defaultdict(dict)
    consistency_warnings = []

    for dataset in datasets:
        print(f"\n================ {dataset} ================")
        for bias_type in bias_types:
            found = []
            for model_name in model_names:
                npz_path = choose_path(args.data_root, args.experiment, dataset, bias_type, model_name, args.split_file)
                if not npz_path.exists():
                    continue
                counts, used_keys, all_keys = load_counts(npz_path, bias_type, drop_negative=args.drop_negative)
                found.append((model_name, str(npz_path), counts, used_keys))

            if not found:
                print(f"[MISSING] {dataset} / {bias_type}: no .npz found for provided models")
                stats[dataset][bias_type] = {"train": None, "validation": None, "test": None}
                provenance[dataset][bias_type] = None
                continue

            ref_model, ref_path, ref_counts, ref_keys = found[0]
            stats[dataset][bias_type] = ref_counts
            provenance[dataset][bias_type] = {"model_name_or_path": ref_model, "npz_path": ref_path, "keys": ref_keys}

            for other_model, other_path, other_counts, other_keys in found[1:]:
                if other_counts != ref_counts:
                    consistency_warnings.append({
                        "dataset": dataset,
                        "bias_type": bias_type,
                        "reference_model": ref_model,
                        "different_model": other_model,
                        "reference_path": ref_path,
                        "different_path": other_path,
                    })

            print(f"[{bias_type}] source={ref_path}")
            for split_name in ["train", "validation", "test"]:
                print(f"  {split_name:<10} key={str(ref_keys[split_name]):<28} counts={ref_counts[split_name]}")

    class_orders = merge_class_orders(stats, bias_types)
    latex = make_latex_table(stats, datasets, bias_types, args.caption, args.label, class_orders)

    tex_path = Path(args.output_dir) / f"{args.experiment}_{args.split_file}_user_counts.tex"
    json_path = Path(args.output_dir) / f"{args.experiment}_{args.split_file}_user_counts.json"
    prov_path = Path(args.output_dir) / f"{args.experiment}_{args.split_file}_user_counts_provenance.json"

    with open(tex_path, "w") as f:
        f.write(latex + "\n")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    with open(prov_path, "w") as f:
        json.dump({"provenance": provenance, "consistency_warnings": consistency_warnings}, f, indent=2)

    print("\n================ LaTeX table ================")
    print(latex)
    print(f"\nSaved LaTeX table to: {tex_path}")
    print(f"Saved raw counts to:   {json_path}")
    print(f"Saved provenance to:   {prov_path}")

    if consistency_warnings:
        print("\n[WARNING] Some model-specific npz files have different Z distributions.")
        print("See provenance JSON for details.")


if __name__ == "__main__":
    main()
