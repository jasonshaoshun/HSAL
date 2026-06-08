"""
Plot textual-Z singular-value decay directly from a STE dataset NPZ.

Example:
python src/experiments/plot_singular_values.py \
  --model LlamaModel \
  --dataset mathematics \
  --bias_type engagement \
  --out_dir figures/singular_values/mathematics_llama \
  --logy

The script finds the matching NPZ under data/STE/<dataset>/<bias_type>/<model>/...,
then computes the SAL textual-Z matrix from STE_compute_sal.py:
    A = x_train_text_cls.T @ x_train_aboutme_cls / n
and plots the singular values of A.

If --bias_type is omitted, the script writes one plot each for:
gender, location, reputation, engagement, and name.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


MODEL_TO_MODEL_NAME_OR_PATH = {
    "BertModel": "bert-base-uncased",
    "LlamaModel": "meta-llama/Llama-3.1-8B",
    "MistralModel": "mistralai/Mistral-7B-v0.3",
    "bert": "bert-base-uncased",
    "llama": "meta-llama/Llama-3.1-8B",
    "mistral": "mistralai/Mistral-7B-v0.3",
    "Bert-uncased": "bert-base-uncased",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Mistral-7B-v0.3": "mistralai/Mistral-7B-v0.3",
}

TEXTUAL_KEY_PAIRS = [
    ("x_train_text_cls", "x_train_aboutme_cls"),
    ("x_text_cls", "x_aboutme_cls"),
]

DEFAULT_BIAS_TYPES = ["gender", "location", "reputation", "engagement", "name"]

SPLIT_PRIORITY = {
    "natural_disjoint": 0,
    "disjoint": 1,
    "joint": 2,
    "raw": 3,
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values_txt", type=str, default=None,
                    help="Optional txt file containing precomputed singular values.")
    ap.add_argument("--npz_path", "--dataset_path", dest="npz_path", type=str, default=None,
                    help="Optional explicit STE dataset NPZ path. If omitted, it is found from --model, --dataset, and optional --bias_type.")
    ap.add_argument("--model", type=str, default=None,
                    help="Model key or model path, e.g. BertModel, LlamaModel, MistralModel, bert-base-uncased.")
    ap.add_argument("--model_name_or_path", type=str, default=None,
                    help="HuggingFace model path. Overrides the model-name inferred from --model.")
    ap.add_argument("--dataset", type=str, default=None,
                    help="STE dataset name, e.g. mathematics, stackoverflow, english, superuser.")
    ap.add_argument("--bias_type", type=str, default=None,
                    help="Bias subset used to locate split NPZ files. If omitted, all five default bias types are plotted.")
    ap.add_argument("--split", type=str, default="natural_disjoint",
                    help="Split filename to prefer when locating NPZ files.")
    ap.add_argument("--experiment", type=str, default="STE",
                    help="Experiment directory under the data root.")
    ap.add_argument("--persistent_dir", type=str, default=".",
                    help="Repository/persistent directory containing the data directory.")
    ap.add_argument("--data_root", type=str, default=None,
                    help="Optional explicit data root. Defaults to <persistent_dir>/data.")
    ap.add_argument("--out_dir", type=str, default="figures/singular_values",
                    help="Directory to save the figure.")
    ap.add_argument("--output_prefix", type=str, default=None,
                    help="Optional filename prefix. Defaults to the bias type, e.g. reputation.png.")
    ap.add_argument("--logy", action="store_true",
                    help="Use log scale on y-axis.")
    args = ap.parse_args()

    if args.values_txt is None and args.npz_path is None:
        missing = []
        if args.model is None and args.model_name_or_path is None:
            missing.append("--model")
        if args.dataset is None:
            missing.append("--dataset")
        if missing:
            ap.error(
                "Either pass --values_txt, pass --npz_path/--dataset_path, "
                f"or provide {' and '.join(missing)} to locate the NPZ automatically."
            )

    return args


def load_values(path: str) -> np.ndarray:
    import numpy as np

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not tokens:
        raise ValueError(f"No numeric values found in {path}")

    values = np.array([float(x) for x in tokens], dtype=float)
    values = np.maximum(values, 0.0)
    values = np.sort(values)[::-1]
    return values


def resolve_model_name_or_path(model: str | None, model_name_or_path: str | None) -> str:
    if model_name_or_path:
        return model_name_or_path
    if model is None:
        raise ValueError("--model is required when --model_name_or_path is not provided")
    return MODEL_TO_MODEL_NAME_OR_PATH.get(model, model)


def parts_contain_sequence(parts: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    if not sequence:
        return True
    if len(sequence) > len(parts):
        return False
    for start in range(len(parts) - len(sequence) + 1):
        if parts[start:start + len(sequence)] == sequence:
            return True
    return False


def candidate_data_roots(args) -> list[Path]:
    roots = []
    if args.data_root:
        roots.append(Path(args.data_root))
    else:
        roots.append(Path(args.persistent_dir) / "data")

    default_root = Path("data")
    if default_root not in roots:
        roots.append(default_root)

    return roots


def available_textual_keys(npz_path: Path) -> tuple[str, str] | None:
    import numpy as np

    with np.load(npz_path, allow_pickle=True) as data:
        keys = set(data.files)

    for x_key, y_key in TEXTUAL_KEY_PAIRS:
        if x_key in keys and y_key in keys:
            return x_key, y_key
    return None


def path_priority(path: Path, key_pair: tuple[str, str], args) -> tuple:
    split_name = path.stem
    split_rank = SPLIT_PRIORITY.get(split_name, 50)
    if split_name == args.split:
        split_rank = -1

    parts = path.parts
    bias_rank = 0 if args.bias_type is None or args.bias_type in parts else 100
    key_rank = TEXTUAL_KEY_PAIRS.index(key_pair)
    return key_rank, split_rank, bias_rank, len(parts), str(path)


def find_dataset_npz(args) -> tuple[Path, tuple[str, str], str]:
    if args.npz_path:
        npz_path = Path(args.npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        key_pair = available_textual_keys(npz_path)
        if key_pair is None:
            raise KeyError(
                f"{npz_path} does not contain textual-Z keys. "
                f"Expected one of {TEXTUAL_KEY_PAIRS}."
            )
        model_name_or_path = args.model_name_or_path or args.model or "<explicit-npz>"
        return npz_path, key_pair, model_name_or_path

    model_name_or_path = resolve_model_name_or_path(args.model, args.model_name_or_path)
    model_parts = Path(model_name_or_path).parts
    matches = []

    for data_root in candidate_data_roots(args):
        dataset_root = data_root / args.experiment / args.dataset
        if not dataset_root.exists():
            continue

        for npz_path in dataset_root.rglob("*.npz"):
            if not parts_contain_sequence(npz_path.parts, model_parts):
                continue
            if args.bias_type is not None and args.bias_type not in npz_path.parts:
                continue

            key_pair = available_textual_keys(npz_path)
            if key_pair is not None:
                matches.append((path_priority(npz_path, key_pair, args), npz_path, key_pair))

    if not matches:
        roots = ", ".join(str(root / args.experiment / args.dataset) for root in candidate_data_roots(args))
        bias_msg = f" Bias type: {args.bias_type}." if args.bias_type else ""
        raise FileNotFoundError(
            "Could not find a matching textual-Z NPZ. "
            f"Looked under: {roots}. "
            f"Model path: {model_name_or_path}. "
            f"{bias_msg} "
            f"Expected keys: {TEXTUAL_KEY_PAIRS}."
        )

    matches.sort(key=lambda item: item[0])
    if len(matches) > 1:
        print("Multiple matching NPZ files found; using the highest-priority match:")
        for _, path, _ in matches[:5]:
            print(f"  - {path}")
        if len(matches) > 5:
            print(f"  ... and {len(matches) - 5} more")

    _, npz_path, key_pair = matches[0]
    return npz_path, key_pair, model_name_or_path


def ensure_2d(name: str, array: np.ndarray) -> np.ndarray:
    import numpy as np

    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape}")
    return array


def compute_textual_z_singular_values(npz_path: Path, x_key: str, y_key: str) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    import numpy as np

    with np.load(npz_path, allow_pickle=True) as data:
        x = ensure_2d(x_key, data[x_key])
        y = ensure_2d(y_key, data[y_key])

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"{x_key} and {y_key} must have the same number of rows, "
            f"got {x.shape[0]} and {y.shape[0]}"
        )

    a = np.dot(x.T, y) / x.shape[0]
    s = np.linalg.svd(a, full_matrices=False, compute_uv=False)
    s = np.maximum(s, 0.0)
    return s, x.shape, y.shape


def plot_values(values: np.ndarray, out_dir: str, output_prefix: str, logy: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(1, len(values) + 1)

    plt.figure(figsize=(7, 4.8))
    plt.plot(x, values, linewidth=2)

    plt.xlabel("Singular value order", fontsize=24)
    plt.ylabel("Singular value", fontsize=24)

    if logy:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{output_prefix}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"{output_prefix}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def bias_types_to_plot(args) -> list[str | None]:
    if args.values_txt or args.npz_path:
        return [args.bias_type]
    if args.bias_type:
        return [args.bias_type]
    return DEFAULT_BIAS_TYPES


def args_for_bias(args, bias_type: str | None):
    values = vars(args).copy()
    values["bias_type"] = bias_type
    return argparse.Namespace(**values)


def output_prefix_for(args, bias_type: str | None, num_outputs: int) -> str:
    if args.output_prefix is None:
        return bias_type or "singular_values_decay"
    if bias_type and "{bias_type}" in args.output_prefix:
        return args.output_prefix.format(bias_type=bias_type)
    if bias_type and num_outputs > 1:
        return f"{args.output_prefix}_{bias_type}"
    return args.output_prefix


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    bias_types = bias_types_to_plot(args)

    import numpy as np

    for bias_type in bias_types:
        run_args = args_for_bias(args, bias_type)
        output_prefix = output_prefix_for(args, bias_type, len(bias_types))

        if args.values_txt:
            s = load_values(args.values_txt)
            print(f"Loaded {len(s)} singular values from {args.values_txt}")
        else:
            npz_path, (x_key, y_key), model_name_or_path = find_dataset_npz(run_args)
            s, x_shape, y_shape = compute_textual_z_singular_values(npz_path, x_key, y_key)
            print(f"Loaded NPZ: {npz_path}")
            print(f"Bias type: {bias_type}")
            print(f"Model path: {model_name_or_path}")
            print(f"Using textual Z: {y_key}")
            print(f"X key/shape: {x_key} {x_shape}")
            print(f"Y key/shape: {y_key} {y_shape}")
            print(f"Computed {len(s)} singular values")

        np.savetxt(os.path.join(args.out_dir, f"{output_prefix}.txt"), s)
        plot_values(s, args.out_dir, output_prefix, args.logy)
        print(f"Saved {output_prefix}.pdf/.png/.txt to {args.out_dir}")


if __name__ == "__main__":
    main()
