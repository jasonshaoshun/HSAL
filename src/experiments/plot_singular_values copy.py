import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values_txt", type=str, required=True,
                    help="Path to txt file containing singular values.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Directory to save the figure.")
    ap.add_argument("--logy", action="store_true",
                    help="Use log scale on y-axis.")
    return ap.parse_args()


def load_values(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not tokens:
        raise ValueError(f"No numeric values found in {path}")

    values = np.array([float(x) for x in tokens], dtype=float)
    values = np.maximum(values, 0.0)
    values = np.sort(values)[::-1]
    return values


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    s = load_values(args.values_txt)
    x = np.arange(1, len(s) + 1)

    plt.figure(figsize=(7, 4.8))
    plt.plot(x, s, linewidth=2)

    plt.xlabel("Singular value order", fontsize=24)
    plt.ylabel("Singular value", fontsize=24)

    if args.logy:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "singular_values_decay.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(args.out_dir, "singular_values_decay.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loaded {len(s)} singular values")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()

