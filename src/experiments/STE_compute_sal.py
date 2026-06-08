import numpy as np
import random
from sklearn.svm import LinearSVC, SVC
import argparse
import json
from sklearn.linear_model import SGDClassifier, SGDRegressor, Perceptron, LogisticRegression
import pickle
from collections import defaultdict, Counter
from typing import List, Dict
import time
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Compute the debiasing matrix of SAL on multilingual SEFiar dataset.")

parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    help="Directory where all persistent data will be stored.",
)

parser.add_argument(
    "--model",
    action="store",
    type=str,
    help="Model to evalute (e.g., SALMultilingualBertForMaskedLM).",
)

parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-multilingual-cased",
    help="HuggingFace model name or path (e.g., bert-base-multilingual-cased). Checkpoint from which a model is instantiated.",
)

parser.add_argument(
    "--bias_type",
    action="store",
    type=str,
    help="The type of bias to mitigate.",
)

parser.add_argument(
    "--dataset_path",
    action="store",
    type=str,
    help="Path to the dataset.",
)

parser.add_argument(
    "--save_path",
    action="store",
    type=str,
    help="Path to the file save SAL debiasing matrix.",
)

parser.add_argument(
    "--Z_attribute",
    action="store",
    type=str,
    help="The type of attribute Z, e.g., binary or textual or random_textual.",
)

parser.add_argument(
    "--num_eigenvectors_to_remove",
    action="store",
    type=int,
    default=5,
    help="Number of eigenvectors to remove from the global SVD U matrix.",
)


def generate_experiment_id(
    name,
    model=None,
    model_name_or_path=None,
    bias_type=None,
    seed=None,
    pre_assignment_type=None,
):
    experiment_id = f"{name}"

    # Build the experiment ID.
    if isinstance(model, str):
        experiment_id += f"_m-{model}"
    if isinstance(model_name_or_path, str):
        experiment_id += f"_c-{model_name_or_path}"
    if isinstance(bias_type, str):
        experiment_id += f"_i-{bias_type}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"
    if isinstance(pre_assignment_type, str):
        experiment_id += f"_p-{pre_assignment_type}"

    return experiment_id



if __name__=="__main__":
    args = parser.parse_args()
    
    experiment_id = generate_experiment_id(
        name="compute-multilingualSEFiar",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
    )

    print("Running multilingualBiasBios:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - dataset_path: {args.dataset_path}")
    print(f" - save_path: {args.save_path}")
    print(f" - experiment_id {experiment_id}")
    print(f" - num_eigenvectors_to_remove: {args.num_eigenvectors_to_remove}")
    print(f" - Z_attribute: {args.Z_attribute}")

    random.seed(0)
    np.random.seed(0)
    data = np.load(os.path.join(args.dataset_path))
    X = data['x_train_text_cls']

    if args.Z_attribute == "binary":
        if args.bias_type == "reputation":
            Y = data['y_train_reputation']
        elif args.bias_type == "location":
            Y = data['y_train_location']
        elif args.bias_type == "engagement":
            Y = data['y_train_engagement']
        elif args.bias_type == "name":
            Y = data['x_train_name_cls']
        elif args.bias_type == "gender":
            Y = data['y_train_gender']
        else:
            assert False, "Invalid bias_type for binary Z_attribute"
        
        if args.bias_type != "name":
            Y = np.reshape(Y, (-1, 1))

    elif args.Z_attribute == "random_textual":
        Y = data['x_train_aboutme_cls']
        rng = np.random.default_rng(0)
        rng.shuffle(Y)
    elif args.Z_attribute == "textual":
        Y = data['x_train_aboutme_cls']
    else:
        assert False, "Invalid Z_attribute"

    print(f"x_train_aboutme_cls and Y shape is {X.shape}, {Y.shape}; Number of samples is {len(X)}\n")

    if args.Z_attribute == "binary":
        if args.bias_type == "reputation" or args.bias_type == "engagement" or args.bias_type == "gender":
            Y_2d = np.hstack((Y, 1 - Y))
        elif args.bias_type == "location":
            Y = Y.astype(int).ravel()   # guaranteed 1..5
            Y_2d = np.eye(5, dtype=int)[Y - 1]                 # (n, 5) int one-hot
            print(f"the first 5 rows of Y_2d are {Y_2d[:5,:]}\n, the first five rows of Y are {Y[:5]}\n")
        elif args.bias_type == "name":
            Y_2d = Y
        A = np.dot(X.T, Y_2d) / X.shape[0]
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        print(f"Compute SAL, X shape is {X.shape}, Y_2d shape is {Y_2d.shape}.\n")
    elif args.Z_attribute in ["textual", "random_textual"]:
        # Y_2d = np.hstack((Y, 1 - Y))
        A = np.dot(X.T, Y) / X.shape[0]
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        print(f"Compute SAL, X shape is {X.shape}, Y shape is {Y.shape}.\n")
    else:
        assert False, "Invalid Z_attribute"
    


    removal = args.num_eigenvectors_to_remove
    rank_A = np.linalg.matrix_rank(A)
    print(f"Rank of A: {rank_A}")
    print(f"A shape {A.shape}.\n\n A is {A}.\n\n u shape {u.shape}.\n\n u is {u}\n\n")
    u_r = u[:, removal:]
    proj = u_r @ u_r.T
    # End of computing SAL

    print(f"u_r is {u_r}.\n\n proj is {proj}.\n\n\n\n\n\n\n\n\n\n")
    np.set_printoptions(threshold=np.inf)
    print(f"s (all values): {s}.\n\n\n\n")
    
    os.makedirs(args.save_path, exist_ok=True)
    # np.savez(os.path.join(args.save_path, args.source_language_list), proj_accumulated = proj)
    np.savez(os.path.join(args.save_path, f"{args.bias_type}.npz"), proj_accumulated = proj)



