import numpy as np
import random
import pickle
from collections import defaultdict, Counter
from typing import List, Dict
import argparse
from sklearn.linear_model import LogisticRegression
import json
import os
import sys

# Add current directory to path for imports
current_dir = os.getcwd()
sys.path.append(current_dir)
from src.util import generate_experiment_id

parser = argparse.ArgumentParser(description="Evaluate multilingual SEFiar.")
parser.add_argument("--persistent_dir", action="store", type=str)
parser.add_argument("--experiment", action="store", type=str)
parser.add_argument("--model", action="store", type=str)
parser.add_argument("--model_name_or_path", action="store", type=str, default="bert-base-multilingual-cased")
parser.add_argument("--X_attribute", action="store", choices=["input", "self-description"], type=str, default="input")
parser.add_argument("--debiasing", action="store", type=str)
parser.add_argument("--projection_matrix", action="store", type=str)
parser.add_argument("--bias_type", action="store", type=str)
parser.add_argument("--npz_path", action="store", type=str)
parser.add_argument("--results_save_path", action="store", type=str)
parser.add_argument("--num_eigenvectors_to_remove", action="store", type=int, default=1)
parser.add_argument("--num_eigenvectors_to_remove_from_global_U", action="store", type=int)

def train_clf(x_train, y_train, x_test, y_test):
    """
    Trains a Logistic Regression classifier and returns train/test accuracy.
    """
    random.seed(0)
    np.random.seed(0)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=0)
    
    clf.fit(x_train, y_train)
    score_test = float(clf.score(x_test, y_test))
    score_train = float(clf.score(x_train, y_train))

    return clf, score_train, score_test

def get_tpr_gap(y_pred, y_true, z_true):
    """
    Computes RMS Gap across ALL main task classes (Y).
    
    Metric: 2 * Standard Deviation
    ------------------------------
    1. Binary Case: 2 * StdDev is mathematically identical to (Max - Min).
       This ensures it matches your 'Elder Code' results exactly.
    2. Multiclass Case: StdDev captures changes in ALL groups (2nd, 3rd, etc.),
       not just the best and worst.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    z_true = z_true.astype(int)

    # Identify all classes in the main task (e.g., 0 and 1)
    unique_y = np.unique(y_true)
    # Identify all groups in the sensitive attribute
    unique_z = np.unique(z_true)
    
    # Store the gap for each class Y
    gaps_per_class = []

    for y_c in unique_y:
        # We want to calculate P(Pred=y_c | True=y_c, Z=g) for every group g
        tprs_for_this_class = []
        
        for g in unique_z:
            # Denominator: Count of (True Label == y_c AND Group == g)
            mask_true = (y_true == y_c) & (z_true == g)
            denom = mask_true.sum()
            
            # If a group is missing from this class (denom=0), skip it
            if denom == 0:
                continue
            
            # Numerator: Count of (Predicted == y_c AND True Label == y_c AND Group == g)
            mask_pred = (y_pred[mask_true] == y_c)
            tpr = mask_pred.sum() / denom
            tprs_for_this_class.append(tpr)
        
        # Calculate the disparity for this specific class y_c
        if tprs_for_this_class:
            # === CHANGED HERE ===
            # We use 2 * StdDev.
            # In Binary: 2 * StdDev([A, B]) == |A - B| (Exact match to Elder Code)
            # In Multiclass: Captures variance of ALL groups, including middle ones.
            gap = 2 * np.std(tprs_for_this_class)
            gaps_per_class.append(gap)

    if not gaps_per_class:
        return 0.0

    # RMS (Root Mean Square) of the gaps across all classes
    # Matches 'rms_diff' from Elder Code
    rms_gap = np.sqrt(np.mean(np.array(gaps_per_class)**2))
    
    return rms_gap

if __name__ == "__main__":
    args = parser.parse_args()

    # Generate experiment ID for saving
    experiment_id = generate_experiment_id(
        name=args.experiment,
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        debiasing_method=args.debiasing,
        bias_type=args.bias_type
    )

    print(f"Running Eval on: {args.npz_path}")
    data = np.load(args.npz_path)

    # Load Features
    if args.X_attribute == "input":
        x_train = data['x_train_text_cls']
        x_test = data['x_test_text_cls']
    elif args.X_attribute == "self-description":
        x_train = data['x_train_aboutme_cls']
        x_test = data['x_test_aboutme_cls']
    else:
        raise ValueError(f"Unknown X_attribute: {args.X_attribute}")
    
    # Load Main Task Labels (Helpfulness)
    y_train_help = data['y_train_helpfulness'].astype(int)
    y_test_help = data['y_test_helpfulness'].astype(int)

    # --- DYNAMIC Z LOADING ---
    if args.bias_type == "location":
        print("Loading Z as LOCATION")
        z_train = data['y_train_location'].astype(int)
        z_test = data['y_test_location'].astype(int)
    elif args.bias_type == "engagement":
        print("Loading Z as ENGAGEMENT")
        z_train = data['y_train_engagement'].astype(int)
        z_test = data['y_test_engagement'].astype(int)
    elif args.bias_type == "reputation":
        print("Loading Z as REPUTATION")
        z_train = data['y_train_reputation'].astype(int)
        z_test = data['y_test_reputation'].astype(int)
    elif args.bias_type == "name":
        print("Loading Z as NAME")
        z_train = data['y_train_name_cluster'].astype(int)
        z_test = data['y_test_name_cluster'].astype(int)
    elif args.bias_type == "gender":
        print("Loading Z as NAME")
        z_train = data['y_train_gender'].astype(int)
        z_test = data['y_test_gender'].astype(int)
    else:
        raise ValueError(f"Unknown bias_type: {args.bias_type}")

    # ========================== BIASED EVALUATION ==========================
    # 1. Train Main Task Classifier
    clf_help, _, help_acc = train_clf(x_train, y_train_help, x_test, y_test_help)
    y_pred_help = clf_help.predict(x_test)
    
    # 2. Calculate Gap (RMS of 2*StdDev)
    tpr_gap_before = get_tpr_gap(y_pred_help, y_test_help, z_test)
    
    # 3. Train Bias Predictor
    _, _, z_acc = train_clf(x_train, z_train, x_test, z_test)

    print(f"BIASED -> HelpAcc: {help_acc:.8f}, BiasPredAcc: {z_acc:.8f}, Gap(RMS): {tpr_gap_before:.8f}")

    # ========================== DEBIASED EVALUATION ==========================
    help_acc_after = np.nan
    z_acc_after = np.nan
    tpr_gap_after = np.nan
    
    if args.projection_matrix and os.path.exists(args.projection_matrix):
        print(f"Loading Projection: {args.projection_matrix}")
        P = np.load(args.projection_matrix)['proj_accumulated']
        
        # Project features
        x_train_deb = P.dot(x_train.T).T
        x_test_deb = P.dot(x_test.T).T

        # 1. Train Main Task Classifier
        clf_help_af, _, help_acc_after = train_clf(x_train_deb, y_train_help, x_test_deb, y_test_help)
        y_pred_help_af = clf_help_af.predict(x_test_deb)
        
        # 2. Calculate Gap
        tpr_gap_after = get_tpr_gap(y_pred_help_af, y_test_help, z_test)
        
        # 3. Train Bias Predictor
        _, _, z_acc_after = train_clf(x_train_deb, z_train, x_test_deb, z_test)
        
        print(f"DEBIASED -> HelpAcc: {help_acc_after:.8f}, BiasPredAcc: {z_acc_after:.8f}, Gap(RMS): {tpr_gap_after:.8f}")

    # ========================== SAVE RESULTS ==========================
    result = [{
        "experiment_id": experiment_id,
        "experiment": args.experiment,
        "debiasing": args.debiasing,
        "model": args.model,
        "main_task_accuracy_before": help_acc,
        "main_task_accuracy_after": help_acc_after,
        "tpr_gap_before": tpr_gap_before,
        "tpr_gap_after": tpr_gap_after,
        "bias_prediction_accuracy_before": z_acc,
        "bias_prediction_accuracy_after": z_acc_after,
        "bias_type": args.bias_type,
        "num_eigenvectors_removed": args.num_eigenvectors_to_remove,
    }]
    
    os.makedirs(f"{args.results_save_path}", exist_ok=True)
    with open(f"{args.results_save_path}/{experiment_id}.json", "w") as f:
        json.dump(result, f, indent=4)