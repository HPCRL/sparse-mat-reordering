import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score  # We'll remove standard classification_report usage
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
from tabulate import tabulate
import argparse
import os
import pickle
from constants import MACHINE_NUM_CORES, MACHINE_L1_CACHE, MACHINE_L2_CACHE, MACHINE_L3_CACHE


def cal_print_cls_metrics(sum_tp, sum_tn, sum_fp, sum_fn):
    # Compute precision, recall, and F-score
    precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (sum_tp + sum_tn) / (sum_tp + sum_fp + sum_tn + sum_fn)

    # Print classification report
    print("\Method Classification Report")
    print(tabulate([
        ["True Positives (TP)", sum_tp],
        ["False Positives (FP)", sum_fp],
        ["True Negatives (TN)", sum_tn],
        ["False Negatives (FN)", sum_fn],
        ["Precision", precision],
        ["Recall", recall],
        ["F-score", fscore],
        ["Accuracy", accuracy]
    ], headers=["Metric", "Value"], tablefmt="grid"))


# ------------------------------------------------------------------------------------
# PARSE ARGUMENTS
# ------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="5-fold crossval for predicting best reordering that improves more than threshold")
parser.add_argument("--th", type=float, default=1.00, help="Threshold value (default: 1.00)")
parser.add_argument("--train", action="store_true", help="Set it to train")
args = parser.parse_args()

# ------------------------------------------------------------------------------------
# CONFIG AND DATA LOADING
# ------------------------------------------------------------------------------------
SAVE_DIR = "./saved_models/"
os.makedirs(SAVE_DIR, exist_ok=True)
BLK_DENSITY_CSV = "../../csvdata/data_wdensities.csv"
# MEASUREMENT_CSV = "../../csvdata/cleaned_all_withI9IOS256_cor.csv"
MEASUREMENT_CSV = "../../csvdata/cleaned_all_wI9_w256_cor_complete.csv"

ONE_D_BLK_DENSITY_CSV = "../../csvdata/1d_blk_densities.csv"  # Update with the correct path


PERF_METRIC = "ios"
THRESHOLD = args.th  # Speedup threshold

density_df = pd.read_csv(BLK_DENSITY_CSV)
one_d_density_df = pd.read_csv(ONE_D_BLK_DENSITY_CSV)
measurement_df = pd.read_csv(MEASUREMENT_CSV)
measurement_df = measurement_df[measurement_df["machine"] != "intel-DesktopI9"]

# Keep only baseline from density data
density_df = density_df[density_df["reordering"] == "baseline"]

# Merge block density features
columns_to_keep = ["matrix_name"] + [f"block_density_{i}" for i in range(1, 11)]
data = measurement_df.merge(density_df[columns_to_keep], on=["matrix_name"], how="left")

# Rename 1D block density columns to avoid collision
one_d_density_df = one_d_density_df.rename(
    columns={f"block_density_{i}": f"block_density_1d_{i}" for i in range(1, 11)}
)

# Keep only necessary columns of 1d densities
one_d_density_cols = ["matrix_name"] + [f"block_density_1d_{i}" for i in range(1, 11)]
one_d_density_df = one_d_density_df[one_d_density_cols]

# Merge 1D block density features
data = data.merge(one_d_density_df, on=["matrix_name"], how="left")

# Add machine/core/cache features
data["num_cores"] = data["machine"].map(MACHINE_NUM_CORES)
data["l1_cache"]  = data["machine"].map(MACHINE_L1_CACHE)
data["l2_cache"]  = data["machine"].map(MACHINE_L2_CACHE)
data["l3_cache"]  = data["machine"].map(MACHINE_L3_CACHE)

# Compute actual speedup
speedup_col = f"median_GFLOPs_{PERF_METRIC}"
baseline_col = f"median_GFLOPs_{PERF_METRIC}_baseline"
data["actual_speedup"] = data[speedup_col] / data[baseline_col]

# ------------------------------------------------------------------------------------
# DETERMINE THE "BEST" REORDERING PER GROUP (FOR TRAINING ONLY)
# ------------------------------------------------------------------------------------
group_cols = ["matrix_name", "machine", "method", "n", "nth_"]

def find_best_reordering(group_df: pd.DataFrame) -> str:
    """Return the single reordering with highest speedup >= THRESHOLD or 'baseline' if none qualifies."""
    above_threshold = group_df[group_df["actual_speedup"] >= THRESHOLD]
    if above_threshold.empty:
        return "baseline"
    best_row = above_threshold.loc[above_threshold["actual_speedup"].idxmax()]
    return best_row["reordering"]

best_labels = []
for _, grp in data.groupby(group_cols):
    best = find_best_reordering(grp)
    best_labels.append({
        **{col: grp.iloc[0][col] for col in group_cols},
        "best_reordering": best
    })

best_label_df = pd.DataFrame(best_labels)

# ------------------------------------------------------------------------------------
# MERGE "BEST REORDERING" LABELS ONTO BASELINE ROWS (FOR TRAINING)
# ------------------------------------------------------------------------------------
baseline_data = data[data["reordering"] == "baseline"].copy()
train_df = pd.merge(baseline_data, best_label_df, on=group_cols, how="inner")

print(train_df["best_reordering"].value_counts())

# ------------------------------------------------------------------------------------
# FEATURE COLUMNS
# ------------------------------------------------------------------------------------
feature_cols = [
    "nnz",
    "m",
    "num_cores",
    "nth_",
] + [f"block_density_{i}" for i in range(1, 11)] + [  # Existing 2D densities
    f"block_density_1d_{i}" for i in range(1, 11)  # Adding 1D densities
] + [
    "l1_cache",
    "l2_cache",
    "l3_cache"
]

# ------------------------------------------------------------------------------------
# MODEL TRAINING + CUSTOM EVALUATION
# ------------------------------------------------------------------------------------
methods = ["SpMV", "SpMM"]
n_values = [8, 64, 256]

results = {}

'''
# Helper function: check if a prediction is correct under the new definition
def is_correct_prediction(predicted_reorder: str,
                          group_info: pd.Series,
                          all_data: pd.DataFrame,
                          threshold: float) -> bool:
    """
    We say a prediction is 'correct' if:
      - predicted == 'baseline' AND there is no reordering in real data that meets threshold, OR
      - predicted != 'baseline' AND that reorder in real data meets threshold.
    """
    # Get all rows in the original measurement data for this group
    group_rows = all_data[
        (all_data["matrix_name"] == group_info["matrix_name"]) &
        (all_data["machine"] == group_info["machine"]) &
        (all_data["method"] == group_info["method"]) &
        (all_data["n"] == group_info["n"]) &
        (all_data["nth_"] == group_info["nth_"])
    ]
    
    # If the model predicts "baseline"
    if predicted_reorder == "baseline":
        # Correct if no reorder in the group_rows has speedup >= threshold
        # i.e., best reorder truly is baseline
        has_any_above = any(group_rows[group_rows["reordering"] != "baseline"]["actual_speedup"] >= threshold)
        return (not has_any_above)
    else:
        # The model predicts some reorder R
        row_for_pred = group_rows[group_rows["reordering"] == predicted_reorder]
        if row_for_pred.empty:
            # If for some reason we don't have that reorder's measurement, consider it incorrect
            return False
        # Check if that reorder has speedup >= threshold
        speedup_vals = row_for_pred["actual_speedup"].values
        if len(speedup_vals) == 0:
            return False
        return (speedup_vals[0] >= threshold)
'''

# Store TP, FP, TN, FN counts
all_tp, all_fp, all_tn, all_fn = [], [], [], []

for method in tqdm(methods, desc="Processing Methods"):
    relevant_n = n_values if method == "SpMM" else [None]
    
    for n_ in relevant_n:
        if n_ is None:
            subset_df = train_df[train_df["method"] == method]
        else:
            subset_df = train_df[(train_df["method"] == method) & (train_df["n"] == n_)]
        
        if subset_df.empty:
            continue
        print("len of method subset: ", len(subset_df))
        # continue
        X = subset_df[feature_cols]
        y = subset_df["best_reordering"]
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        gkf = GroupKFold(n_splits=5)
        model = xgb.XGBClassifier(eval_metric="mlogloss")
        
        fold_tp, fold_fp, fold_tn, fold_fn = [], [], [], []
        f_counter = 1
        for train_idx, test_idx in tqdm(gkf.split(X, y_encoded, groups=subset_df["matrix_name"]),
                                        desc=f"Cross-val for {method} (n={n_})"):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            print(f"Fold's train set size: ", len(X_train))
            print(f"Fold's test set size: ", len(X_test))
            # continue
            save_file_path = f"{SAVE_DIR}th{args.th}-{method}-{n_}-fold{f_counter}-preds.pkl"
            if args.train:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                predicted_reorders = label_encoder.inverse_transform(y_pred)
                with open(save_file_path, "wb") as file:
                    pickle.dump(predicted_reorders, file)
            else:
                with open(save_file_path, "rb") as file:
                    predicted_reorders = pickle.load(file)
                # also separate spmv and spmm metric calculation
            f_counter = f_counter + 1
            tp, fp, tn, fn = 0, 0, 0, 0
            
            for i, test_i in enumerate(test_idx):
                group_info = subset_df.iloc[test_i]
                pred_r = predicted_reorders[i]
                
                actual_has_above = any(
                    (data[(data["matrix_name"] == group_info["matrix_name"]) &
                          (data["machine"] == group_info["machine"]) &
                          (data["method"] == group_info["method"]) &
                          (data["n"] == group_info["n"]) &
                          (data["nth_"] == group_info["nth_"]) &
                          (data["reordering"] != "baseline")
                    ]["actual_speedup"] >= THRESHOLD)
                )
                
                pred_speedup = data[(data["matrix_name"] == group_info["matrix_name"]) &
                                     (data["machine"] == group_info["machine"]) &
                                     (data["method"] == group_info["method"]) &
                                     (data["n"] == group_info["n"]) &
                                     (data["nth_"] == group_info["nth_"]) &
                                     (data["reordering"] == pred_r)]
                
                pred_has_above = not pred_speedup.empty and pred_speedup["actual_speedup"].values[0] >= THRESHOLD
                
                # if actual_has_above and pred_has_above: # no need to check actual_has_above (redundant here)
                #     tp += 1
                # elif not actual_has_above and pred_r != "baseline":
                #     fp += 1
                # elif not actual_has_above and pred_r == "baseline":
                #     tn += 1
                # elif actual_has_above and pred_r == "baseline":
                #     fn += 1
                if pred_has_above and pred_r != "baseline": 
                    tp += 1
                elif not pred_has_above and pred_r != "baseline":
                    fp += 1
                elif not actual_has_above and pred_r == "baseline":
                    tn += 1
                elif actual_has_above and pred_r == "baseline":
                    fn += 1
                
            fold_tp.append(tp)
            fold_fp.append(fp)
            fold_tn.append(tn)
            fold_fn.append(fn)
        all_tp.append(sum(fold_tp))
        all_fp.append(sum(fold_fp))
        all_tn.append(sum(fold_tn))
        all_fn.append(sum(fold_fn))
        cal_print_cls_metrics(sum(fold_tp), sum(fold_tn), sum(fold_fp), sum(fold_fn))
        print("------------------------------------------------------------------------")
        
# Compute precision, recall, and F-score
precision = sum(all_tp) / (sum(all_tp) + sum(all_fp)) if (sum(all_tp) + sum(all_fp)) > 0 else 0
recall = sum(all_tp) / (sum(all_tp) + sum(all_fn)) if (sum(all_tp) + sum(all_fn)) > 0 else 0
fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (sum(all_tp) + sum(all_tn)) / (sum(all_tp) + sum(all_fp) + sum(all_tn) + sum(all_fn))

# Print classification report
print("\nAvg-over-all-methods Classification Report")
print(tabulate([
    ["True Positives (TP)", sum(all_tp)],
    ["False Positives (FP)", sum(all_fp)],
    ["True Negatives (TN)", sum(all_tn)],
    ["False Negatives (FN)", sum(all_fn)],
    ["Precision", precision],
    ["Recall", recall],
    ["F-score", fscore],
    ["Accuracy", accuracy]
], headers=["Metric", "Value"], tablefmt="grid"))

