import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score  # We'll remove standard classification_report usage
from tqdm import tqdm
from tabulate import tabulate
import argparse
from constants import MACHINE_NUM_CORES, MACHINE_L1_CACHE, MACHINE_L2_CACHE, MACHINE_L3_CACHE

# ------------------------------------------------------------------------------------
# PARSE ARGUMENTS
# ------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="5-fold crossval for predicting best reordering that improves more than threshold")
parser.add_argument("--th", type=float, default=1.00, help="Threshold value (default: 1.00)")
args = parser.parse_args()

# ------------------------------------------------------------------------------------
# CONFIG AND DATA LOADING
# ------------------------------------------------------------------------------------
BLK_DENSITY_CSV = "../../csvdata/data_wdensities.csv"
MEASUREMENT_CSV = "../../csvdata/cleaned_all_withI9IOS256_cor.csv"

PERF_METRIC = "ios"
THRESHOLD = args.th  # Speedup threshold

density_df = pd.read_csv(BLK_DENSITY_CSV)
measurement_df = pd.read_csv(MEASUREMENT_CSV)

# Keep only baseline from density data
density_df = density_df[density_df["reordering"] == "baseline"]

# Merge block density features
columns_to_keep = ["matrix_name"] + [f"block_density_{i}" for i in range(1, 11)]
data = measurement_df.merge(density_df[columns_to_keep], on=["matrix_name"], how="left")

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
] + [f"block_density_{i}" for i in range(1, 11)] + [
    "l1_cache",
    "l2_cache",
    "l3_cache"
]

# ------------------------------------------------------------------------------------
# MODEL TRAINING + CUSTOM EVALUATION
# ------------------------------------------------------------------------------------
methods = ["SpMV", "SpMM"]
n_values = [8, 64]

results = {}

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

for method in tqdm(methods, desc="Processing Methods"):
    relevant_n = n_values if method == "SpMM" else [None]
    
    for n_ in relevant_n:
        if n_ is None:
            subset_df = train_df[train_df["method"] == method]
        else:
            subset_df = train_df[(train_df["method"] == method) & (train_df["n"] == n_)]
        
        if subset_df.empty:
            continue
        
        X = subset_df[feature_cols]
        print("Len of whole X: ", len(X))
        y = subset_df["best_reordering"]
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        gkf = GroupKFold(n_splits=5)
        model = xgb.XGBClassifier(eval_metric="mlogloss")
        
        # We'll store the new "correctness" measure for each fold
        fold_accuracies = []
        
        for train_idx, test_idx in tqdm(gkf.split(X, y_encoded, groups=subset_df["matrix_name"]),
                                        desc=f"Cross-val for {method} (n={n_})"):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Convert predicted int labels back to reorder strings
            predicted_reorders = label_encoder.inverse_transform(y_pred)
            
            # Now evaluate correctness under the new rule
            correct_count = 0
            for i, test_i in enumerate(test_idx):
                # 'test_i' is the index into subset_df
                group_info = subset_df.iloc[test_i]  # row with group
                pred_r = predicted_reorders[i]
                
                if is_correct_prediction(pred_r, group_info, data, THRESHOLD):
                    correct_count += 1
            
            accuracy_custom = correct_count / len(test_idx)
            fold_accuracies.append(accuracy_custom)
        
        avg_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        
        # Save results
        results[(method, n_)] = {
            "accuracy_mean": avg_acc,
            "accuracy_std": std_acc,
            "num_samples": len(subset_df)
        }

# ------------------------------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------------------------------
for (method, n_), res in results.items():
    print("--------------------------------------------------------------------------------")
    print(f"Method: {method}, n: {n_}")
    print(f"Custom Accuracy (meets-threshold correctness): {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}")
    print(f"Number of samples in subset: {res['num_samples']}")
    print("--------------------------------------------------------------------------------")
