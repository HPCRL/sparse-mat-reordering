import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from tabulate import tabulate
import argparse

from constants import (
    MACHINE_NUM_CORES, MACHINE_L1_CACHE, MACHINE_L2_CACHE, MACHINE_L3_CACHE
)

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="5-fold crossval for three-class classification (resistant/conducive/none)")

# Add an optional argument --th with default value 1.00
parser.add_argument("--th", type=float, default=1.00, help="Threshold value (default: 1.00)")

# Parse the arguments
args = parser.parse_args()

# ------------------------------------------------------------------------------------
# CONFIG AND DATA LOADING
# ------------------------------------------------------------------------------------
BLK_DENSITY_CSV = "../../csvdata/data_wdensities.csv"
ONE_D_BLK_DENSITY_CSV = "../../csvdata/1d_blk_densities.csv"
# MEASUREMENT_CSV = "../../csvdata/cleaned_all_withI9IOS256_cor.csv"
MEASUREMENT_CSV = "../../csvdata/cleaned_all_wI9_w256_cor_complete.csv"

# PERFORMANCE METRIC & THRESHOLD
PERF_METRIC = "ios"
THRESHOLD = args.th

# Load CSVs
density_df = pd.read_csv(BLK_DENSITY_CSV)
one_d_density_df = pd.read_csv(ONE_D_BLK_DENSITY_CSV)
measurement_df = pd.read_csv(MEASUREMENT_CSV)

# Keep only baseline from density data (used later to merge in 2D block densities)
density_df = density_df[density_df["reordering"] == "baseline"]

# Merge 2D block density features
columns_to_keep = ["matrix_name"] + [f"block_density_{i}" for i in range(1, 11)]
data = measurement_df.merge(density_df[columns_to_keep], on=["matrix_name"], how="left")

# Rename 1D block density columns to avoid collision
one_d_density_df = one_d_density_df.rename(
    columns={f"block_density_{i}": f"block_density_1d_{i}" for i in range(1, 11)}
)

# Keep only relevant columns for 1D
one_d_density_cols = ["matrix_name"] + [f"block_density_1d_{i}" for i in range(1, 11)]
one_d_density_df = one_d_density_df[one_d_density_cols]

# Merge 1D block densities
data = data.merge(one_d_density_df, on=["matrix_name"], how="left")

# Calculate speedup vs baseline
speedup_col = f"median_GFLOPs_{PERF_METRIC}"
baseline_col = f"median_GFLOPs_{PERF_METRIC}_baseline"
data["actual_speedup"] = data[speedup_col] / data[baseline_col]

# ------------------------------------------------------------------------------------
# DETERMINE CLASS: "resistant", "conducive", OR "none"
# ------------------------------------------------------------------------------------
def determine_class_label(group_df: pd.DataFrame, threshold: float) -> str:
    """
    Given all rows for a single (matrix_name, method, n, nth_) across every machine 
    and reordering, label as:
      - 'resistant': speedup < threshold on *every* machine's best reordering
      - 'conducive': on the majority of machines, there's *some* reordering >= threshold
      - 'none': otherwise
    """
    machines = group_df["machine"].unique()
    n_machines = len(machines)
    
    # Max speedup per machine (across reorderings)
    max_speedup_by_machine = group_df.groupby("machine")["actual_speedup"].max()
    
    # If *all* machines have max_speedup < threshold => resistant
    if (max_speedup_by_machine < threshold).all():
        return "resistant"
    
    # Count how many machines can reach >= threshold
    n_good_machines = (max_speedup_by_machine >= threshold).sum()
    
    # If majority of machines can hit >= threshold => conducive
    if n_good_machines >= (n_machines / 2.0):
        return "conducive"
    
    # Otherwise => none
    return "none"

# We now group by (matrix_name, method, n, nth_)
change to nth instead of nth_
group_cols_for_labeling = ["matrix_name", "method", "n", "nth_"]

class_labels = []
for keys, grp in data.groupby(group_cols_for_labeling):
    # keys is (matrix_name, method, n, nth_)
    matrix_name, method, n_val, nth_val = keys
    label = determine_class_label(grp, THRESHOLD)
    class_labels.append({
        "matrix_name": matrix_name,
        "method": method,
        "n": n_val,
        "nth_": nth_val,
        "class_label": label
    })

class_label_df = pd.DataFrame(class_labels)
print("class label df:\n", class_label_df)
# exit()

# Add prints to show overall counts of resistant/conducive
resistant_count = (class_label_df["class_label"] == "resistant").sum()
conducive_count = (class_label_df["class_label"] == "conducive").sum()
none_count = (class_label_df["class_label"] == "none").sum()

print("Count labeled resistant (overall):", resistant_count)
print("Count labeled conducive (overall):", conducive_count)
print("Count labeled none (overall):", none_count)
# exit()
# ------------------------------------------------------------------------------------
# CREATE ONE BASELINE ROW PER (matrix_name, method, n, nth_)
# ------------------------------------------------------------------------------------
baseline_data = data[data["reordering"] == "baseline"].copy()

# Group by the same 4 columns, pick the first row
baseline_data_agg = baseline_data.groupby(
    ["matrix_name", "method", "n", "nth_"], 
    as_index=False
).first()

# Merge in the 3-class labels
train_df = pd.merge(
    baseline_data_agg,
    class_label_df,
    on=["matrix_name", "method", "n", "nth_"],
    how="inner",
)

# For matrix-level features, exclude machine-dependent columns 
# (since we have one row per matrix or scenario). 
# We'll keep typical matrix properties + block densities + block densities 1D.
feature_cols = [
    "m",
    # "n",   # yes, we can keep 'n' as a dimension if you like
    "nnz",
] + [f"block_density_{i}" for i in range(1, 11)] \
  + [f"block_density_1d_{i}" for i in range(1, 11)]

# Optional: If you do NOT want 'nth_' as part of the features, you can leave it out here.
# If you do want it as a feature, you could do:
# feature_cols.append("nth_")

print("Final training set shape:", train_df.shape)
print("Class distribution (overall):")
print(train_df["class_label"].value_counts())

# ------------------------------------------------------------------------------------
# MULTI-CLASS MODEL TRAINING (PER METHOD, N, nth_)
# ------------------------------------------------------------------------------------
methods = ["SpMV", "SpMM"]
n_values = [8, 64]  # For SpMV, we typically do n=None or n=1. If your code uses something else, adjust accordingly.

# Let's extract the unique nth_ values from the data
all_nth_values = sorted(train_df["nth_"].unique())

results = {}

for method in tqdm(methods, desc="Processing Methods"):
    # For SpMM, we look at n_values = [8, 64], for SpMV perhaps n_values = [None], 
    # but let's see if your data actually uses None or 1. Adjust as needed:
    if method == "SpMV":
        relevant_n = [None]  # or possibly [1] if your data has n=1 for SpMV
    else:
        relevant_n = n_values
    
    for n_ in relevant_n:
        # if your data doesn't literally have "n = None", you might skip that or handle it carefully
        # e.g. subset_df = train_df[ (train_df["method"]==method) & (train_df["n"].isna()) ] for None
        for nth_val in all_nth_values:
            if n_ is None:
                subset_df = train_df[
                    (train_df["method"] == method) & (train_df["n"].isna()) & (train_df["nth_"] == nth_val)
                ]
            else:
                subset_df = train_df[
                    (train_df["method"] == method) & (train_df["n"] == n_) & (train_df["nth_"] == nth_val)
                ]
            
            if subset_df.empty:
                continue
            
            # X,y
            X = subset_df[feature_cols]
            y = subset_df["class_label"]
            
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Use StratifiedKFold for balancing classes in each fold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            model = xgb.XGBClassifier(eval_metric="mlogloss")
            
            accuracy_scores = []
            confusion_matrices = []
            classif_reports = []
            
            for train_idx, test_idx in skf.split(X, y_encoded):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                accuracy_scores.append(acc)
                
                conf_m = confusion_matrix(y_test, y_pred, 
                                          labels=range(len(label_encoder.classes_)))
                confusion_matrices.append(conf_m)
                
                report_dict = classification_report(
                    y_test,
                    y_pred,
                    target_names=label_encoder.classes_,
                    output_dict=True,
                    zero_division=0,
                )
                classif_reports.append(report_dict)
            
            # Aggregate accuracy
            avg_acc = np.mean(accuracy_scores)
            std_acc = np.std(accuracy_scores)
            
            # Sum confusion matrices across folds
            total_conf_matrix = np.sum(confusion_matrices, axis=0)
            
            # Merge the classification reports 
            all_label_names = list(label_encoder.classes_) + ["macro avg", "weighted avg"]
            merged_report = {
                lbl: {"precision": [], "recall": [], "f1-score": [], "support": []}
                for lbl in all_label_names
            }
            accuracies = []
            
            for rpt in classif_reports:
                for lbl in all_label_names:
                    if lbl in rpt:
                        merged_report[lbl]["precision"].append(rpt[lbl].get("precision", 0.0))
                        merged_report[lbl]["recall"].append(rpt[lbl].get("recall", 0.0))
                        merged_report[lbl]["f1-score"].append(rpt[lbl].get("f1-score", 0.0))
                        merged_report[lbl]["support"].append(rpt[lbl].get("support", 0.0))
                if "accuracy" in rpt:
                    accuracies.append(rpt["accuracy"])
            
            final_report = {}
            for lbl in all_label_names:
                final_report[lbl] = {}
                final_report[lbl]["precision"] = np.mean(merged_report[lbl]["precision"])
                final_report[lbl]["recall"]    = np.mean(merged_report[lbl]["recall"])
                final_report[lbl]["f1-score"]  = np.mean(merged_report[lbl]["f1-score"])
                final_report[lbl]["support"]   = np.mean(merged_report[lbl]["support"])
            
            # Accuracy: average across folds
            final_report["accuracy"] = np.mean(accuracies) if accuracies else float('nan')
            
            # Store results
            results[(method, n_, nth_val)] = {
                "accuracy_mean": avg_acc,
                "accuracy_std": std_acc,
                "conf_matrix": total_conf_matrix,
                "classification_report": final_report,
                "classes": label_encoder.classes_.tolist(),
                "num_samples": len(subset_df),
            }

# ------------------------------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------------------------------
for (method, n_, nth_val), res in results.items():
    print("--------------------------------------------------------------------------------")
    print(f"Method: {method}, n: {n_}, nth_: {nth_val}")
    print(f"Num Samples: {res['num_samples']}")
    print(f"Accuracy: {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}")
    
    print("Confusion Matrix (rows=true, cols=pred):")
    print(res["conf_matrix"])
    
    print("\nClassification Report (averaged across folds):")
    cr = res["classification_report"]
    label_list = res["classes"]
    
    table_data = []
    # 1) Each class
    for lbl in label_list:
        row = [
            lbl,
            f"{cr[lbl]['precision']:.4f}",
            f"{cr[lbl]['recall']:.4f}",
            f"{cr[lbl]['f1-score']:.4f}",
            f"{cr[lbl]['support']:.1f}",   # average support
        ]
        table_data.append(row)
    
    # 2) Accuracy row
    table_data.append([
        "accuracy",
        "",  # precision
        "",  # recall
        f"{cr['accuracy']:.4f}",
        "",  # support
    ])
    
    # 3) macro avg
    mac = cr["macro avg"]
    table_data.append([
        "macro avg",
        f"{mac['precision']:.4f}",
        f"{mac['recall']:.4f}",
        f"{mac['f1-score']:.4f}",
        f"{mac['support']:.1f}",
    ])
    
    # 4) weighted avg
    wgt = cr["weighted avg"]
    table_data.append([
        "weighted avg",
        f"{wgt['precision']:.4f}",
        f"{wgt['recall']:.4f}",
        f"{wgt['f1-score']:.4f}",
        f"{wgt['support']:.1f}",
    ])
    
    from tabulate import tabulate
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("--------------------------------------------------------------------------------")
