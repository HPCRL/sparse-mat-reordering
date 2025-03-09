import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from tabulate import tabulate
import argparse
from constants import MACHINE_NUM_CORES, MACHINE_L1_CACHE, MACHINE_L2_CACHE, MACHINE_L3_CACHE


# Create ArgumentParser object
parser = argparse.ArgumentParser(description="5-fold crossval for predicting best reordering improving more than threshold")

# Add an optional argument --th with default value 1.00
parser.add_argument("--th", type=float, default=1.00, help="Threshold value (default: 1.00)")

# Parse the arguments
args = parser.parse_args()

# ------------------------------------------------------------------------------------
# CONFIG AND DATA LOADING
# ------------------------------------------------------------------------------------
BLK_DENSITY_CSV = "../../csvdata/data_wdensities.csv"
MEASUREMENT_CSV = "../../csvdata/cleaned_all_withI9IOS256_cor.csv"

# PERFORMANCE METRIC & THRESHOLD
PERF_METRIC = "ios"
THRESHOLD = args.th

# Load CSVs
density_df = pd.read_csv(BLK_DENSITY_CSV)
measurement_df = pd.read_csv(MEASUREMENT_CSV)

# Keep only baseline from density data
density_df = density_df[density_df["reordering"] == "baseline"]

# Merge block density features
columns_to_keep = ["matrix_name"] + [f"block_density_{i}" for i in range(1, 11)]
data = measurement_df.merge(density_df[columns_to_keep], on=["matrix_name"], how="left")

# Add machine-specific cache/core features
data["num_cores"] = data["machine"].map(MACHINE_NUM_CORES)
data["l1_cache"]  = data["machine"].map(MACHINE_L1_CACHE)
data["l2_cache"]  = data["machine"].map(MACHINE_L2_CACHE)
data["l3_cache"]  = data["machine"].map(MACHINE_L3_CACHE)

# For convenience
speedup_col = f"median_GFLOPs_{PERF_METRIC}"
baseline_col = f"median_GFLOPs_{PERF_METRIC}_baseline"
data["actual_speedup"] = data[speedup_col] / data[baseline_col]

# ------------------------------------------------------------------------------------
# DETERMINE THE BEST REORDERING OR BASELINE FOR EACH GROUP
# ------------------------------------------------------------------------------------
group_cols = ["matrix_name", "machine", "method", "n", "nth_"]

def find_best_reordering(group_df: pd.DataFrame) -> str:
    """Return the best reordering (highest speedup >= THRESHOLD), or 'baseline' if none qualifies."""
    above_threshold = group_df[group_df["actual_speedup"] >= THRESHOLD]
    if above_threshold.empty:
        return "baseline"
    best_row = above_threshold.loc[above_threshold["actual_speedup"].idxmax()]
    return best_row["reordering"]

best_labels = []
for _, grp in data.groupby(group_cols):
    best = find_best_reordering(grp)
    best_labels.append({
        **{col: grp.iloc[0][col] for col in group_cols},  # group identifiers
        "best_reordering": best
    })
best_label_df = pd.DataFrame(best_labels)

# ------------------------------------------------------------------------------------
# MERGE BEST REORDERING LABELS ONTO BASELINE ROWS
# ------------------------------------------------------------------------------------
baseline_data = data[data["reordering"] == "baseline"].copy()
train_df = pd.merge(
    baseline_data,
    best_label_df,
    on=group_cols,
    how="inner",
)

print(train_df["best_reordering"].value_counts())

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
# MULTI-CLASS MODEL TRAINING (PER METHOD, N)
# ------------------------------------------------------------------------------------
methods = ["SpMV", "SpMM"]
n_values = [8, 64]

results = {}

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
        
        # Encode reordering labels for multi-class
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Prepare cross-validation with GroupKFold to avoid splitting the same matrix
        gkf = GroupKFold(n_splits=5)
        
        model = xgb.XGBClassifier(
            # use_label_encoder=False,
            eval_metric="mlogloss"
        )
        
        accuracy_scores = []
        confusion_matrices = []
        classif_reports = []
        
        # Perform group-based CV
        for train_idx, test_idx in tqdm(gkf.split(X, y_encoded, groups=subset_df["matrix_name"]),
                                        desc=f"Cross-val for {method} (n={n_})"):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            accuracy_scores.append(acc)
            
            conf_m = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))
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
        
        # --------------------------------------------------------------------------------
        # MERGE THE CLASSIFICATION REPORTS (INCL. MACRO AVG, WEIGHTED AVG, SUPPORT)
        # --------------------------------------------------------------------------------
        
        # We'll combine the per-fold 'classification_report' dicts for each label.
        # We'll average precision/recall/f1 *and* average the 'support' across folds.
        
        all_label_names = list(label_encoder.classes_) + ["macro avg", "weighted avg"]
        merged_report = {}
        for lbl in all_label_names:
            merged_report[lbl] = {
                "precision": [],
                "recall": [],
                "f1-score": [],
                "support": []
            }
        
        # We'll track 'accuracy' separately
        accuracies = []
        
        for rpt in classif_reports:
            # For each label (including "macro avg" and "weighted avg"), gather metrics
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
            # Now we AVERAGE 'support' as requested
            final_report[lbl]["support"]   = np.mean(merged_report[lbl]["support"])
        
        # Accuracy: average across folds
        final_report["accuracy"] = np.mean(accuracies) if accuracies else float('nan')
        
        results[(method, n_)] = {
            "accuracy_mean": avg_acc,
            "accuracy_std": std_acc,
            "conf_matrix": total_conf_matrix,
            "classification_report": final_report,
            "classes": label_encoder.classes_.tolist()
        }

# ------------------------------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------------------------------
for (method, n_), res in results.items():
    print("--------------------------------------------------------------------------------")
    print(f"Method: {method}, n: {n_}")
    print(f"Accuracy: {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}")
    
    print("Confusion Matrix (rows=true, cols=pred):")
    print(res["conf_matrix"])
    
    print("\nClassification Report (averaged across folds):")
    cr = res["classification_report"]
    # We'll show each label, then accuracy, then macro avg, then weighted avg
    label_list = res["classes"]
    
    table_data = []
    # 1) Each class
    for lbl in label_list:
        row = [
            lbl,
            f"{cr[lbl]['precision']:.4f}",
            f"{cr[lbl]['recall']:.4f}",
            f"{cr[lbl]['f1-score']:.4f}",
            f"{cr[lbl]['support']:.1f}",   # average support, can keep 1 decimal
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
    
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("--------------------------------------------------------------------------------")
