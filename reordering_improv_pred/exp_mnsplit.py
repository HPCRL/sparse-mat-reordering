import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
from tabulate import tabulate
from constants import MACHINE_NUM_CORES, MACHINE_L1_CACHE, MACHINE_L2_CACHE, MACHINE_L3_CACHE

# Data path
BLK_DENSITY_CSV = "../../csvdata/data_wdensities.csv"
MEASUREMENT_CSV = "../../csvdata/cleaned_all_withI9IOS256.csv"

# Configurable performance metric
PERF_METRIC = "ios"  # Change to "real", "yax", or "xyp" as needed
THRESHOLD = 1.0  # Speedup threshold for classification

# Load datasets
density_df = pd.read_csv(BLK_DENSITY_CSV)
measurement_df = pd.read_csv(MEASUREMENT_CSV)

# Filter density data for baseline version only
density_df = density_df[density_df["reordering"] == "baseline"]

# Merge block density features based on baseline matrix_name
columns_to_keep = ["matrix_name"] + [f"block_density_{i}" for i in range(1, 11)]
data = measurement_df.merge(density_df[columns_to_keep], on=["matrix_name"], how="left")

# Add machine-specific features
data["num_cores"] = data["machine"].map(MACHINE_NUM_CORES)
data["l1_cache"] = data["machine"].map(MACHINE_L1_CACHE)
data["l2_cache"] = data["machine"].map(MACHINE_L2_CACHE)
data["l3_cache"] = data["machine"].map(MACHINE_L3_CACHE)

# Define speedup label
speedup_col = f"median_GFLOPs_{PERF_METRIC}"  # e.g., median_GFLOPs_ios
baseline_col = f"median_GFLOPs_{PERF_METRIC}_baseline"
data["speedup"] = (data[speedup_col] / data[baseline_col]) >= THRESHOLD

# Define experiments
methods = ["SpMV", "SpMM"]
n_values = [8, 64]
reorderings = [r for r in data["reordering"].unique() if r != "baseline"]

# Run experiments
results = {}
for method in tqdm(methods, desc="Processing Methods"):
    for n in ([None] if method == "SpMV" else n_values):
        method_data = data[data["method"] == method]
        if n is not None:
            method_data = method_data[method_data["n"] == n]
        
        for reordering in tqdm(reorderings, desc=f"Processing Reorderings for {method} (n={n})"):
            subset = method_data[method_data["reordering"] == reordering]
            
            if subset.empty:
                continue
            
            X = subset[["nnz", "m", "num_cores", "nth_"] + [f"block_density_{i}" for i in range(1, 11)] +
                      ["l1_cache", "l2_cache", "l3_cache"]]
            y = subset["speedup"].astype(int)
            
            unique_matrices = subset["matrix_name"].unique()
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            model = xgb.XGBClassifier(eval_metric="logloss")
            
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
            conf_matrices = []
            classification_reports = []
            
            for train_matrices, test_matrices in tqdm(skf.split(unique_matrices, np.zeros(len(unique_matrices))), desc=f"Cross-validation for {reordering}"):
                train_matrices = unique_matrices[train_matrices]
                test_matrices = unique_matrices[test_matrices]
                
                train_idx = subset[subset["matrix_name"].isin(train_matrices)].index
                test_idx = subset[subset["matrix_name"].isin(test_matrices)].index
                
                X_train, X_test = X.loc[train_idx], X.loc[test_idx]
                y_train, y_test = y.loc[train_idx], y.loc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy_scores.append(accuracy_score(y_test, y_pred))
                precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
                recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
                f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
                conf_matrices.append(confusion_matrix(y_test, y_pred))
                classification_reports.append(classification_report(y_test, y_pred, output_dict=True))
            
            avg_classification_report = {
                label: {metric: np.mean([report[label][metric] for report in classification_reports])
                        for metric in ["precision", "recall", "f1-score", "support"]}
                for label in ["0", "1", "macro avg", "weighted avg"]
            }
            
            results[(method, n, reordering)] = {
                "accuracy_mean": np.mean(accuracy_scores),
                "accuracy_std": np.std(accuracy_scores),
                "precision_mean": np.mean(precision_scores),
                "recall_mean": np.mean(recall_scores),
                "f1_mean": np.mean(f1_scores),
                "conf_matrix": np.sum(conf_matrices, axis=0),  # Aggregate confusion matrices
                "classification_report": avg_classification_report
            }

# Print results
for (method, n, reordering), res in results.items():
    print(f"Method: {method}, n: {n}, Reordering: {reordering}, "
          f"Accuracy: {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}, "
          f"Precision: {res['precision_mean']:.4f}, Recall: {res['recall_mean']:.4f}, F1-score: {res['f1_mean']:.4f}")
    print("Confusion Matrix:")
    print(res["conf_matrix"])
    print("Classification Report:")
    table_data = [[label] + list(metrics.values()) for label, metrics in res["classification_report"].items()]
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
