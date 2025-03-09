import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from tabulate import tabulate
from constants import MACHINE_NUM_CORES, MACHINE_L1_CACHE, MACHINE_L2_CACHE, MACHINE_L3_CACHE

BLK_DENSITY_CSV = "../../csvdata/data_wdensities.csv"
MEASUREMENT_CSV = "../../csvdata/cleaned_all_withI9IOS256.csv"

PERF_METRIC = "ios"
THRESHOLD = 1.0

density_df = pd.read_csv(BLK_DENSITY_CSV)
measurement_df = pd.read_csv(MEASUREMENT_CSV)

density_df = density_df[density_df["reordering"] == "baseline"]
columns_to_keep = ["matrix_name"] + [f"block_density_{i}" for i in range(1, 11)]
data = measurement_df.merge(density_df[columns_to_keep], on=["matrix_name"], how="left")
data["num_cores"] = data["machine"].map(MACHINE_NUM_CORES)
data["l1_cache"] = data["machine"].map(MACHINE_L1_CACHE)
data["l2_cache"] = data["machine"].map(MACHINE_L2_CACHE)
data["l3_cache"] = data["machine"].map(MACHINE_L3_CACHE)
data["speedup"] = data[f"median_GFLOPs_{PERF_METRIC}"] / data[f"median_GFLOPs_{PERF_METRIC}_baseline"]

def get_best_reordering(group):
    valid = group[group["speedup"] > THRESHOLD]
    if valid.empty:
        return "baseline"
    return valid.loc[valid["speedup"].idxmax(), "reordering"]

data["best_reordering"] = data.groupby("matrix_name", group_keys=False).apply(get_best_reordering)
data["best_reordering"] = data["best_reordering"].fillna("baseline").astype("category")
data["best_reordering"] = data["best_reordering"].cat.codes

methods = ["SpMV", "SpMM"]
n_values = [8, 64]

results = {}

def safe_get_metric(rep, label, metric):
    if label not in rep:
        return np.nan
    val = rep[label]
    if not isinstance(val, dict):
        return np.nan
    return val.get(metric, np.nan)

for method in tqdm(methods, desc="Processing Methods"):
    for n in ([None] if method == "SpMV" else n_values):
        method_data = data[data["method"] == method]
        if n is not None:
            method_data = method_data[method_data["n"] == n]

        X = method_data[[
            "nnz", "m", "num_cores", "nth_"
        ] + [f"block_density_{i}" for i in range(1, 11)] + [
            "l1_cache", "l2_cache", "l3_cache"
        ]]
        y = method_data["best_reordering"]

        unique_matrices = method_data["matrix_name"].unique()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = xgb.XGBClassifier(eval_metric="mlogloss")
        accuracy_scores = []
        classification_reports = []

        for train_matrices_ids, test_matrices_ids in tqdm(
            skf.split(unique_matrices, np.zeros(len(unique_matrices))),
            desc=f"Cross-validation for {method} (n={n})"
        ):
            train_matrices = unique_matrices[train_matrices_ids]
            test_matrices = unique_matrices[test_matrices_ids]
            train_idx = method_data[method_data["matrix_name"].isin(train_matrices)].index
            test_idx = method_data[method_data["matrix_name"].isin(test_matrices)].index
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            classification_reports.append(classification_report(y_test, y_pred, output_dict=True))

        all_labels = set()
        for rep in classification_reports:
            all_labels.update(rep.keys())

        avg_classification_report = {}
        for label in all_labels:
            label_metrics = {}
            for metric in ["precision", "recall", "f1-score", "support"]:
                vals = [safe_get_metric(rep, label, metric) for rep in classification_reports]
                vals = [v for v in vals if not pd.isna(v)]
                label_metrics[metric] = np.mean(vals) if len(vals) > 0 else np.nan
            avg_classification_report[label] = label_metrics

        results[(method, n)] = {
            "accuracy_mean": np.mean(accuracy_scores),
            "accuracy_std": np.std(accuracy_scores),
            "classification_report": avg_classification_report
        }

for (method, n), res in results.items():
    print(f"Method: {method}, n: {n}, Accuracy: {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}")
    print("Classification Report:")
    table_data = [
        [label] + list(metrics.values())
        for label, metrics in res["classification_report"].items()
    ]
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
