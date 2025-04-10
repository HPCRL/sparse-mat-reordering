import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import argparse

# Paths for saving
SAVE_DIR = "./saved_models" +  + "/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load your dataset
BLK_DENSITY_CSV = "../../csvdata/data_wdensities.csv"
ONE_D_BLK_DENSITY_CSV = "../../csvdata/1d_blk_densities.csv"
MEASUREMENT_CSV = "../../csvdata/cleaned_all_wI9_w256_cor_complete.csv"

density_df = pd.read_csv(BLK_DENSITY_CSV)
one_d_density_df = pd.read_csv(ONE_D_BLK_DENSITY_CSV)
measurement_df = pd.read_csv(MEASUREMENT_CSV)
measurement_df = measurement_df[measurement_df["machine"] != "intel-DesktopI9"]

# Keep only baseline from density data
density_df = density_df[density_df["reordering"] == "baseline"]
columns_to_keep = ["matrix_name"] + [f"block_density_{i}" for i in range(1, 11)]
data = measurement_df.merge(density_df[columns_to_keep], on=["matrix_name"], how="left")

# Merge 1D block density features
one_d_density_df = one_d_density_df.rename(
    columns={f"block_density_{i}": f"block_density_1d_{i}" for i in range(1, 11)}
)
data = data.merge(one_d_density_df, on=["matrix_name"], how="left")

# Compute actual speedup
PERF_METRIC = "ios"
data["actual_speedup"] = data[f"median_GFLOPs_{PERF_METRIC}"] / data[f"median_GFLOPs_{PERF_METRIC}_baseline"]

# Define feature columns
feature_cols = [
    "nnz", "m", "num_cores", "nth_"
] + [f"block_density_{i}" for i in range(1, 11)] + [f"block_density_1d_{i}" for i in range(1, 11)] + [
    "l1_cache", "l2_cache", "l3_cache"
]

# Cross-validation
methods = ["SpMV", "SpMM"]
n_values = [8, 64, 256]
gkf = GroupKFold(n_splits=5)

for method in tqdm(methods, desc="Processing Methods"):
    relevant_n = n_values if method == "SpMM" else [None]
    
    for n_ in relevant_n:
        if n_ is None:
            subset_df = data[data["method"] == method]
        else:
            subset_df = data[(data["method"] == method) & (data["n"] == n_)]
        
        if subset_df.empty:
            continue

        X = subset_df[feature_cols]
        y = subset_df["best_reordering"]
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Save label encoder mapping
        np.save(f"{SAVE_DIR}/label_mapping_{method}_{n_}.npy", label_encoder.classes_)

        model = xgb.XGBClassifier(eval_metric="mlogloss")

        predictions_list = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y_encoded, groups=subset_df["matrix_name"])):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predicted_reorders = label_encoder.inverse_transform(y_pred)

            # Save test fold and predictions
            fold_predictions = subset_df.iloc[test_idx].copy()
            fold_predictions["predicted_reordering"] = predicted_reorders
            predictions_list.append(fold_predictions)

        # Save full test results
        predictions_df = pd.concat(predictions_list)
        predictions_df.to_csv(f"{SAVE_DIR}/predictions_{method}_{n_}.csv", index=False)

        # Save trained model
        model.save_model(f"{SAVE_DIR}/xgb_model_{method}_{n_}.json")
