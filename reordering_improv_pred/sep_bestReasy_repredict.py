import pandas as pd
import numpy as np
import xgboost as xgb
import os

SAVE_DIR = "./saved_models/"
methods = ["SpMV", "SpMM"]
n_values = [8, 64, 256]

for method in methods:
    relevant_n = n_values if method == "SpMM" else [None]

    for n_ in relevant_n:
        model_path = f"{SAVE_DIR}/xgb_model_{method}_{n_}.json"
        data_path = f"{SAVE_DIR}/predictions_{method}_{n_}.csv"
        label_path = f"{SAVE_DIR}/label_mapping_{method}_{n_}.npy"

        if not os.path.exists(model_path) or not os.path.exists(data_path):
            print(f"Skipping {method} (n={n_}) as model or predictions are missing.")
            continue

        # Load saved model
        model = xgb.XGBClassifier()
        model.load_model(model_path)

        # Load test data
        test_df = pd.read_csv(data_path)

        # Load label encoder
        label_encoder = np.load(label_path, allow_pickle=True)

        # Extract features
        feature_cols = [
            "nnz", "m", "num_cores", "nth_"
        ] + [f"block_density_{i}" for i in range(1, 11)] + [f"block_density_1d_{i}" for i in range(1, 11)] + [
            "l1_cache", "l2_cache", "l3_cache"
        ]

        X_test = test_df[feature_cols]

        # Predict
        y_pred = model.predict(X_test)
        predicted_labels = label_encoder[y_pred]

        # Save new predictions
        test_df["new_predicted_reordering"] = predicted_labels
        test_df.to_csv(f"{SAVE_DIR}/new_predictions_{method}_{n_}.csv", index=False)

        print(f"Predictions updated for {method} (n={n_}).")
