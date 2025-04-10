import pandas as pd
import numpy as np
from tabulate import tabulate

SAVE_DIR = "./saved_models/"
methods = ["SpMV", "SpMM"]
n_values = [8, 64, 256]

for method in methods:
    relevant_n = n_values if method == "SpMM" else [None]

    for n_ in relevant_n:
        file_path = f"{SAVE_DIR}/predictions_{method}_{n_}.csv"
        if not os.path.exists(file_path):
            print(f"Skipping {method} (n={n_}) as no saved predictions found.")
            continue

        df = pd.read_csv(file_path)

        # Compute TP, FP, TN, FN
        tp, fp, tn, fn = 0, 0, 0, 0

        for _, row in df.iterrows():
            actual_has_above = row["actual_speedup"] >= 1.0
            pred_has_above = row["predicted_reordering"] != "baseline"

            if pred_has_above and actual_has_above:
                tp += 1
            elif pred_has_above and not actual_has_above:
                fp += 1
            elif not pred_has_above and not actual_has_above:
                tn += 1
            elif not pred_has_above and actual_has_above:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)

        print(f"\nMetrics for {method} (n={n_}):")
        print(tabulate([
            ["True Positives (TP)", tp],
            ["False Positives (FP)", fp],
            ["True Negatives (TN)", tn],
            ["False Negatives (FN)", fn],
            ["Precision", precision],
            ["Recall", recall],
            ["F-score", fscore],
            ["Accuracy", accuracy]
        ], headers=["Metric", "Value"], tablefmt="grid"))
