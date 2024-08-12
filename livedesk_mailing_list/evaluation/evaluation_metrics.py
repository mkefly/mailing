from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score, roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from typing import Dict

def evaluate_model(
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: np.ndarray,
        k: int = 10,
        threshold: float = None
    ) -> Dict[str, float]:

    def calculate_metrics(likelihood_scores, y_true):
        top_k_indices = np.argsort(likelihood_scores)[-k:]
        precision_at_k = np.mean(y_true[top_k_indices])

        sorted_indices = np.argsort(-likelihood_scores)
        sorted_labels = y_true[sorted_indices]
        average_precision_at_k = np.mean([
            precision_score(sorted_labels[:i+1], np.ones(i+1))
            for i in range(k) if sorted_labels[i]
        ])

        ndcg_at_k = ndcg_score([y_true], [likelihood_scores], k=k)
        auc_roc = roc_auc_score(y_true, likelihood_scores)
        avg_precision_score = average_precision_score(y_true, likelihood_scores)

        return precision_at_k, average_precision_at_k, ndcg_at_k, auc_roc, avg_precision_score

    def calculate_threshold_metrics(likelihood_scores, y_true, threshold):
        predicted_labels = (likelihood_scores >= threshold).astype(int)
        precision = precision_score(y_true, predicted_labels)
        recall = recall_score(y_true, predicted_labels)
        f1 = f1_score(y_true, predicted_labels)
        return precision, recall, f1

    likelihood_scores = pipeline.predict_proba(X)[:, 1]

    if not threshold:
        thresholds = np.linspace(0.1, 0.9, 9)
        best_threshold = 0.0
        best_f1_score = 0.0
        for th in thresholds:
            _, _, f1 = calculate_threshold_metrics(likelihood_scores, y, th)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = th
    else:
        best_threshold = threshold

    metrics = calculate_metrics(likelihood_scores, y)
    precision, recall, f1 = calculate_threshold_metrics(likelihood_scores, y, best_threshold)

    return {
        'precision_at_k': metrics[0],
        'map_at_k': metrics[1],
        'ndcg_at_k': metrics[2],
        'auc_roc': metrics[3],
        'average_precision_score': metrics[4],
        'best_threshold': best_threshold,
        'precision_at_best_threshold': precision,
        'recall_at_best_threshold': recall,
        'f1_score_at_best_threshold': f1,
    }
