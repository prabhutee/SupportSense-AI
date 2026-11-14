"""
Evaluation and Metrics Utility for SupportSense.

Computes:
- Overall Accuracy / Top-1 Accuracy
- Top-3 Accuracy
- Macro-averaged F1 Score
- Weighted-averaged F1 Score
- Confusion Matrix summary (top confused intent pairs)
- Inference Latency (milliseconds per sample and throughput)
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def compute_top_k_accuracy(y_true_indices: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    """
    Computes top-k categorical accuracy: whether the true class index
    is within the top-k highest predicted probability indices.
    """
    if y_proba.shape[1] < k:
        k = y_proba.shape[1]
    
    # Get top k indices for each sample (sorted descending)
    top_k_indices = np.argsort(y_proba, axis=1)[:, -k:]
    # Check if true label is in top k
    matches = [y_true_indices[i] in top_k_indices[i] for i in range(len(y_true_indices))]
    return float(np.mean(matches))


def measure_isolated_inference_latency(
    model: Any,
    X_sample: Any,
    num_runs: int = 5,
    batch_size: int = 64
) -> Dict[str, float]:
    """
    Measures isolated matrix-inference latency (pre-vectorized feature input).
    Does NOT include text cleaning or TF-IDF transformation.
    """
    # 1. Single sample latency
    single_sample = X_sample[0:1]
    # Warmup
    for _ in range(3):
        _ = model.predict(single_sample)

    latencies_single = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model.predict(single_sample)
        latencies_single.append((time.perf_counter() - t0) * 1000)  # ms

    # 2. Batch throughput
    n_samples = X_sample.shape[0] if hasattr(X_sample, "shape") else len(X_sample)
    actual_batch_size = min(batch_size, n_samples)
    batch = X_sample[:actual_batch_size]

    # Warmup batch
    for _ in range(2):
        _ = model.predict(batch)

    batch_latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model.predict(batch)
        batch_latencies.append(time.perf_counter() - t0)

    avg_batch_time = np.mean(batch_latencies)
    throughput = actual_batch_size / avg_batch_time if avg_batch_time > 0 else 0.0

    return {
        "benchmark_type": "isolated_model_inference (pre-vectorized sparse CSR matrix)",
        "mean_latency_ms": round(float(np.mean(latencies_single)), 3),
        "median_latency_ms": round(float(np.median(latencies_single)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies_single, 95)), 3),
        "batch_size": actual_batch_size,
        "batch_throughput_samples_per_sec": round(float(throughput), 1)
    }


def measure_end_to_end_latency(
    raw_texts: List[str],
    cleaner_func: Any,
    vectorizer: Any,
    model: Any,
    num_runs: int = 5,
    batch_size: int = 64
) -> Dict[str, float]:
    """
    Measures realistic end-to-end inference latency:
    Raw Text -> Text Preprocessing -> TF-IDF Transformation -> Production Prediction (predict_proba).
    """
    sample_text = raw_texts[0]

    def _predict_single(text: str):
        cleaned = cleaner_func(text)
        vec = vectorizer.transform([cleaned])
        if hasattr(model, "predict_proba"):
            return model.predict_proba(vec)
        return model.predict(vec)

    # Warmup single
    for _ in range(3):
        _ = _predict_single(sample_text)

    latencies_single = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = _predict_single(sample_text)
        latencies_single.append((time.perf_counter() - t0) * 1000)

    # Batch end-to-end
    actual_batch_size = min(batch_size, len(raw_texts))
    batch_texts = raw_texts[:actual_batch_size]

    def _predict_batch(texts: List[str]):
        cleaned_list = [cleaner_func(t) for t in texts]
        vec = vectorizer.transform(cleaned_list)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(vec)
        return model.predict(vec)

    # Warmup batch
    for _ in range(2):
        _ = _predict_batch(batch_texts)

    batch_latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = _predict_batch(batch_texts)
        batch_latencies.append(time.perf_counter() - t0)

    avg_batch_time = np.mean(batch_latencies)
    throughput = actual_batch_size / avg_batch_time if avg_batch_time > 0 else 0.0

    return {
        "benchmark_type": "end_to_end_pipeline (raw text -> preprocessor -> TF-IDF -> predict_proba)",
        "mean_latency_ms": round(float(np.mean(latencies_single)), 3),
        "median_latency_ms": round(float(np.median(latencies_single)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies_single, 95)), 3),
        "batch_size": actual_batch_size,
        "batch_throughput_samples_per_sec": round(float(throughput), 1)
    }


def find_top_confused_pairs(
    y_true: List[str],
    y_pred: List[str],
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Extracts the top-N most frequent misclassification pairs.
    """
    misclassifications: Dict[Tuple[str, str], int] = {}
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label != pred_label:
            pair = (true_label, pred_label)
            misclassifications[pair] = misclassifications.get(pair, 0) + 1

    sorted_pairs = sorted(misclassifications.items(), key=lambda x: x[1], reverse=True)
    return [
        {"true_intent": pair[0], "predicted_intent": pair[1], "count": count}
        for pair, count in sorted_pairs[:top_n]
    ]


def evaluate_model_comprehensive(
    model: Any,
    X_features: Any,
    y_true_labels: List[str],
    label_encoder: Any,
    model_name: str,
    raw_texts: Optional[List[str]] = None,
    cleaner_func: Any = None,
    vectorizer: Any = None
) -> Dict[str, Any]:
    """
    Performs full evaluation of a model and returns structured metrics.
    Measures both isolated model inference latency and end-to-end pipeline latency.
    """
    y_true_indices = label_encoder.transform(y_true_labels)
    
    # Predictions
    y_pred_indices = model.predict(X_features)
    y_pred_labels = label_encoder.inverse_transform(y_pred_indices)
    
    # Probabilities for Top-K
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_features)
        top_1_acc = compute_top_k_accuracy(y_true_indices, y_proba, k=1)
        top_3_acc = compute_top_k_accuracy(y_true_indices, y_proba, k=3)
    else:
        top_1_acc = float(accuracy_score(y_true_indices, y_pred_indices))
        top_3_acc = top_1_acc

    acc = float(accuracy_score(y_true_indices, y_pred_indices))
    macro_f1 = float(f1_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0))
    
    # Isolated matrix inference latency
    isolated_latency = measure_isolated_inference_latency(model, X_features)

    # Realistic end-to-end latency (text -> cleaner -> TF-IDF -> predict_proba)
    end_to_end_latency = None
    if raw_texts is not None and cleaner_func is not None and vectorizer is not None:
        end_to_end_latency = measure_end_to_end_latency(
            raw_texts=raw_texts,
            cleaner_func=cleaner_func,
            vectorizer=vectorizer,
            model=model
        )

    # Top confused pairs
    confused_pairs = find_top_confused_pairs(y_true_labels, y_pred_labels, top_n=5)

    return {
        "model_name": model_name,
        "sample_count": len(y_true_labels),
        "accuracy": round(acc, 4),
        "top_1_accuracy": round(top_1_acc, 4),
        "top_3_accuracy": round(top_3_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "latency": isolated_latency,  # Backward compatibility
        "isolated_inference_latency": isolated_latency,
        "end_to_end_latency": end_to_end_latency,
        "top_confused_pairs": confused_pairs
    }
