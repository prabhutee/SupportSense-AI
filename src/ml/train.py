"""
SupportSense Model Training, Tuning, and Benchmarking Pipeline.

Executes:
1. Strict data split (official train 10,003 / test 3,080, with stratified 80/20 train/validation split).
2. Empirical comparison: basic vs extended preprocessing (lemmatization & stopword filtering).
3. Hyperparameter tuning of Logistic Regression and LinearSVC using validation split only.
4. Final model fitting on full training data (TF-IDF fitted exclusively on train to prevent leakage).
5. Probability calibration of LinearSVC (Platt Scaling) for confidence scoring.
6. Unbiased final benchmarking on official test set (Accuracy, Top-1, Top-3, Macro F1, Weighted F1, Latency, Confused Pairs).
7. Serialization of all models and metrics to models/.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple

# Ensure project root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import sklearn

from src.ml.preprocessor import clean_text_basic, clean_text_extended
from src.ml.evaluate import evaluate_model_comprehensive

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAIN_PATH = os.path.join(DATA_RAW_DIR, "banking77_train.csv")
TEST_PATH = os.path.join(DATA_RAW_DIR, "banking77_test.csv")
CATEGORIES_PATH = os.path.join(DATA_RAW_DIR, "categories.json")

RANDOM_SEED = 42


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """Loads raw official BANKING77 datasets and categories."""
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        categories = json.load(f)
    return df_train, df_test, categories


def compare_preprocessing(
    X_train_sub: pd.Series,
    y_train_sub: np.ndarray,
    X_val_sub: pd.Series,
    y_val_sub: np.ndarray
) -> Tuple[str, Dict[str, Any]]:
    """
    Empirically benchmarks Basic vs Extended Preprocessing on validation split.
    Uses identical baseline Logistic Regression (C=1.0) and TF-IDF parameters.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: EMPIRICAL PREPROCESSING COMPARISON (ON VALIDATION SET)")
    print("=" * 60)

    results = {}
    for mode in ["basic", "extended"]:
        cleaner = clean_text_basic if mode == "basic" else clean_text_extended
        print(f"Applying '{mode}' preprocessing to train & validation splits...")
        
        X_tr_clean = X_train_sub.apply(cleaner)
        X_val_clean = X_val_sub.apply(cleaner)

        # Fit TF-IDF ONLY on training split
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
        X_tr_tfidf = vec.fit_transform(X_tr_clean)
        X_val_tfidf = vec.transform(X_val_clean)

        clf = LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)
        clf.fit(X_tr_tfidf, y_train_sub)
        val_acc = clf.score(X_val_tfidf, y_val_sub)

        results[mode] = {
            "validation_accuracy": round(float(val_acc), 4),
            "vocab_size": len(vec.vocabulary_)
        }
        print(f"  -> [{mode.upper()}] Val Accuracy: {val_acc:.4f} (Vocab size: {len(vec.vocabulary_)})")

    # Select mode with higher validation accuracy (or simpler 'basic' if tied)
    best_mode = "basic" if results["basic"]["validation_accuracy"] >= results["extended"]["validation_accuracy"] else "extended"
    print(f"\n[DECISION] Selected '{best_mode}' preprocessing based on empirical validation.")
    return best_mode, results


def tune_hyperparameters(
    X_tr_tfidf: Any,
    y_train_sub: np.ndarray,
    X_val_tfidf: Any,
    y_val_sub: np.ndarray
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Tunes regularization parameter C for Logistic Regression and LinearSVC on validation split.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: HYPERPARAMETER TUNING (ON VALIDATION SET)")
    print("=" * 60)

    tuning_history: Dict[str, Any] = {"logistic_regression": {}, "linear_svc": {}}

    # 1. Tune Logistic Regression
    lr_c_candidates = [0.5, 1.0, 2.0, 5.0, 10.0]
    best_lr_c = 1.0
    best_lr_score = -1.0

    print("Tuning Multinomial Logistic Regression (C in [0.5, 1.0, 2.0, 5.0, 10.0])...")
    for c in lr_c_candidates:
        lr = LogisticRegression(C=c, max_iter=600, random_state=RANDOM_SEED)
        lr.fit(X_tr_tfidf, y_train_sub)
        val_acc = lr.score(X_val_tfidf, y_val_sub)
        tuning_history["logistic_regression"][str(c)] = round(float(val_acc), 4)
        print(f"  LR (C={c}): Val Accuracy = {val_acc:.4f}")
        if val_acc > best_lr_score:
            best_lr_score = val_acc
            best_lr_c = c

    # 2. Tune LinearSVC
    svm_c_candidates = [0.1, 0.5, 1.0, 2.0]
    best_svm_c = 1.0
    best_svm_score = -1.0

    print("\nTuning LinearSVC (C in [0.1, 0.5, 1.0, 2.0])...")
    for c in svm_c_candidates:
        svm = LinearSVC(C=c, max_iter=2000, random_state=RANDOM_SEED)
        svm.fit(X_tr_tfidf, y_train_sub)
        val_acc = svm.score(X_val_tfidf, y_val_sub)
        tuning_history["linear_svc"][str(c)] = round(float(val_acc), 4)
        print(f"  LinearSVC (C={c}): Val Accuracy = {val_acc:.4f}")
        if val_acc > best_svm_score:
            best_svm_score = val_acc
            best_svm_c = c

    print(f"\n[BEST HYPERPARAMETERS]")
    print(f"  Logistic Regression Best C: {best_lr_c} (Val Acc: {best_lr_score:.4f})")
    print(f"  LinearSVC Best C: {best_svm_c} (Val Acc: {best_svm_score:.4f})")

    return best_lr_c, best_svm_c, tuning_history


def train_and_benchmark():
    """Main training, evaluation, and serialization routine."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("SUPPORTSENSE PHASE 3: ML TRAINING & BENCHMARKING")
    print("=" * 60)

    # 1. Load Data
    df_train, df_test, categories = load_datasets()
    print(f"[OK] Loaded Official Train: {len(df_train)} rows, Official Test: {len(df_test)} rows")
    assert len(categories) == 77, "Expected exactly 77 categories"

    # 2. Fit Label Encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(categories)
    assert len(label_encoder.classes_) == 77, "Label encoder must have 77 classes"

    y_train = label_encoder.transform(df_train["category"])
    y_test = label_encoder.transform(df_test["category"])

    # 3. Create Validation Split from Train Data (80/20 Stratified)
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        df_train["text"],
        y_train,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_train
    )
    print(f"[OK] Training Sub-Split: {len(X_train_sub)} samples | Validation Split: {len(X_val_sub)} samples")

    # 4. Stage 1: Empirical Preprocessing Selection
    best_prep_mode, prep_results = compare_preprocessing(
        X_train_sub, y_train_sub, X_val_sub, y_val_sub
    )
    cleaner = clean_text_basic if best_prep_mode == "basic" else clean_text_extended

    # Apply selected cleaner to splits for hyperparameter tuning
    X_tr_sub_clean = X_train_sub.apply(cleaner)
    X_val_sub_clean = X_val_sub.apply(cleaner)

    # Fit TF-IDF on train_sub for tuning
    vec_tuning = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X_tr_sub_tfidf = vec_tuning.fit_transform(X_tr_sub_clean)
    X_val_sub_tfidf = vec_tuning.transform(X_val_sub_clean)

    # 5. Stage 2: Hyperparameter Tuning
    best_lr_c, best_svm_c, tuning_history = tune_hyperparameters(
        X_tr_sub_tfidf, y_train_sub, X_val_sub_tfidf, y_val_sub
    )

    # 6. Stage 3: Final Model Fitting on Full Official Training Set (10,003 rows)
    print("\n" + "=" * 60)
    print("STAGE 3: FITTING FINAL MODELS ON FULL TRAINING SET (10,003 SAMPLES)")
    print("=" * 60)

    # Clean full train and test texts using chosen cleaner
    print("Cleaning full training and official test texts...")
    X_train_full_clean = df_train["text"].apply(cleaner)
    X_test_clean = df_test["text"].apply(cleaner)

    # Fit TF-IDF ONLY on full training set (Zero leakage!)
    print("Fitting final TF-IDF Vectorizer on full training data...")
    final_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )
    X_train_full_tfidf = final_vectorizer.fit_transform(X_train_full_clean)
    X_test_tfidf = final_vectorizer.transform(X_test_clean)
    print(f"[OK] Final Vocabulary Size: {len(final_vectorizer.vocabulary_)} features")

    # Fit Final Tuned Logistic Regression
    print(f"Training Final Multinomial Logistic Regression (C={best_lr_c})...")
    final_lr = LogisticRegression(
        C=best_lr_c,
        max_iter=1000,
        random_state=RANDOM_SEED
    )
    final_lr.fit(X_train_full_tfidf, y_train)

    # Fit Final Tuned LinearSVC with Probability Calibration (Platt Scaling)
    print(f"Training Final Calibrated LinearSVC (C={best_svm_c}, 3-fold Platt calibration)...")
    base_svm = LinearSVC(
        C=best_svm_c,
        max_iter=2500,
        random_state=RANDOM_SEED
    )
    final_svm_calibrated = CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",
        cv=3
    )
    final_svm_calibrated.fit(X_train_full_tfidf, y_train)

    # 7. Stage 4: Unbiased Final Evaluation on Official Test Set (3,080 rows)
    print("\n" + "=" * 60)
    print("STAGE 4: UNBIASED FINAL EVALUATION ON OFFICIAL TEST SET (3,080 SAMPLES)")
    print("=" * 60)

    raw_test_texts = list(df_test["text"])

    lr_eval = evaluate_model_comprehensive(
        model=final_lr,
        X_features=X_test_tfidf,
        y_true_labels=list(df_test["category"]),
        label_encoder=label_encoder,
        model_name="Multinomial Logistic Regression",
        raw_texts=raw_test_texts,
        cleaner_func=cleaner,
        vectorizer=final_vectorizer
    )

    svm_eval = evaluate_model_comprehensive(
        model=final_svm_calibrated,
        X_features=X_test_tfidf,
        y_true_labels=list(df_test["category"]),
        label_encoder=label_encoder,
        model_name="Calibrated Linear Support Vector Machine (LinearSVC)",
        raw_texts=raw_test_texts,
        cleaner_func=cleaner,
        vectorizer=final_vectorizer
    )

    print("\n[RESULTS SUMMARY - OFFICIAL TEST SET]")
    print(f"{'Metric':<30} | {'Logistic Regression':<22} | {'Calibrated LinearSVC':<22}")
    print("-" * 80)
    print(f"{'Accuracy':<30} | {lr_eval['accuracy']:<22.4f} | {svm_eval['accuracy']:<22.4f}")
    print(f"{'Top-1 Accuracy':<30} | {lr_eval['top_1_accuracy']:<22.4f} | {svm_eval['top_1_accuracy']:<22.4f}")
    print(f"{'Top-3 Accuracy':<30} | {lr_eval['top_3_accuracy']:<22.4f} | {svm_eval['top_3_accuracy']:<22.4f}")
    print(f"{'Macro F1 Score':<30} | {lr_eval['macro_f1']:<22.4f} | {svm_eval['macro_f1']:<22.4f}")
    print(f"{'Weighted F1 Score':<30} | {lr_eval['weighted_f1']:<22.4f} | {svm_eval['weighted_f1']:<22.4f}")
    print(f"{'Isolated Matrix Latency (ms)':<30} | {lr_eval['isolated_inference_latency']['mean_latency_ms']:<22.3f} | {svm_eval['isolated_inference_latency']['mean_latency_ms']:<22.3f}")
    print(f"{'Isolated Batch Throughput':<30} | {lr_eval['isolated_inference_latency']['batch_throughput_samples_per_sec']:<22.1f} | {svm_eval['isolated_inference_latency']['batch_throughput_samples_per_sec']:<22.1f}")
    if lr_eval.get("end_to_end_latency") and svm_eval.get("end_to_end_latency"):
        print(f"{'End-to-End Latency (ms)':<30} | {lr_eval['end_to_end_latency']['mean_latency_ms']:<22.3f} | {svm_eval['end_to_end_latency']['mean_latency_ms']:<22.3f}")
        print(f"{'End-to-End Batch Throughput':<30} | {lr_eval['end_to_end_latency']['batch_throughput_samples_per_sec']:<22.1f} | {svm_eval['end_to_end_latency']['batch_throughput_samples_per_sec']:<22.1f}")

    print("\nTop Confused Pairs (LinearSVC):")
    for pair in svm_eval["top_confused_pairs"]:
        print(f"  - True: '{pair['true_intent']}' -> Pred: '{pair['predicted_intent']}' ({pair['count']} errors)")

    # 8. Stage 5: Serialize Artifacts
    print("\n" + "=" * 60)
    print("STAGE 5: SERIALIZING MODEL ARTIFACTS TO models/")
    print("=" * 60)

    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    lr_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
    svm_path = os.path.join(MODELS_DIR, "linear_svm.joblib")
    le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    metrics_path = os.path.join(MODELS_DIR, "eval_metrics.json")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    joblib.dump(final_vectorizer, vec_path)
    joblib.dump(final_lr, lr_path)
    joblib.dump(final_svm_calibrated, svm_path)
    joblib.dump(label_encoder, le_path)

    full_metrics = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "dataset": "BANKING77",
        "num_classes": 77,
        "official_train_size": len(df_train),
        "official_test_size": len(df_test),
        "validation_split_ratio": 0.20,
        "selected_preprocessing": best_prep_mode,
        "preprocessing_validation_comparison": prep_results,
        "hyperparameter_tuning": tuning_history,
        "models": {
            "logistic_regression": lr_eval,
            "calibrated_linear_svm": svm_eval
        }
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    metadata = {
        "sklearn_version": sklearn.__version__,
        "random_seed": RANDOM_SEED,
        "vectorizer": {
            "type": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "max_features": 10000,
            "sublinear_tf": True,
            "vocabulary_size": len(final_vectorizer.vocabulary_)
        },
        "logistic_regression_params": {
            "C": best_lr_c,
            "max_iter": 1000,
            "solver": "lbfgs"
        },
        "linear_svm_params": {
            "C": best_svm_c,
            "calibration_method": "sigmoid (Platt scaling)",
            "cv": 3
        },
        "preprocessing": {
            "mode": best_prep_mode,
            "negation_and_intent_words_preserved": True
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved Vectorizer: {vec_path}")
    print(f"[OK] Saved Logistic Regression: {lr_path}")
    print(f"[OK] Saved Calibrated LinearSVM: {svm_path}")
    print(f"[OK] Saved Label Encoder: {le_path}")
    print(f"[OK] Saved Evaluation Metrics: {metrics_path}")
    print(f"[OK] Saved Model Metadata: {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_benchmark()
