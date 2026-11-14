"""
Test suite for Phase 3: Text Preprocessing, Model Inference, and Artifact Integrity.
"""

import os
import json
import joblib
import pytest
import numpy as np

from src.ml.preprocessor import clean_text_basic, clean_text_extended, expand_contractions
from src.ml.predictor import IntentPredictor

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CATEGORIES_PATH = os.path.join(BASE_DIR, "data", "raw", "categories.json")


# ============================================================================
# 1. PREPROCESSING TESTS
# ============================================================================

def test_contraction_expansion():
    """Verify common contractions expand properly."""
    assert expand_contractions("I can't access my card") == "I can not access my card"
    assert expand_contractions("Why won't it work?") == "Why will not it work?"
    assert expand_contractions("He didn't receive the transfer") == "He did not receive the transfer"


def test_basic_cleaning_preserves_intent_words():
    """Verify basic cleaning removes noise and standardizes case."""
    raw = "HELP!! My card was STOLEN at http://badsite.com, email fraud@scam.org!"
    clean = clean_text_basic(raw)
    assert "http" not in clean
    assert "fraud@scam.org" not in clean
    assert "stolen" in clean
    assert "help" in clean


def test_extended_cleaning_preserves_crucial_intent_tokens():
    """
    Verify that extended cleaning with stopwords does NOT remove crucial negation
    or directional tokens that dictate intent.
    """
    raw = "My contactless is not working again and why was I charged twice?"
    clean = clean_text_extended(raw)
    assert "not" in clean
    assert "why" in clean
    assert "twice" in clean
    assert "again" in clean


# ============================================================================
# 2. MODEL ARTIFACT AND LABEL INTEGRITY TESTS
# ============================================================================

def test_saved_model_artifacts_exist():
    """Verify all serialized models and configuration files exist."""
    expected_files = [
        "tfidf_vectorizer.joblib",
        "logistic_regression.joblib",
        "linear_svm.joblib",
        "label_encoder.joblib",
        "eval_metrics.json",
        "model_metadata.json"
    ]
    for filename in expected_files:
        path = os.path.join(MODELS_DIR, filename)
        assert os.path.exists(path), f"Missing expected artifact: {path}"


def test_label_encoder_integrity():
    """Verify label encoder contains exactly 77 classes matching BANKING77."""
    le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    le = joblib.load(le_path)
    
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        categories = json.load(f)

    assert len(le.classes_) == 77
    assert set(le.classes_) == set(categories)
    # Check exact casing preservation
    assert "Refund_not_showing_up" in le.classes_
    assert "reverted_card_payment?" in le.classes_


def test_eval_metrics_structure():
    """Verify evaluation metrics were recorded for both models with top-3 accuracy."""
    metrics_path = os.path.join(MODELS_DIR, "eval_metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "models" in metrics
    assert "logistic_regression" in metrics["models"]
    assert "calibrated_linear_svm" in metrics["models"]
    
    svm_m = metrics["models"]["calibrated_linear_svm"]
    assert svm_m["accuracy"] > 0.85, f"Expected SVM accuracy > 0.85, got {svm_m['accuracy']}"
    assert svm_m["top_3_accuracy"] > svm_m["accuracy"], "Top-3 accuracy should exceed Top-1 accuracy"
    assert "latency" in svm_m


# ============================================================================
# 3. INFERENCE PREDICTOR TESTS
# ============================================================================

@pytest.fixture(scope="module")
def predictor():
    return IntentPredictor(models_dir=MODELS_DIR)


def test_prediction_card_lost(predictor):
    """Test inference on a typical card lost query."""
    result = predictor.predict("I lost my card while traveling and need it cancelled", model_name="LinearSVM")
    assert result["predicted_intent"] == "lost_or_stolen_card"
    assert result["confidence"] > 0.50
    assert result["needs_manual_review"] is False
    assert len(result["top_k_predictions"]) == 3
    assert result["top_k_predictions"][0]["intent"] == "lost_or_stolen_card"


def test_prediction_exchange_rate(predictor):
    """Test inference on foreign exchange query."""
    result = predictor.predict("What is the current exchange rate for US dollars to Euros?", model_name="LinearSVM")
    assert result["predicted_intent"] in {"exchange_rate", "exchange_via_app", "card_payment_wrong_exchange_rate"}
    assert result["confidence"] > 0.40


def test_low_confidence_flagging(predictor):
    """Test that ambiguous or gibberish input triggers manual review."""
    result = predictor.predict("asdfghjk zxcvbnm qwertyuiop random gibberish 123456")
    assert result["needs_manual_review"] is True
    assert result["confidence"] < 0.40
    assert result["manual_review_reason"] is not None


def test_empty_query_handling(predictor):
    """Test that empty queries are gracefully handled without crashing."""
    result = predictor.predict("    ")
    assert result["needs_manual_review"] is True
    assert result["confidence"] == 0.0
    assert result["clean_text"] == ""


def test_both_models_produce_valid_probabilities(predictor):
    """Verify both LR and Calibrated LinearSVM output valid probabilities."""
    for model_name in ["LogisticRegression", "LinearSVM"]:
        res = predictor.predict("My PIN is blocked how do I unlock it", model_name=model_name)
        assert res["predicted_intent"] == "pin_blocked"
        assert 0.0 <= res["confidence"] <= 1.0
        # Probabilities of top-k should be positive and sorted descending
        probs = [p["confidence"] for p in res["top_k_predictions"]]
        assert probs == sorted(probs, reverse=True)
