"""
Inference Service for SupportSense Intent Classification.

Loads serialized models, vectorizer, label encoder, and preprocessing pipeline.
Provides:
- Real-time intent classification with calibrated confidence scores.
- Top-K ranked predictions.
- Automated low-confidence threshold flagging for manual review.
"""

import os
import sys
import joblib
import numpy as np
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml.preprocessor import clean_text_basic, clean_text_extended
MODELS_DIR = os.path.join(BASE_DIR, "models")


class IntentPredictor:
    """Production inference wrapper for customer query intent classification."""

    def __init__(
        self,
        models_dir: str = MODELS_DIR,
        default_model: str = "LinearSVM",
        confidence_threshold: float = 0.40
    ):
        self.models_dir = models_dir
        self.default_model = default_model
        self.confidence_threshold = confidence_threshold

        # Artifact paths
        self.vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
        self.lr_path = os.path.join(models_dir, "logistic_regression.joblib")
        self.svm_path = os.path.join(models_dir, "linear_svm.joblib")
        self.le_path = os.path.join(models_dir, "label_encoder.joblib")
        self.meta_path = os.path.join(models_dir, "model_metadata.json")

        self.vectorizer = None
        self.lr_model = None
        self.svm_model = None
        self.label_encoder = None
        self.cleaner = clean_text_basic
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads serialized model files."""
        if not os.path.exists(self.vec_path):
            raise FileNotFoundError(f"Missing vectorizer artifact at: {self.vec_path}. Run train.py first.")

        self.vectorizer = joblib.load(self.vec_path)
        self.lr_model = joblib.load(self.lr_path)
        self.svm_model = joblib.load(self.svm_path)
        self.label_encoder = joblib.load(self.le_path)

        # Determine preprocessing mode from metadata if available
        if os.path.exists(self.meta_path):
            import json
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            mode = meta.get("preprocessing", {}).get("mode", "basic")
            self.cleaner = clean_text_basic if mode == "basic" else clean_text_extended

    def predict(
        self,
        text: str,
        model_name: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Classifies an input customer query and returns calibrated intent predictions.

        Args:
            text: Raw customer query string.
            model_name: 'LinearSVM' (default) or 'LogisticRegression'.
            top_k: Number of ranked alternative predictions to return.

        Returns:
            Dictionary containing:
            - predicted_intent: Top-1 predicted label string
            - confidence: Posterior probability of the top-1 label (0.0 to 1.0)
            - model_used: Name of the classification model utilized
            - top_k_predictions: List of dicts with intent and probability
            - needs_manual_review: Boolean flag if confidence is below threshold
            - manual_review_reason: Reason string if flagged
            - clean_text: Text after preprocessing
        """
        if not text or not text.strip():
            return {
                "predicted_intent": "unknown",
                "confidence": 0.0,
                "model_used": model_name or self.default_model,
                "top_k_predictions": [],
                "needs_manual_review": True,
                "manual_review_reason": "Empty or whitespace-only query text provided.",
                "clean_text": ""
            }

        selected_model_name = model_name or self.default_model
        model = self.svm_model if selected_model_name.lower() in {"linearsvm", "svm", "linear_svm"} else self.lr_model
        canonical_model_name = "LinearSVM" if model == self.svm_model else "LogisticRegression"

        clean_text = self.cleaner(text)
        tfidf_vec = self.vectorizer.transform([clean_text])

        # Get calibrated probabilities
        probabilities = model.predict_proba(tfidf_vec)[0]
        
        # Rank descending
        ranked_indices = np.argsort(probabilities)[::-1]
        top_idx = ranked_indices[0]
        top_intent = str(self.label_encoder.inverse_transform([top_idx])[0])
        top_confidence = float(probabilities[top_idx])

        # Build top-K list
        top_k = min(top_k, len(probabilities))
        top_k_preds = [
            {
                "intent": str(self.label_encoder.inverse_transform([idx])[0]),
                "confidence": round(float(probabilities[idx]), 4)
            }
            for idx in ranked_indices[:top_k]
        ]

        # Low confidence check
        needs_review = top_confidence < self.confidence_threshold
        review_reason = None
        if needs_review:
            review_reason = (
                f"Confidence score ({top_confidence:.2f}) is below the minimum operational threshold "
                f"({self.confidence_threshold:.2f}); manual agent verification required."
            )

        return {
            "predicted_intent": top_intent,
            "confidence": round(top_confidence, 4),
            "model_used": canonical_model_name,
            "top_k_predictions": top_k_preds,
            "needs_manual_review": needs_review,
            "manual_review_reason": review_reason,
            "clean_text": clean_text
        }
