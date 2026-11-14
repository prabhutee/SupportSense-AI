"""
Resolution Recommendation Engine for SupportSense.

Retrieves and ranks curated Knowledge Base (KB) resolution SOPs using:
1. Pre-filtering by predicted BANKING77 intent or category group.
2. Vector space representation via a dedicated KB TF-IDF vectorizer (independent of the intent classifier).
3. Cosine similarity scoring and Top-K ranking.
4. Configurable similarity thresholding with automated fallback and manual review escalation.
"""

import os
import sys
import json
import joblib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml.preprocessor import clean_text_extended

DEFAULT_KB_PATH = os.path.join(BASE_DIR, "data", "seed", "kb_articles.json")
DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_VECTORIZER_PATH = os.path.join(DEFAULT_MODELS_DIR, "kb_tfidf_vectorizer.joblib")


class KBRecommender:
    """
    Dedicated Knowledge Base retrieval engine utilizing TF-IDF and Cosine Similarity.
    Decoupled from the intent classification model pipeline.
    """

    def __init__(
        self,
        kb_path: str = DEFAULT_KB_PATH,
        models_dir: str = DEFAULT_MODELS_DIR,
        default_threshold: float = 0.15,
        default_top_k: int = 3,
        auto_build: bool = True
    ):
        self.kb_path = kb_path
        self.models_dir = models_dir
        self.vectorizer_path = os.path.join(models_dir, "kb_tfidf_vectorizer.joblib")
        self.default_threshold = default_threshold
        self.default_top_k = default_top_k

        self.articles: List[Dict[str, Any]] = []
        self.article_id_to_index: Dict[str, int] = {}
        self.kb_vectorizer: Optional[TfidfVectorizer] = None
        self.kb_matrix = None  # Pre-computed TF-IDF matrix for all KB articles

        if auto_build:
            self._initialize_corpus_and_vectorizer()

    def _build_article_text(self, article: Dict[str, Any]) -> str:
        """Concatenates searchable content for an article."""
        parts = [
            article.get("title", ""),
            article.get("summary", ""),
            " ".join(article.get("tags", [])),
            article.get("content", ""),
            " ".join(article.get("resolution_steps", [])),
            " ".join(article.get("relevant_intents", []))
        ]
        raw_combined = " ".join(p for p in parts if p)
        return clean_text_extended(raw_combined)

    def _initialize_corpus_and_vectorizer(self) -> None:
        """Loads KB JSON, creates or loads the dedicated TF-IDF vectorizer, and pre-computes document vectors."""
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge Base seed file not found at: {self.kb_path}")

        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.articles = json.load(f)

        self.article_id_to_index = {a["kb_id"]: idx for idx, a in enumerate(self.articles)}

        # Build cleaned corpus texts
        corpus_texts = [self._build_article_text(a) for a in self.articles]

        os.makedirs(self.models_dir, exist_ok=True)

        # Check if saved dedicated KB vectorizer exists
        if os.path.exists(self.vectorizer_path):
            try:
                self.kb_vectorizer = joblib.load(self.vectorizer_path)
            except Exception:
                self.kb_vectorizer = None

        if self.kb_vectorizer is None:
            # Fit dedicated vectorizer exclusively on the KB corpus
            self.kb_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1
            )
            self.kb_vectorizer.fit(corpus_texts)
            joblib.dump(self.kb_vectorizer, self.vectorizer_path)

        # Pre-compute document feature matrix
        self.kb_matrix = self.kb_vectorizer.transform(corpus_texts)

    def get_eligible_indices(
        self,
        predicted_intent: Optional[str] = None,
        category_group: Optional[str] = None
    ) -> Tuple[List[int], str]:
        """
        Pre-filters KB article indices by intent or category.
        Returns: (eligible_indices, strategy_description).
        """
        # 1. Intent-level filtering
        if predicted_intent:
            indices = [
                idx for idx, a in enumerate(self.articles)
                if predicted_intent in a.get("relevant_intents", [])
            ]
            if indices:
                return indices, "intent_prefilter"

        # 2. Category group-level filtering
        if category_group:
            indices = [
                idx for idx, a in enumerate(self.articles)
                if a.get("category_group", "").lower() == category_group.lower()
            ]
            if indices:
                return indices, "category_prefilter"

        # 3. Global fallback (all articles eligible)
        return list(range(len(self.articles))), "global_fallback"

    def recommend(
        self,
        query_text: str,
        predicted_intent: Optional[str] = None,
        category_group: Optional[str] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Retrieves Top-K resolution recommendations for a customer query:
        1. Pre-filters articles by intent or category.
        2. Vectorizes query in dedicated KB TF-IDF space.
        3. Computes Cosine Similarity against candidate articles.
        4. Drops items below similarity threshold.
        5. Flags for manual review if no item meets threshold.
        """
        applied_threshold = threshold if threshold is not None else self.default_threshold
        applied_top_k = top_k if top_k is not None else self.default_top_k

        # Edge case: Empty or whitespace query
        if not query_text or not query_text.strip():
            return {
                "status": "no_recommendation_found",
                "recommendations": [],
                "top_similarity_score": 0.0,
                "threshold_applied": applied_threshold,
                "filtered_candidate_count": 0,
                "filtering_strategy": "none",
                "intent_used_for_filtering": predicted_intent,
                "needs_manual_review": True,
                "manual_review_reason": "Empty or whitespace-only query text provided."
            }

        # 1. Pre-filtering
        eligible_indices, strategy = self.get_eligible_indices(
            predicted_intent=predicted_intent,
            category_group=category_group
        )

        if not eligible_indices:
            return {
                "status": "no_recommendation_found",
                "recommendations": [],
                "top_similarity_score": 0.0,
                "threshold_applied": applied_threshold,
                "filtered_candidate_count": 0,
                "filtering_strategy": strategy,
                "intent_used_for_filtering": predicted_intent,
                "needs_manual_review": True,
                "manual_review_reason": "No knowledge-base articles matched the candidate pre-filter."
            }

        # 2. Vectorize Query
        cleaned_query = clean_text_extended(query_text)
        query_vec = self.kb_vectorizer.transform([cleaned_query])

        # 3. Compute Cosine Similarity against eligible candidate subset
        candidate_matrix = self.kb_matrix[eligible_indices]
        similarities = cosine_similarity(query_vec, candidate_matrix)[0]

        # 4. Rank candidates descending
        ranked_local_indices = np.argsort(similarities)[::-1]

        recommendations: List[Dict[str, Any]] = []
        top_score = 0.0

        if len(similarities) > 0:
            top_score = round(float(similarities[ranked_local_indices[0]]), 4)

        for local_idx in ranked_local_indices:
            score = float(similarities[local_idx])
            if score < applied_threshold:
                # Stop as soon as scores drop below threshold (since sorted descending)
                break
            
            global_article_idx = eligible_indices[local_idx]
            article = self.articles[global_article_idx]

            recommendations.append({
                "kb_id": article["kb_id"],
                "title": article["title"],
                "category_group": article["category_group"],
                "similarity_score": round(score, 4),
                "summary": article.get("summary", ""),
                "resolution_steps": article.get("resolution_steps", []),
                "disclaimer": article.get("disclaimer", "")
            })

            if len(recommendations) >= applied_top_k:
                break

        # 5. Threshold Handling and Manual Review Flagging
        if not recommendations:
            return {
                "status": "no_recommendation_found",
                "recommendations": [],
                "top_similarity_score": top_score,
                "threshold_applied": applied_threshold,
                "filtered_candidate_count": len(eligible_indices),
                "filtering_strategy": strategy,
                "intent_used_for_filtering": predicted_intent,
                "needs_manual_review": True,
                "manual_review_reason": (
                    f"Top candidate similarity score ({top_score:.2f}) was below the "
                    f"minimum confidence threshold ({applied_threshold:.2f}); manual agent resolution required."
                )
            }

        return {
            "status": "success",
            "recommendations": recommendations,
            "top_similarity_score": top_score,
            "threshold_applied": applied_threshold,
            "filtered_candidate_count": len(eligible_indices),
            "filtering_strategy": strategy,
            "intent_used_for_filtering": predicted_intent,
            "needs_manual_review": False,
            "manual_review_reason": None
        }
