"""
Test suite for Phase 4: Resolution Recommendation Engine and End-to-End Latency Benchmark.
"""

import os
import joblib
import pytest
from src.recommendation.recommender import KBRecommender
from src.ml.evaluate import measure_end_to_end_latency, measure_isolated_inference_latency
from src.ml.preprocessor import clean_text_extended

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
KB_PATH = os.path.join(BASE_DIR, "data", "seed", "kb_articles.json")


@pytest.fixture(scope="module")
def recommender():
    """Instantiate and initialize the KBRecommender."""
    return KBRecommender(kb_path=KB_PATH, models_dir=MODELS_DIR, default_threshold=0.20)


# ============================================================================
# 1. DEDICATED VECTORIZER TESTS
# ============================================================================

def test_kb_vectorizer_separate_from_intent_vectorizer(recommender):
    """
    Verify that the KB retrieval vectorizer is fitted exclusively on the KB corpus
    and is stored completely separately from the intent-classification vectorizer.
    """
    kb_vec_path = os.path.join(MODELS_DIR, "kb_tfidf_vectorizer.joblib")
    intent_vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

    assert os.path.exists(kb_vec_path), "Missing dedicated KB TF-IDF vectorizer artifact."
    assert os.path.exists(intent_vec_path), "Missing intent classification vectorizer artifact."

    kb_vec = joblib.load(kb_vec_path)
    intent_vec = joblib.load(intent_vec_path)

    # The intent classification vectorizer was fitted on 10,003 queries (vocab: 10,000)
    # The KB vectorizer was fitted exclusively on the 20 KB articles (vocab < 2,000)
    assert len(intent_vec.vocabulary_) == 10000
    assert len(kb_vec.vocabulary_) < 5000
    assert len(kb_vec.vocabulary_) != len(intent_vec.vocabulary_)


# ============================================================================
# 2. PRE-FILTERING AND COSINE SIMILARITY TESTS
# ============================================================================

def test_intent_pre_filtering(recommender):
    """Verify that candidate articles are properly pre-filtered by predicted intent."""
    res = recommender.recommend(
        query_text="I lost my card and suspect it was stolen",
        predicted_intent="lost_or_stolen_card"
    )
    assert res["status"] == "success"
    assert res["filtering_strategy"] == "intent_prefilter"
    assert res["filtered_candidate_count"] >= 1
    # Check that top recommendation is the lost card SOP
    assert res["recommendations"][0]["kb_id"] == "KB-CARD-001"
    assert "Immediate Protocol" in res["recommendations"][0]["title"]


def test_cosine_similarity_ranking_descending(recommender):
    """Verify that multiple recommendations are strictly ranked in descending order of similarity."""
    res = recommender.recommend(
        query_text="I need help activating my new physical card or getting a spare card delivered",
        predicted_intent="activate_my_card",
        top_k=3,
        threshold=0.10
    )
    assert res["status"] == "success"
    assert len(res["recommendations"]) >= 1
    
    scores = [r["similarity_score"] for r in res["recommendations"]]
    assert scores == sorted(scores, reverse=True), "Recommendations must be sorted descending by similarity."
    for s in scores:
        assert s >= 0.10


def test_similarity_threshold_rejection(recommender):
    """
    Verify that if similarity scores do not meet a strict threshold,
    no_recommendation_found is returned and the ticket is flagged for manual review.
    """
    res = recommender.recommend(
        query_text="Completely irrelevant cooking recipe with tomatoes and onions",
        predicted_intent="lost_or_stolen_card",
        threshold=0.90  # Impossibly high threshold
    )
    assert res["status"] == "no_recommendation_found"
    assert len(res["recommendations"]) == 0
    assert res["needs_manual_review"] is True
    assert "manual agent resolution required" in res["manual_review_reason"]
    assert res["threshold_applied"] == 0.90


def test_empty_query_handling(recommender):
    """Verify empty or whitespace-only queries are safely rejected and flagged."""
    res = recommender.recommend(
        query_text="     ",
        predicted_intent="exchange_rate"
    )
    assert res["status"] == "no_recommendation_found"
    assert res["needs_manual_review"] is True
    assert "Empty or whitespace-only query" in res["manual_review_reason"]


def test_unknown_intent_global_fallback(recommender):
    """
    Verify that an unrecognized or unmapped intent gracefully falls back to global search
    across all KB articles without throwing errors.
    """
    res = recommender.recommend(
        query_text="What are the foreign exchange rates for changing euros to dollars?",
        predicted_intent="non_existent_intent_12345",
        threshold=0.15
    )
    assert res["filtering_strategy"] == "global_fallback"
    assert res["filtered_candidate_count"] == len(recommender.articles)
    assert res["status"] == "success"
    # Should locate the FX article (KB-DISP-003)
    assert any("Exchange" in r["title"] or "Foreign" in r["title"] for r in res["recommendations"])


def test_top_k_parameter_honored(recommender):
    """Verify that top_k restricts the number of returned articles."""
    for k in [1, 2]:
        res = recommender.recommend(
            query_text="How do I change my PIN at an ATM or in the mobile application?",
            predicted_intent="change_pin",
            top_k=k,
            threshold=0.05
        )
        assert len(res["recommendations"]) <= k


# ============================================================================
# 3. END-TO-END LATENCY BENCHMARK STRUCTURE TEST
# ============================================================================

def test_end_to_end_latency_benchmark_method():
    """Verify that measure_end_to_end_latency runs and returns proper keys."""
    intent_vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))

    dummy_texts = ["My card was stolen in Madrid", "How do I top up my balance?"]
    result = measure_end_to_end_latency(
        raw_texts=dummy_texts,
        cleaner_func=clean_text_extended,
        vectorizer=intent_vec,
        model=lr_model,
        num_runs=2,
        batch_size=2
    )

    assert "mean_latency_ms" in result
    assert "batch_throughput_samples_per_sec" in result
    assert "end_to_end_pipeline" in result["benchmark_type"]
    assert result["mean_latency_ms"] > 0
