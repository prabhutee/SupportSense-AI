"""
Test suite for Phase 2: Priority Rules Engine and Knowledge Base Validation.
"""

import os
import json
import pytest
from src.rules.priority_engine import PriorityEngine

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CATEGORIES_PATH = os.path.join(BASE_DIR, "data", "raw", "categories.json")
KB_PATH = os.path.join(BASE_DIR, "data", "seed", "kb_articles.json")
RULES_PATH = os.path.join(BASE_DIR, "config", "priority_rules.json")


@pytest.fixture(scope="module")
def canonical_categories():
    """Load canonical 77 categories from BANKING77 raw definition."""
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        categories = json.load(f)
    return categories


@pytest.fixture(scope="module")
def priority_engine():
    """Instantiate PriorityEngine."""
    return PriorityEngine(config_path=RULES_PATH)


@pytest.fixture(scope="module")
def kb_data():
    """Load curated Knowledge Base JSON."""
    with open(KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ============================================================================
# PART A: PRIORITY ENGINE TESTS
# ============================================================================

def test_all_77_intents_are_mapped(priority_engine, canonical_categories):
    """Ensure every single one of the 77 BANKING77 intents has an explicit priority mapping."""
    mapped_intents = priority_engine.get_all_mapped_intents()
    
    assert len(canonical_categories) == 77, "BANKING77 should have 77 categories"
    for cat in canonical_categories:
        assert cat in mapped_intents, f"Missing priority mapping for intent: '{cat}'"
        assert mapped_intents[cat] in {"Critical", "High", "Medium", "Low"}, (
            f"Invalid priority tier '{mapped_intents[cat]}' for intent '{cat}'"
        )


def test_priority_tier_counts(priority_engine):
    """Verify tier distribution covers all 77 intents."""
    mapped = priority_engine.get_all_mapped_intents()
    tier_counts = {}
    for tier in mapped.values():
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    assert sum(tier_counts.values()) == 77
    assert tier_counts["Critical"] == 8
    assert tier_counts["High"] == 18
    assert tier_counts["Medium"] == 28
    assert tier_counts["Low"] == 23


def test_standard_intent_evaluation(priority_engine):
    """Test evaluation without keywords returns the correct base priority and reason."""
    result = priority_engine.evaluate(
        query_text="Where is my new physical card?",
        predicted_intent="card_arrival"
    )
    assert result["priority"] == "Medium"
    assert result["rule_triggered"] == "intent_mapping"
    assert result["is_escalated"] is False
    assert "Business Rule: Intent 'card_arrival' is classified in the Medium priority tier" in result["priority_reason"]


def test_critical_intent_evaluation(priority_engine):
    """Test that security-critical intents evaluate to Critical priority."""
    result = priority_engine.evaluate(
        query_text="I cannot find my debit card anywhere",
        predicted_intent="lost_or_stolen_card"
    )
    assert result["priority"] == "Critical"
    assert result["rule_triggered"] == "intent_mapping"
    assert result["is_escalated"] is False


def test_urgency_keyword_escalation(priority_engine):
    """Test that urgency keywords escalate lower-priority intents and log reasons."""
    # Base intent 'activate_my_card' is Low priority, but query mentions 'stolen'
    result = priority_engine.evaluate(
        query_text="Someone has stolen my identity and opened an account",
        predicted_intent="activate_my_card"
    )
    assert result["priority"] == "Critical"
    assert result["rule_triggered"] == "keyword_override"
    assert result["is_escalated"] is True
    assert result["matched_keyword"] == "stolen"
    assert "Urgency Override" in result["priority_reason"]
    assert "escalating priority from Low to Critical" in result["priority_reason"]


def test_high_urgency_keyword_escalation(priority_engine):
    """Test that high-urgency keywords ('urgent', 'asap') escalate Medium to High."""
    result = priority_engine.evaluate(
        query_text="I need this transfer fee reviewed urgently please",
        predicted_intent="transfer_fee_charged"
    )
    assert result["priority"] == "High"
    assert result["is_escalated"] is True
    assert result["matched_keyword"] == "urgently"


def test_unknown_intent_fallback(priority_engine):
    """Test that an unmapped intent gracefully falls back to Medium with reason."""
    result = priority_engine.evaluate(
        query_text="Random general question",
        predicted_intent="completely_unknown_intent"
    )
    assert result["priority"] == "Medium"
    assert result["rule_triggered"] == "intent_mapping"
    assert "Default Fallback" in result["priority_reason"]


def test_low_confidence_flagging(priority_engine):
    """Test that low confidence (<0.40) is annotated in priority reason."""
    result = priority_engine.evaluate(
        query_text="I have a question",
        predicted_intent="exchange_rate",
        confidence_score=0.25
    )
    assert "Low model confidence of 0.25; flagged for review" in result["priority_reason"]


# ============================================================================
# PART B: KNOWLEDGE BASE TESTS
# ============================================================================

def test_kb_json_structure(kb_data):
    """Validate structure and essential attributes of knowledge base articles."""
    assert len(kb_data) >= 15, f"Expected at least 15 articles, got {len(kb_data)}"
    
    required_keys = {
        "kb_id", "title", "category_group", "summary",
        "relevant_intents", "tags", "content", "resolution_steps",
        "disclaimer", "helpful_count"
    }

    seen_ids = set()
    for article in kb_data:
        # Check required fields
        for key in required_keys:
            assert key in article, f"Article missing required key '{key}': {article.get('kb_id')}"
        
        # Check unique ID
        kb_id = article["kb_id"]
        assert kb_id not in seen_ids, f"Duplicate kb_id detected: {kb_id}"
        seen_ids.add(kb_id)
        
        # Check non-empty content
        assert len(article["title"]) >= 5
        assert len(article["content"]) >= 50
        assert len(article["resolution_steps"]) >= 2
        assert len(article["relevant_intents"]) >= 1
        assert "Demo Knowledge-Base Content" in article["disclaimer"]


def test_kb_intents_are_valid_banking77(kb_data, canonical_categories):
    """Verify that every relevant_intent listed in KB belongs to canonical BANKING77 categories."""
    canonical_set = set(canonical_categories)
    
    for article in kb_data:
        for intent in article["relevant_intents"]:
            assert intent in canonical_set, (
                f"Article '{article['kb_id']}' contains unknown intent: '{intent}'"
            )


def test_kb_covers_all_77_intents(kb_data, canonical_categories):
    """Ensure that all 77 BANKING77 intents are mapped to at least one KB article."""
    covered_intents = set()
    for article in kb_data:
        for intent in article["relevant_intents"]:
            covered_intents.add(intent)
            
    uncovered = set(canonical_categories) - covered_intents
    assert len(uncovered) == 0, f"The following BANKING77 intents have no KB article coverage: {uncovered}"
