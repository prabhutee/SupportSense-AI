"""
Test suite for Phase 5: MongoDB Schema Validation, CRUD Operations, and Aggregation Pipelines.

Clearly distinguishes:
1. Mocked Unit Tests: Executed in-memory via mongomock for fast, isolated CI/CD testing.
2. Schema Validator Tests: Verifies $jsonSchema compliance and rejection of invalid payloads.
3. Aggregation Pipeline Tests: Verifies analytical computations ($group, $project, $facet, etc.).
4. Live Integration Tests: Conditionally executed against live MongoDB if a daemon is reachable.
"""

import os
import json
import pytest
from datetime import datetime
from jsonschema import validate, ValidationError

from config.db_config import COLLECTION_TICKETS, COLLECTION_KB
from src.database.connection import MongoDBConnection, get_db
from src.database.schema_validator import TICKETS_VALIDATOR, KB_VALIDATOR, apply_schema_validators
from src.database.indexes import create_all_indexes
from src.database.ticket_repo import TicketRepository
from src.database.kb_repo import KBRepository
from src.database.aggregations import (
    get_intent_volume_and_confidence,
    get_priority_sla_breakdown,
    get_model_performance_in_production,
    get_executive_kpis,
    get_daily_volume_trend
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CATEGORIES_PATH = os.path.join(BASE_DIR, "data", "raw", "categories.json")


@pytest.fixture(scope="module")
def mock_db():
    """Provides an isolated in-memory mongomock Database instance."""
    db = MongoDBConnection.get_database(mock=True, db_name="test_supportsense_db")
    # Clean up prior to test run
    for col in db.list_collection_names():
        db[col].drop()
    return db


@pytest.fixture(scope="module")
def ticket_repo(mock_db):
    return TicketRepository(mock_db)


@pytest.fixture(scope="module")
def kb_repo(mock_db):
    return KBRepository(mock_db)


# ============================================================================
# PART 1: MOCKED TICKET REPOSITORY CRUD TESTS
# ============================================================================

def test_ticket_create_and_read(ticket_repo):
    """Test ticket creation and retrieval by ticket_id."""
    sample_ticket = {
        "customer_email": "john.doe@example.com",
        "query_text": "I lost my debit card at the station",
        "clean_text": "lose debit card station",
        "predicted_intent": "lost_or_stolen_card",
        "intent_confidence": 0.9450,
        "model_used": "LinearSVM",
        "priority": "Critical",
        "priority_reason": "Security Risk: Intent 'lost_or_stolen_card' mapped to Critical tier.",
        "status": "Open",
        "needs_manual_review": False
    }

    ticket_id = ticket_repo.create_ticket(sample_ticket)
    assert ticket_id.startswith("TICKET-")

    retrieved = ticket_repo.get_ticket_by_id(ticket_id)
    assert retrieved is not None
    assert retrieved["ticket_id"] == ticket_id
    assert retrieved["customer_email"] == "john.doe@example.com"
    assert retrieved["priority"] == "Critical"
    assert retrieved["status"] == "Open"
    assert isinstance(retrieved["created_at"], datetime)


def test_ticket_update_status(ticket_repo):
    """Test updating ticket workflow status and assigning support agent."""
    sample_ticket = {
        "customer_email": "jane.smith@example.com",
        "query_text": "Why was a transfer fee charged on my account?",
        "clean_text": "transfer fee charge account",
        "predicted_intent": "transfer_fee_charged",
        "intent_confidence": 0.8820,
        "model_used": "LogisticRegression",
        "priority": "Medium",
        "priority_reason": "Business Rule: Intent mapped to Medium tier.",
        "status": "Open"
    }
    ticket_id = ticket_repo.create_ticket(sample_ticket)

    success = ticket_repo.update_ticket_status(
        ticket_id=ticket_id,
        status="In Progress",
        assigned_agent="Agent_Smith",
        resolution_notes="Investigating fee schedule."
    )
    assert success is True

    updated = ticket_repo.get_ticket_by_id(ticket_id)
    assert updated["status"] == "In Progress"
    assert updated["assigned_agent"] == "Agent_Smith"
    assert updated["resolution_notes"] == "Investigating fee schedule."


def test_ticket_list_and_filter(ticket_repo):
    """Test filtering tickets by status and priority."""
    # Count open tickets
    open_tickets = ticket_repo.list_tickets(status="Open")
    assert len(open_tickets) >= 1
    for t in open_tickets:
        assert t["status"] == "Open"


def test_ticket_delete(ticket_repo):
    """Test deleting a ticket by ticket_id."""
    sample_ticket = {
        "customer_email": "temp.user@example.com",
        "query_text": "Temporary test ticket",
        "clean_text": "temporary test ticket",
        "predicted_intent": "card_arrival",
        "intent_confidence": 0.75,
        "model_used": "LinearSVM",
        "priority": "Low",
        "priority_reason": "General inquiry",
        "status": "Closed"
    }
    ticket_id = ticket_repo.create_ticket(sample_ticket)
    assert ticket_repo.get_ticket_by_id(ticket_id) is not None

    deleted = ticket_repo.delete_ticket(ticket_id)
    assert deleted is True
    assert ticket_repo.get_ticket_by_id(ticket_id) is None


# ============================================================================
# PART 2: MOCKED KNOWLEDGE BASE REPOSITORY CRUD & SEEDING TESTS
# ============================================================================

def test_kb_seed_from_json(kb_repo):
    """Test bulk upsert seeding from data/seed/kb_articles.json."""
    count = kb_repo.seed_kb_from_json()
    assert count >= 15, f"Expected at least 15 articles seeded, got {count}"

    # Verify article retrieval
    article = kb_repo.get_article_by_id("KB-CARD-001")
    assert article is not None
    assert article["category_group"] == "Card Services"
    assert "lost_or_stolen_card" in article["relevant_intents"]


def test_kb_list_by_category(kb_repo):
    """Test retrieving articles filtered by category_group."""
    cards = kb_repo.list_articles(category_group="Card Services")
    assert len(cards) >= 1
    for a in cards:
        assert a["category_group"] == "Card Services"


def test_kb_helpful_count_increment(kb_repo):
    """Test atomic increment of helpfulness counter."""
    before = kb_repo.get_article_by_id("KB-CARD-001")["helpful_count"]
    success = kb_repo.increment_helpful_count("KB-CARD-001")
    assert success is True

    after = kb_repo.get_article_by_id("KB-CARD-001")["helpful_count"]
    assert after == before + 1


# ============================================================================
# PART 3: SCHEMA VALIDATOR UNIT TESTS ($jsonSchema Compliance)
# ============================================================================

def _mongo_schema_to_jsonschema(mongo_schema: dict) -> dict:
    """Converts MongoDB-specific bsonTypes to standard jsonschema types for local validation."""
    schema_str = json.dumps(mongo_schema["$jsonSchema"])
    # Map bsonType -> type, and date/double types
    schema_str = schema_str.replace('"bsonType": "object"', '"type": "object"')
    schema_str = schema_str.replace('"bsonType": "string"', '"type": "string"')
    schema_str = schema_str.replace('"bsonType": "double"', '"type": "number"')
    schema_str = schema_str.replace('"bsonType": "int"', '"type": "integer"')
    schema_str = schema_str.replace('"bsonType": "bool"', '"type": "boolean"')
    schema_str = schema_str.replace('"bsonType": "array"', '"type": "array"')
    schema_str = schema_str.replace('"bsonType": ["string", "null"]', '"type": ["string", "null"]')
    schema_str = schema_str.replace('"bsonType": "date"', '"type": "string"')
    return json.loads(schema_str)


def test_ticket_schema_validation_success():
    """Verify that a compliant ticket dictionary conforms to the schema rules."""
    compliant_ticket = {
        "ticket_id": "TICKET-12345678",
        "customer_email": "valid.user@bank.com",
        "query_text": "I need help with my pin",
        "clean_text": "need help pin",
        "predicted_intent": "pin_blocked",
        "intent_confidence": 0.92,
        "model_used": "LinearSVM",
        "priority": "High",
        "priority_reason": "High Financial Impact: Blocked access.",
        "status": "Open",
        "created_at": "2026-11-14T12:00:00Z"
    }
    validator = _mongo_schema_to_jsonschema(TICKETS_VALIDATOR)
    validate(instance=compliant_ticket, schema=validator)


def test_ticket_schema_validation_rejection():
    """Verify that malformed tickets (invalid priority enum, invalid email) fail validation."""
    validator = _mongo_schema_to_jsonschema(TICKETS_VALIDATOR)

    # 1. Invalid Priority Enum
    invalid_priority = {
        "ticket_id": "TICKET-12345678",
        "customer_email": "user@bank.com",
        "query_text": "Valid text query",
        "clean_text": "valid text query",
        "predicted_intent": "pin_blocked",
        "intent_confidence": 0.90,
        "model_used": "LinearSVM",
        "priority": "Super_Urgent_Invalid_Tier",  # Invalid
        "priority_reason": "...",
        "status": "Open",
        "created_at": "2026-11-14T12:00:00Z"
    }
    with pytest.raises(ValidationError):
        validate(instance=invalid_priority, schema=validator)

    # 2. Invalid Email Format
    invalid_email = dict(invalid_priority, priority="High", customer_email="not-an-email")
    with pytest.raises(ValidationError):
        validate(instance=invalid_email, schema=validator)


# ============================================================================
# PART 4: AGGREGATION PIPELINE TESTS
# ============================================================================

def test_aggregation_pipelines(mock_db, ticket_repo):
    """
    Seeds mock database with multi-status tickets and tests all aggregation pipelines.
    """
    # Clear and seed known set of tickets
    mock_db[COLLECTION_TICKETS].delete_many({})

    sample_seed = [
        # Critical
        {"customer_email": "c1@test.com", "query_text": "Lost card", "clean_text": "lost card", "predicted_intent": "lost_or_stolen_card", "intent_confidence": 0.95, "model_used": "LinearSVM", "priority": "Critical", "priority_reason": "Fraud risk", "status": "Open", "needs_manual_review": False},
        {"customer_email": "c2@test.com", "query_text": "Lost card 2", "clean_text": "lost card", "predicted_intent": "lost_or_stolen_card", "intent_confidence": 0.90, "model_used": "LinearSVM", "priority": "Critical", "priority_reason": "Fraud risk", "status": "Resolved", "needs_manual_review": False},
        # High
        {"customer_email": "h1@test.com", "query_text": "Pin blocked", "clean_text": "pin blocked", "predicted_intent": "pin_blocked", "intent_confidence": 0.85, "model_used": "LogisticRegression", "priority": "High", "priority_reason": "Lockout", "status": "Open", "needs_manual_review": False},
        # Low with manual review flag
        {"customer_email": "l1@test.com", "query_text": "Ambiguous question", "clean_text": "ambiguous", "predicted_intent": "exchange_rate", "intent_confidence": 0.25, "model_used": "LinearSVM", "priority": "Low", "priority_reason": "General", "status": "Open", "needs_manual_review": True}
    ]
    for s in sample_seed:
        ticket_repo.create_ticket(s)

    # 1. Pipeline: Intent Breakdown
    intent_agg = get_intent_volume_and_confidence(mock_db, limit=5)
    assert len(intent_agg) >= 2
    top_intent = next(item for item in intent_agg if item["intent"] == "lost_or_stolen_card")
    assert top_intent["total_tickets"] == 2
    assert top_intent["resolved_tickets"] == 1
    assert top_intent["open_tickets"] == 1
    assert top_intent["resolution_rate_pct"] == 50.0

    # 2. Pipeline: Priority SLA Breakdown
    priority_agg = get_priority_sla_breakdown(mock_db)
    crit_p = next(item for item in priority_agg if item["priority"] == "Critical")
    assert crit_p["total"] == 2
    assert crit_p["resolved_count"] == 1

    # 3. Pipeline: Model Comparison
    model_agg = get_model_performance_in_production(mock_db)
    assert len(model_agg) == 2  # LinearSVM and LogisticRegression

    # 4. Pipeline: Executive KPIs via $facet
    kpis = get_executive_kpis(mock_db)
    assert kpis["total_tickets"] == 4
    assert kpis["critical_tickets"] == 2
    assert kpis["resolved_tickets"] == 1
    assert kpis["manual_review_tickets"] == 1
    assert kpis["resolution_rate_pct"] == 25.0


# ============================================================================
# PART 5: LIVE MONGODB INTEGRATION TEST (CONDITIONAL)
# ============================================================================

def is_live_mongodb_running() -> bool:
    """Helper to detect if a real MongoDB daemon is listening."""
    return MongoDBConnection.ping(timeout_ms=500)


@pytest.mark.skipif(not is_live_mongodb_running(), reason="Live MongoDB instance not available at MONGODB_URI.")
def test_live_mongodb_full_integration():
    """
    Runs only when a live MongoDB instance is available.
    Verifies actual database connection, index creation, and schema enforcement.
    """
    live_db = MongoDBConnection.get_database(mock=False, db_name="supportsense_integration_test")
    # Apply schema validators
    val_results = apply_schema_validators(live_db)
    assert val_results["tickets"] is True
    assert val_results["knowledge_base"] is True

    # Apply indexes
    idx_results = create_all_indexes(live_db)
    assert len(idx_results["tickets"]) >= 5
    assert len(idx_results["knowledge_base"]) >= 3

    # Clean up test DB
    live_db.client.drop_database("supportsense_integration_test")
