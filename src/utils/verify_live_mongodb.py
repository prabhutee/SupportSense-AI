"""
Dedicated Live MongoDB Verification Script for SupportSense.

Executes and audits:
1. Live ping and connection verification on localhost:27017.
2. Creation of supportsense_db database and collections.
3. Enforcement of database-level $jsonSchema validators.
4. Index creation and verification via list_indexes().
5. Seeding of the 20 KB articles into knowledge_base collection.
6. Complete CRUD lifecycle test on actual tickets collection.
7. Execution of all 5 aggregation pipelines against live MongoDB data.
8. Proof that invalid documents are rejected by MongoDB server ($jsonSchema WriteError).
9. Safe cleanup of test ticket documents.
"""

import os
import sys
import json
from datetime import datetime, timezone
from pymongo.errors import OperationFailure, WriteError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.database.connection import MongoDBConnection
from src.database.schema_validator import apply_schema_validators
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
from config.db_config import COLLECTION_TICKETS, COLLECTION_KB, MONGODB_DB_NAME


def run_live_verification():
    print("=" * 70)
    print("SUPPORTSENSE: DEDICATED LIVE MONGODB VERIFICATION")
    print("=" * 70)

    # 1. Connection Ping
    print("\n[STEP 1] Testing Connection to Live MongoDB...")
    is_connected = MongoDBConnection.ping(timeout_ms=2000)
    if not is_connected:
        print("[FAIL] Could not connect to MongoDB on localhost:27017.")
        sys.exit(1)
    print("  -> [PASS] Successfully connected to live MongoDB server!")

    db = MongoDBConnection.get_database(mock=False, db_name=MONGODB_DB_NAME)
    ticket_repo = TicketRepository(db)
    kb_repo = KBRepository(db)

    # 2. Apply $jsonSchema Validators
    print(f"\n[STEP 2] Applying Database-Level $jsonSchema Validators to '{MONGODB_DB_NAME}'...")
    val_results = apply_schema_validators(db)
    for col, status in val_results.items():
        print(f"  -> Collection '{col}': $jsonSchema Validator Applied = {status}")
        assert status is True, f"Failed applying validator to {col}"

    # 3. Create and Verify Indexes
    print("\n[STEP 3] Creating and Inspecting MongoDB Indexes...")
    create_all_indexes(db)
    
    ticket_indexes = list(db[COLLECTION_TICKETS].list_indexes())
    kb_indexes = list(db[COLLECTION_KB].list_indexes())

    print(f"  -> '{COLLECTION_TICKETS}' active indexes ({len(ticket_indexes)}):")
    for idx in ticket_indexes:
        print(f"     - Name: '{idx['name']}', Keys: {idx['key'].to_dict()}")

    print(f"  -> '{COLLECTION_KB}' active indexes ({len(kb_indexes)}):")
    for idx in kb_indexes:
        print(f"     - Name: '{idx['name']}', Keys: {idx['key'].to_dict()}")

    assert len(ticket_indexes) >= 5, "Expected at least 5 indexes on tickets collection."
    assert len(kb_indexes) >= 3, "Expected at least 3 indexes on knowledge_base collection."

    # 4. Seed 21 KB Articles
    print("\n[STEP 4] Seeding Curated Knowledge Base Articles into Live MongoDB...")
    seeded_count = kb_repo.seed_kb_from_json()
    total_kb = db[COLLECTION_KB].count_documents({})
    print(f"  -> Seeded/Replaced {seeded_count} articles. Total in collection: {total_kb}")
    assert total_kb >= 20, f"Expected at least 20 articles in knowledge_base, got {total_kb}"

    # 5. Schema Validation Rejection Test (Proof of Database-Level Enforcement)
    print("\n[STEP 5] Testing MongoDB $jsonSchema Enforcement (Rejection of Invalid Docs)...")
    invalid_ticket = {
        "ticket_id": "TICKET-INVALID-001",
        "customer_email": "not-an-email-address",  # Fails regex pattern
        "query_text": "A",                          # Fails minLength 3
        "clean_text": "a",
        "predicted_intent": "completely_invalid_intent_xyz",  # Fails 77-enum restriction
        "intent_confidence": 1.5,                  # Fails maximum 1.0
        "model_used": "InvalidModelType",          # Fails enum
        "priority": "ExtremeUrgentTier",           # Fails priority enum
        "priority_reason": "Test",
        "status": "NotAStatus",                    # Fails status enum
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None)
    }

    schema_rejected = False
    try:
        db[COLLECTION_TICKETS].insert_one(invalid_ticket)
    except (OperationFailure, WriteError) as e:
        schema_rejected = True
        print(f"  -> [PASS] MongoDB actively rejected invalid document as expected!")
        print(f"     Server error message: {e.details.get('errmsg', str(e))}")

    assert schema_rejected, "MongoDB failed to enforce $jsonSchema! Invalid document was accepted."

    # 6. Real Ticket CRUD Operations Lifecycle
    print("\n[STEP 6] Executing Ticket CRUD Operations on Live MongoDB...")
    test_ticket_data = {
        "customer_email": "live.test.user@bank.com",
        "query_text": "My debit card was swallowed by an ATM in Central Station",
        "clean_text": "debit card swallow atm central station",
        "predicted_intent": "card_swallowed",
        "intent_confidence": 0.9620,
        "model_used": "LinearSVM",
        "priority": "Critical",
        "priority_reason": "Security/Lockout: Intent 'card_swallowed' mapped to Critical tier.",
        "status": "Open",
        "needs_manual_review": False,
        "recommended_solutions": [
            {
                "kb_id": "KB-CARD-001",
                "title": "Immediate Protocol for Lost, Stolen, or Compromised Cards",
                "category_group": "Card Services",
                "similarity_score": 0.4850
            }
        ]
    }

    # CREATE
    created_id = ticket_repo.create_ticket(test_ticket_data)
    print(f"  -> [CREATE] Created live ticket with ID: {created_id}")
    assert created_id.startswith("TICKET-")

    # READ
    read_doc = ticket_repo.get_ticket_by_id(created_id)
    assert read_doc is not None
    assert read_doc["customer_email"] == "live.test.user@bank.com"
    assert read_doc["priority"] == "Critical"
    print(f"  -> [READ] Successfully retrieved ticket: intent='{read_doc['predicted_intent']}', priority='{read_doc['priority']}'")

    # UPDATE
    update_ok = ticket_repo.update_ticket_status(
        ticket_id=created_id,
        status="In Progress",
        assigned_agent="Agent_Live_Verifier",
        resolution_notes="Contacting ATM maintenance team to retrieve machine logs."
    )
    assert update_ok is True
    updated_doc = ticket_repo.get_ticket_by_id(created_id)
    assert updated_doc["status"] == "In Progress"
    assert updated_doc["assigned_agent"] == "Agent_Live_Verifier"
    print(f"  -> [UPDATE] Ticket status updated to '{updated_doc['status']}' assigned to '{updated_doc['assigned_agent']}'")

    # 7. Execute Aggregation Pipelines against Live MongoDB
    print("\n[STEP 7] Executing Aggregation Pipelines against Live MongoDB Data...")
    
    # Insert 3 additional temporary tickets to test aggregation diversity
    extra_tickets = [
        {
            "customer_email": "t2@bank.com",
            "query_text": "Pin blocked please help",
            "clean_text": "pin block help",
            "predicted_intent": "pin_blocked",
            "intent_confidence": 0.9100,
            "model_used": "LogisticRegression",
            "priority": "High",
            "priority_reason": "Access block",
            "status": "Resolved",
            "needs_manual_review": False
        },
        {
            "customer_email": "t3@bank.com",
            "query_text": "What is the fee for wire transfer?",
            "clean_text": "fee wire transfer",
            "predicted_intent": "transfer_fee_charged",
            "intent_confidence": 0.8400,
            "model_used": "LinearSVM",
            "priority": "Medium",
            "priority_reason": "Standard fee query",
            "status": "Open",
            "needs_manual_review": False
        },
        {
            "customer_email": "t4@bank.com",
            "query_text": "Unknown vague request",
            "clean_text": "vague request",
            "predicted_intent": "exchange_rate",
            "intent_confidence": 0.2800,
            "model_used": "LinearSVM",
            "priority": "Low",
            "priority_reason": "Low confidence query",
            "status": "Open",
            "needs_manual_review": True
        }
    ]
    created_extra_ids = [ticket_repo.create_ticket(t) for t in extra_tickets]
    all_temp_ids = [created_id] + created_extra_ids

    # Pipeline 1: Intent Volume & Confidence
    p1 = get_intent_volume_and_confidence(db, limit=5)
    print(f"  -> [AGG 1 - Top Intents ({len(p1)})]:")
    for row in p1:
        print(f"     * Intent: {row['intent']:<30} Total: {row['total_tickets']} | Avg Conf: {row['avg_confidence']} | Res Rate: {row['resolution_rate_pct']}%")

    # Pipeline 2: Priority SLA Breakdown
    p2 = get_priority_sla_breakdown(db)
    print(f"  -> [AGG 2 - Priority Breakdown ({len(p2)})]:")
    for row in p2:
        print(f"     * Priority: {row['priority']:<10} Total: {row['total']} | Open: {row['open_count']} | Resolved: {row['resolved_count']}")

    # Pipeline 3: Model Comparison
    p3 = get_model_performance_in_production(db)
    print(f"  -> [AGG 3 - Model Comparison ({len(p3)})]:")
    for row in p3:
        print(f"     * Model: {row['model_name']:<25} Count: {row['ticket_count']} | Mean Conf: {row['mean_confidence']} | Review Rate: {row['manual_review_rate_pct']}%")

    # Pipeline 4: Executive KPIs ($facet)
    kpis = get_executive_kpis(db)
    print(f"  -> [AGG 4 - Executive KPIs ($facet)]:")
    print(f"     * Total Tickets: {kpis['total_tickets']}")
    print(f"     * Open Tickets: {kpis['open_tickets']}")
    print(f"     * Resolved Tickets: {kpis['resolved_tickets']}")
    print(f"     * Resolution Rate: {kpis['resolution_rate_pct']}%")
    print(f"     * Critical Security Tickets: {kpis['critical_tickets']}")
    print(f"     * Manual Review Flags: {kpis['manual_review_tickets']}")

    # Pipeline 5: Daily Volume Trend
    p5 = get_daily_volume_trend(db)
    print(f"  -> [AGG 5 - Daily Volume Trend ({len(p5)} entries)]")

    # 8. Cleanup Temporary Verification Tickets
    print("\n[STEP 8] Cleaning up Temporary Verification Tickets...")
    deleted_count = 0
    for tid in all_temp_ids:
        if ticket_repo.delete_ticket(tid):
            deleted_count += 1
    print(f"  -> [PASS] Cleaned up {deleted_count} temporary test tickets.")
    print("  -> Knowledge Base collection preserved with all 20 curated banking SOPs.")

    print("\n" + "=" * 70)
    print("ALL LIVE MONGODB VERIFICATION CHECKS PASSED (100% SUCCESS)")
    print("=" * 70)


if __name__ == "__main__":
    run_live_verification()
