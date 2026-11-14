"""
MongoDB $jsonSchema Validators for SupportSense.

Enforces strict document structure, data types, value constraints, and enums
directly at the MongoDB database level using PyMongo.
"""

import json
import os
from typing import Dict, Any
from pymongo.database import Database
from config.db_config import COLLECTION_TICKETS, COLLECTION_KB

# Load canonical categories for exact intent validation
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CATEGORIES_PATH = os.path.join(BASE_DIR, "data", "raw", "categories.json")

CANONICAL_INTENTS = []
if os.path.exists(CATEGORIES_PATH):
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        CANONICAL_INTENTS = json.load(f)

# Fallback in case raw categories file is accessed in isolated test
if not CANONICAL_INTENTS:
    CANONICAL_INTENTS = ["lost_or_stolen_card", "activate_my_card", "exchange_rate", "pin_blocked"]


# ============================================================================
# 1. TICKETS COLLECTION SCHEMA VALIDATOR
# ============================================================================
TICKETS_VALIDATOR: Dict[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "ticket_id",
            "customer_email",
            "query_text",
            "clean_text",
            "predicted_intent",
            "intent_confidence",
            "model_used",
            "priority",
            "priority_reason",
            "status",
            "created_at"
        ],
        "properties": {
            "ticket_id": {
                "bsonType": "string",
                "description": "Unique UUID or formatted ticket identifier (e.g. TICKET-2026-XXXX)."
            },
            "customer_email": {
                "bsonType": "string",
                "pattern": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
                "description": "Standard RFC 5322 compliant customer email address."
            },
            "query_text": {
                "bsonType": "string",
                "minLength": 3,
                "description": "Original raw text inquiry submitted by the customer."
            },
            "clean_text": {
                "bsonType": "string",
                "description": "Normalized and preprocessed text used for vectorization."
            },
            "predicted_intent": {
                "bsonType": "string",
                "enum": CANONICAL_INTENTS,
                "description": "Canonical BANKING77 intent predicted by the classifier."
            },
            "intent_confidence": {
                "bsonType": "double",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Posterior calibrated confidence probability (0.0 to 1.0)."
            },
            "model_used": {
                "bsonType": "string",
                "enum": ["LogisticRegression", "LinearSVM"],
                "description": "Classification model utilized for intent inference."
            },
            "priority": {
                "bsonType": "string",
                "enum": ["Critical", "High", "Medium", "Low"],
                "description": "Operational priority assigned by the business rules engine."
            },
            "priority_reason": {
                "bsonType": "string",
                "description": "Human-auditable explanation of how priority was assigned."
            },
            "status": {
                "bsonType": "string",
                "enum": ["Open", "In Progress", "Resolved", "Closed"],
                "description": "Ticket lifecycle workflow status."
            },
            "assigned_agent": {
                "bsonType": ["string", "null"],
                "description": "Support agent assigned to the ticket, or null if unassigned."
            },
            "resolution_notes": {
                "bsonType": ["string", "null"],
                "description": "Agent resolution summary and closure notes."
            },
            "recommended_solutions": {
                "bsonType": "array",
                "description": "Ranked Knowledge Base SOP recommendations.",
                "items": {
                    "bsonType": "object",
                    "required": ["kb_id", "title", "similarity_score"],
                    "properties": {
                        "kb_id": { "bsonType": "string" },
                        "title": { "bsonType": "string" },
                        "category_group": { "bsonType": "string" },
                        "similarity_score": { "bsonType": "double", "minimum": 0.0, "maximum": 1.0 },
                        "summary": { "bsonType": "string" }
                    }
                }
            },
            "needs_manual_review": {
                "bsonType": "bool",
                "description": "Flag indicating low AI confidence (<0.40) or missing recommendations."
            },
            "manual_review_reason": {
                "bsonType": ["string", "null"],
                "description": "Explanation for manual review escalation."
            },
            "created_at": {
                "bsonType": "date",
                "description": "Timestamp when ticket was created."
            },
            "updated_at": {
                "bsonType": "date",
                "description": "Timestamp of last ticket modification."
            }
        }
    }
}


# ============================================================================
# 2. KNOWLEDGE BASE COLLECTION SCHEMA VALIDATOR
# ============================================================================
KB_VALIDATOR: Dict[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "kb_id",
            "title",
            "category_group",
            "content",
            "relevant_intents",
            "created_at"
        ],
        "properties": {
            "kb_id": {
                "bsonType": "string",
                "description": "Unique knowledge base identifier (e.g. KB-CARD-001)."
            },
            "title": {
                "bsonType": "string",
                "minLength": 5,
                "description": "Descriptive title of the troubleshooting SOP."
            },
            "category_group": {
                "bsonType": "string",
                "enum": [
                    "Card Services",
                    "Transfers & Payments",
                    "Account Security",
                    "Top-Ups & Deposits",
                    "Disputes & Fees"
                ],
                "description": "Functional banking operational domain."
            },
            "summary": {
                "bsonType": "string",
                "description": "High-level summary of the issue."
            },
            "content": {
                "bsonType": "string",
                "minLength": 10,
                "description": "Detailed explanation and operational context."
            },
            "relevant_intents": {
                "bsonType": "array",
                "description": "Canonical BANKING77 intents addressed by this SOP.",
                "items": {
                    "bsonType": "string"
                }
            },
            "tags": {
                "bsonType": "array",
                "items": { "bsonType": "string" }
            },
            "resolution_steps": {
                "bsonType": "array",
                "description": "Actionable sequential steps to resolve the customer inquiry.",
                "items": { "bsonType": "string" }
            },
            "disclaimer": {
                "bsonType": "string",
                "description": "Explicit educational/demo content disclaimer."
            },
            "helpful_count": {
                "bsonType": "int",
                "minimum": 0,
                "description": "Number of upvotes or successful resolutions using this article."
            },
            "created_at": {
                "bsonType": "date"
            },
            "updated_at": {
                "bsonType": "date"
            }
        }
    }
}


def apply_schema_validators(db: Database) -> Dict[str, bool]:
    """
    Applies the $jsonSchema validators to the tickets and knowledge_base collections.
    If the collection exists, updates validation with collMod; otherwise creates it.
    """
    results = {}
    specs = [
        (COLLECTION_TICKETS, TICKETS_VALIDATOR),
        (COLLECTION_KB, KB_VALIDATOR)
    ]

    existing_collections = db.list_collection_names()

    for col_name, validator in specs:
        try:
            if col_name in existing_collections:
                db.command(
                    "collMod",
                    col_name,
                    validator=validator,
                    validationLevel="moderate",
                    validationAction="error"
                )
                results[col_name] = True
            else:
                db.create_collection(
                    col_name,
                    validator=validator,
                    validationLevel="strict",
                    validationAction="error"
                )
                results[col_name] = True
        except Exception as e:
            # mongomock or restricted users might not support collMod command
            results[col_name] = False

    return results
