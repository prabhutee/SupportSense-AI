"""
Index Definitions and Creation for SupportSense MongoDB Collections.

Creates single-field, compound, unique, and full-text indexes to optimize:
- Support agent ticket queue filtering and priority sorting.
- Customer history lookups.
- Full-text keyword searches across queries and KB articles.
- High-performance aggregation pipelines.
"""

import logging
from typing import Dict, List, Any
from pymongo import ASCENDING, DESCENDING, TEXT, IndexModel
from pymongo.database import Database
from config.db_config import COLLECTION_TICKETS, COLLECTION_KB

logger = logging.getLogger(__name__)


def create_ticket_indexes(db: Database) -> List[str]:
    """
    Creates indexes on the tickets collection:
    1. Unique index on ticket_id.
    2. Compound index on (status, priority, created_at) for agent queue queries.
    3. Single index on predicted_intent for analytics aggregations.
    4. Single index on customer_email for customer history lookup.
    5. Text index on query_text for keyword search.
    """
    col = db[COLLECTION_TICKETS]
    models = [
        IndexModel([("ticket_id", ASCENDING)], unique=True, name="idx_unique_ticket_id"),
        IndexModel(
            [("status", ASCENDING), ("priority", ASCENDING), ("created_at", DESCENDING)],
            name="idx_status_priority_created"
        ),
        IndexModel([("predicted_intent", ASCENDING)], name="idx_predicted_intent"),
        IndexModel([("customer_email", ASCENDING)], name="idx_customer_email"),
        IndexModel([("query_text", TEXT)], name="idx_query_text_search")
    ]
    created = col.create_indexes(models)
    logger.info("Created %d indexes on '%s'", len(created), COLLECTION_TICKETS)
    return created


def create_kb_indexes(db: Database) -> List[str]:
    """
    Creates indexes on the knowledge_base collection:
    1. Unique index on kb_id.
    2. Single index on category_group for candidate filtering.
    3. Compound text index on title and content with relevance weighting.
    """
    col = db[COLLECTION_KB]
    models = [
        IndexModel([("kb_id", ASCENDING)], unique=True, name="idx_unique_kb_id"),
        IndexModel([("category_group", ASCENDING)], name="idx_category_group"),
        IndexModel(
            [("title", TEXT), ("content", TEXT)],
            weights={"title": 10, "content": 5},
            name="idx_kb_text_search"
        )
    ]
    created = col.create_indexes(models)
    logger.info("Created %d indexes on '%s'", len(created), COLLECTION_KB)
    return created


def create_all_indexes(db: Database) -> Dict[str, List[str]]:
    """Applies all index specifications across tickets and knowledge_base collections."""
    return {
        COLLECTION_TICKETS: create_ticket_indexes(db),
        COLLECTION_KB: create_kb_indexes(db)
    }
