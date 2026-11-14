"""
Ticket Repository for SupportSense.

Handles all CRUD operations for customer support tickets using PyMongo:
- Create: Validates and inserts new ticket documents with unique UUIDs.
- Read: Retrieves tickets by ID, with filters (status, priority, intent), and pagination.
- Update: Modifies ticket status, assigns support agents, and logs resolution notes.
- Delete: Removes tickets or archives them.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pymongo.database import Database
from config.db_config import COLLECTION_TICKETS


def _utc_now() -> datetime:
    """Returns current naive UTC datetime for standard BSON compatibility."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TicketRepository:
    """Repository handling all CRUD interactions with the tickets collection."""

    def __init__(self, db: Database):
        self.db = db
        self.collection = db[COLLECTION_TICKETS]

    def _format_doc(self, doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Formats BSON document for clean Python consumption (converts ObjectId to string)."""
        if not doc:
            return None
        doc = dict(doc)
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc

    def create_ticket(self, data: Dict[str, Any]) -> str:
        """
        Inserts a new ticket document into MongoDB.
        Generates ticket_id if not present, and sets created_at/updated_at timestamps.
        """
        doc = dict(data)
        if "ticket_id" not in doc or not doc["ticket_id"]:
            doc["ticket_id"] = f"TICKET-{uuid.uuid4().hex[:8].upper()}"

        now = _utc_now()
        if "created_at" not in doc:
            doc["created_at"] = now
        if "updated_at" not in doc:
            doc["updated_at"] = now
        if "status" not in doc:
            doc["status"] = "Open"
        if "recommended_solutions" not in doc:
            doc["recommended_solutions"] = []
        if "needs_manual_review" not in doc:
            doc["needs_manual_review"] = False
        if "assigned_agent" not in doc:
            doc["assigned_agent"] = None
        if "resolution_notes" not in doc:
            doc["resolution_notes"] = None

        self.collection.insert_one(doc)
        return doc["ticket_id"]

    def get_ticket_by_id(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single ticket by its unique ticket_id."""
        doc = self.collection.find_one({"ticket_id": ticket_id})
        return self._format_doc(doc)

    def list_tickets(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        predicted_intent: Optional[str] = None,
        needs_manual_review: Optional[bool] = None,
        search_text: Optional[str] = None,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Queries tickets with multi-field filtering, text search, and pagination.
        Sorted by created_at descending.
        """
        filter_query: Dict[str, Any] = {}

        if status:
            filter_query["status"] = status
        if priority:
            filter_query["priority"] = priority
        if predicted_intent:
            filter_query["predicted_intent"] = predicted_intent
        if needs_manual_review is not None:
            filter_query["needs_manual_review"] = needs_manual_review
        if search_text:
            filter_query["$text"] = {"$search": search_text}

        cursor = self.collection.find(filter_query).sort("created_at", -1).skip(skip).limit(limit)
        return [self._format_doc(doc) for doc in cursor]

    def update_ticket_status(
        self,
        ticket_id: str,
        status: str,
        assigned_agent: Optional[str] = None,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """
        Updates the operational status of a ticket, assigned agent, and resolution notes.
        """
        allowed_statuses = {"Open", "In Progress", "Resolved", "Closed"}
        if status not in allowed_statuses:
            raise ValueError(f"Invalid status '{status}'. Must be one of {allowed_statuses}")

        update_fields: Dict[str, Any] = {
            "status": status,
            "updated_at": _utc_now()
        }

        if assigned_agent is not None:
            update_fields["assigned_agent"] = assigned_agent
        if resolution_notes is not None:
            update_fields["resolution_notes"] = resolution_notes

        result = self.collection.update_one(
            {"ticket_id": ticket_id},
            {"$set": update_fields}
        )
        return result.modified_count > 0

    def delete_ticket(self, ticket_id: str) -> bool:
        """Deletes a ticket by ticket_id."""
        result = self.collection.delete_one({"ticket_id": ticket_id})
        return result.deleted_count > 0

    def count_tickets(self, filter_query: Optional[Dict[str, Any]] = None) -> int:
        """Returns the total number of tickets matching a filter."""
        return self.collection.count_documents(filter_query or {})
