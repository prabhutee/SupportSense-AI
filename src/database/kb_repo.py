"""
Knowledge Base Repository for SupportSense.

Handles all CRUD operations for curated troubleshooting articles using PyMongo:
- Create: Inserts new banking SOP articles.
- Read: Retrieves articles by kb_id, category_group, or full-text search.
- Update: Modifies content, tags, resolution steps, or increments helpfulness votes.
- Delete: Removes retired articles.
- Seed: Ingests data/seed/kb_articles.json into MongoDB with upsert logic.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pymongo import ReplaceOne
from pymongo.database import Database
from config.db_config import COLLECTION_KB

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_SEED_PATH = os.path.join(BASE_DIR, "data", "seed", "kb_articles.json")


def _utc_now() -> datetime:
    """Returns current naive UTC datetime for standard BSON compatibility."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class KBRepository:
    """Repository handling all CRUD interactions with the knowledge_base collection."""

    def __init__(self, db: Database):
        self.db = db
        self.collection = db[COLLECTION_KB]

    def _format_doc(self, doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Formats BSON document for clean Python consumption."""
        if not doc:
            return None
        doc = dict(doc)
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc

    def create_article(self, data: Dict[str, Any]) -> str:
        """Inserts a new knowledge base article."""
        doc = dict(data)
        if "kb_id" not in doc or not doc["kb_id"]:
            raise ValueError("Article must contain a valid 'kb_id'.")

        now = _utc_now()
        if "created_at" not in doc:
            doc["created_at"] = now
        if "updated_at" not in doc:
            doc["updated_at"] = now
        if "helpful_count" not in doc:
            doc["helpful_count"] = 0
        if "tags" not in doc:
            doc["tags"] = []
        if "resolution_steps" not in doc:
            doc["resolution_steps"] = []

        self.collection.insert_one(doc)
        return doc["kb_id"]

    def get_article_by_id(self, kb_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an article by its unique kb_id."""
        doc = self.collection.find_one({"kb_id": kb_id})
        return self._format_doc(doc)

    def list_articles(
        self,
        category_group: Optional[str] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieves articles, optionally filtered by category_group."""
        filter_query: Dict[str, Any] = {}
        if category_group:
            filter_query["category_group"] = category_group

        cursor = self.collection.find(filter_query).sort("kb_id", 1).skip(skip).limit(limit)
        return [self._format_doc(doc) for doc in cursor]

    def update_article(self, kb_id: str, updates: Dict[str, Any]) -> bool:
        """Updates article fields and updates the updated_at timestamp."""
        update_doc = dict(updates)
        update_doc["updated_at"] = _utc_now()
        # Protect immutable ID
        update_doc.pop("kb_id", None)
        update_doc.pop("_id", None)

        result = self.collection.update_one(
            {"kb_id": kb_id},
            {"$set": update_doc}
        )
        return result.modified_count > 0

    def delete_article(self, kb_id: str) -> bool:
        """Deletes an article by kb_id."""
        result = self.collection.delete_one({"kb_id": kb_id})
        return result.deleted_count > 0

    def increment_helpful_count(self, kb_id: str) -> bool:
        """Increments the helpfulness counter for an article using $inc."""
        result = self.collection.update_one(
            {"kb_id": kb_id},
            {"$inc": {"helpful_count": 1}, "$set": {"updated_at": _utc_now()}}
        )
        return result.modified_count > 0

    def search_articles_text(self, query_str: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Searches articles using the MongoDB compound text index."""
        cursor = self.collection.find(
            {"$text": {"$search": query_str}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        return [self._format_doc(doc) for doc in cursor]

    def seed_kb_from_json(self, json_path: Optional[str] = None) -> int:
        """
        Seeds the knowledge_base collection from JSON file using upsert (replace_one).
        Ensures idempotency across both live PyMongo and mongomock.
        """
        target_path = json_path or DEFAULT_SEED_PATH
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Knowledge base seed file not found: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        now = _utc_now()
        count = 0
        for a in articles:
            doc = dict(a)
            if "created_at" not in doc or not isinstance(doc["created_at"], datetime):
                doc["created_at"] = now
            if "updated_at" not in doc or not isinstance(doc["updated_at"], datetime):
                doc["updated_at"] = now
            if "helpful_count" not in doc:
                doc["helpful_count"] = 0

            result = self.collection.replace_one(
                {"kb_id": doc["kb_id"]},
                doc,
                upsert=True
            )
            if result.upserted_id is not None or result.matched_count > 0:
                count += 1

        return count
