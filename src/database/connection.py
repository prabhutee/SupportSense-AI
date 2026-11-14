"""
MongoDB Connection Manager for SupportSense using PyMongo.

Features:
- Singleton connection pooling to avoid socket exhaustion.
- Resilient error handling and diagnostic connection checks.
- Optional mock database creation via mongomock for unit tests.
- Zero hardcoded credentials; driven by config/db_config.py.
"""

import sys
import logging
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

import mongomock

from config.db_config import (
    MONGODB_URI,
    MONGODB_DB_NAME,
    MONGODB_TIMEOUT_MS,
    MONGODB_MAX_POOL_SIZE,
    MONGODB_MIN_POOL_SIZE
)

logger = logging.getLogger(__name__)


class MongoDBConnection:
    """Singleton MongoDB Client and Database Manager."""

    _instance: Optional["MongoDBConnection"] = None
    _client: Optional[MongoClient] = None
    _mock_client: Optional[mongomock.MongoClient] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBConnection, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_client(cls, uri: Optional[str] = None, timeout_ms: Optional[int] = None) -> MongoClient:
        """Returns the shared PyMongo client instance."""
        target_uri = uri or MONGODB_URI
        target_timeout = timeout_ms if timeout_ms is not None else MONGODB_TIMEOUT_MS

        if cls._client is None:
            logger.info("Initializing PyMongo MongoClient with URI: %s", target_uri)
            cls._client = MongoClient(
                target_uri,
                serverSelectionTimeoutMS=target_timeout,
                connectTimeoutMS=target_timeout,
                maxPoolSize=MONGODB_MAX_POOL_SIZE,
                minPoolSize=MONGODB_MIN_POOL_SIZE
            )
        return cls._client

    @classmethod
    def get_database(
        cls,
        mock: bool = False,
        uri: Optional[str] = None,
        db_name: Optional[str] = None
    ) -> Database:
        """
        Returns a MongoDB Database object.
        If mock=True, returns an in-memory mongomock database for unit tests.
        """
        target_db_name = db_name or MONGODB_DB_NAME

        if mock:
            if cls._mock_client is None:
                cls._mock_client = mongomock.MongoClient()
            return cls._mock_client[target_db_name]

        client = cls.get_client(uri=uri)
        return client[target_db_name]

    @classmethod
    def ping(cls, uri: Optional[str] = None, timeout_ms: int = 1500) -> bool:
        """Pings the MongoDB server to verify live connectivity."""
        try:
            client = cls.get_client(uri=uri, timeout_ms=timeout_ms)
            client.admin.command("ping")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            logger.debug("MongoDB ping failed: %s", e)
            return False

    @classmethod
    def close(cls) -> None:
        """Closes all open MongoDB client connections."""
        if cls._client is not None:
            cls._client.close()
            cls._client = None
            logger.info("PyMongo connection closed.")
        if cls._mock_client is not None:
            cls._mock_client.close()
            cls._mock_client = None


def get_db(mock: bool = False) -> Database:
    """Convenience helper to obtain active or mocked Database instance."""
    return MongoDBConnection.get_database(mock=mock)
