"""
Database Configuration for SupportSense.

Loads MongoDB connection parameters securely from environment variables or .env file.
Credentials are never hardcoded.
"""

import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# MongoDB URI and Database Name
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "supportsense_db")

# Timeouts and pooling configuration
MONGODB_TIMEOUT_MS = int(os.getenv("MONGODB_TIMEOUT_MS", "3000"))
MONGODB_MAX_POOL_SIZE = int(os.getenv("MONGODB_MAX_POOL_SIZE", "50"))
MONGODB_MIN_POOL_SIZE = int(os.getenv("MONGODB_MIN_POOL_SIZE", "5"))

# Collection names
COLLECTION_TICKETS = "tickets"
COLLECTION_KB = "knowledge_base"
