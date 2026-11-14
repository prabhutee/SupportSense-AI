
"""
Database Configuration for SupportSense.
Supports local .env and Streamlit Cloud secrets.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_setting(name, default):
    # Local environment variables take priority.
    value = os.getenv(name)
    if value:
        return value

    # Fall back to Streamlit Cloud secrets.
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


MONGODB_URI = get_setting(
    "MONGODB_URI",
    "mongodb://localhost:27017"
)

MONGODB_DB_NAME = get_setting(
    "MONGODB_DB_NAME",
    "supportsense_db"
)

MONGODB_TIMEOUT_MS = int(get_setting("MONGODB_TIMEOUT_MS", "3000"))
MONGODB_MAX_POOL_SIZE = int(get_setting("MONGODB_MAX_POOL_SIZE", "50"))
MONGODB_MIN_POOL_SIZE = int(get_setting("MONGODB_MIN_POOL_SIZE", "5"))

COLLECTION_TICKETS = "tickets"
COLLECTION_KB = "knowledge_base"
