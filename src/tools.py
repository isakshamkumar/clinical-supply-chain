import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import psycopg2
from chromadb import Client
from chromadb.utils import embedding_functions
from psycopg2.extras import RealDictCursor
from langchain_core.tools import tool


DB_CONFIG: Dict[str, Optional[str]] = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "clinical_supply_chain"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT", "5432"),
}


class DatabaseManager:
    """Handles all PostgreSQL interactions with basic error handling."""

    def __init__(self, config: Dict[str, Optional[str]]):
        self.config = config
        self.conn: Optional[psycopg2.extensions.connection] = None

    def connect(self):
        """Establish database connection."""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(**self.config)
        return self.conn

    def execute_query(self, query: str) -> Dict:
        """
        Execute SQL with error capture.

        Returns:
            {
                "success": bool,
                "data": List[Dict],
                "error": Optional[str],
                "suggestion": Optional[str]
            }
        """
        conn = self.connect()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                if cursor.description is None:
                    conn.commit()
                    return {
                        "success": True,
                        "data": [],
                        "error": None,
                        "suggestion": None,
                        "message": "Query executed successfully (no result set).",
                    }

                results = cursor.fetchall()
                return {
                    "success": True,
                    "data": [dict(row) for row in results],
                    "error": None,
                    "suggestion": None,
                }
        except psycopg2.Error as e:
            error_msg = str(e)
            suggestion: Optional[str] = None

            if "does not exist" in error_msg and "Did you mean" in error_msg:
                # crude extraction of Postgres suggestion
                parts = error_msg.split("Did you mean")
                if len(parts) > 1:
                    suggestion = parts[1].strip()

            return {
                "success": False,
                "data": [],
                "error": error_msg,
                "suggestion": suggestion,
            }

    def close(self):
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()


# Global DB instance used by tools
db = DatabaseManager(DB_CONFIG)


@tool
def run_sql_query(query: str) -> Dict:
    """
    Execute PostgreSQL query with structured error handling.

    Args:
        query: Valid PostgreSQL statement (typically SELECT).
    """
    return db.execute_query(query)


@tool
def calculate_runway(total_stock: int, monthly_rate: float) -> float:
    """
    Calculate weeks of supply coverage.

    Args:
        total_stock: Current inventory quantity
        monthly_rate: Patient enrollment rate per month

    Returns:
        Weeks until stockout (float). Returns +inf if monthly_rate <= 0.
    """
    if monthly_rate <= 0:
        return float("inf")

    weekly_rate = monthly_rate / 4
    return total_stock / weekly_rate


@tool
def search_trial_by_name(fuzzy_name: str) -> str:
    """
    Find exact trial ID using vector semantic search.

    Args:
        fuzzy_name: User's natural language trial reference

    Returns:
        Exact trial_id from database (e.g., "CT-2004-PSX") or an error string.
    """
    client = Client()
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction()

    # Load trial registry from DB
    query = 'SELECT DISTINCT "Trial Name" FROM available_inventory_report'
    result = db.execute_query(query)

    if not result["success"]:
        return "ERROR: Could not load trial registry"

    trials: List[str] = [row["Trial Name"] for row in result["data"]]
    if not trials:
        return "ERROR: No trials found in registry"

    # Create ephemeral collection for this lookup
    collection = client.create_collection(
        name="temp_trials",
        embedding_function=embedding_fn,
    )

    collection.add(
        documents=trials,
        ids=[f"trial_{i}" for i in range(len(trials))],
    )

    results = collection.query(
        query_texts=[fuzzy_name],
        n_results=1,
    )

    docs = results.get("documents") or []
    if docs and docs[0]:
        return docs[0][0]

    return "ERROR: No matching trial found"


@tool
def send_alert(payload: Dict, channels: List[str]) -> bool:
    """
    Send formatted alert to Slack/Email (stdout simulation).

    In production, this would integrate with Slack, email, or ticketing APIs.
    """
    banner = "=" * 80
    print(f"\n{banner}")
    print("ALERT SENT TO:", ", ".join(channels))
    print(banner)
    print(json.dumps(payload, indent=2, default=str))
    print(banner + "\n")
    return True


__all__ = [
    "db",
    "run_sql_query",
    "calculate_runway",
    "search_trial_by_name",
    "send_alert",
]



