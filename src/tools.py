import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import sqlite3
from chromadb import Client
from chromadb.utils import embedding_functions
from langchain_core.tools import tool


class DatabaseManager:
    """Handles SQLite database with CSV loading for local demo."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.conn: Optional[sqlite3.Connection] = None
        self._initialized = False

    def connect(self):
        """Establish SQLite connection and load CSVs if not already done."""
        if self.conn is None:
            # Use in-memory database for demo
            self.conn = sqlite3.connect(":memory:")
            # Enable column access by name
            self.conn.row_factory = sqlite3.Row
            # Load CSVs into database
            self._load_csvs()
        return self.conn

    def _load_csvs(self):
        """Load all CSV files from data/ folder into SQLite tables."""
        if self._initialized:
            return

        csv_files = [
            "available_inventory_report.csv",
            "study_level_enrollment_report.csv",
            "lot_status_report.csv",
            "re-evaluation.csv",
            "rim.csv",
            "ip_shipping_timelines_report.csv",
            "allocated_materials_to_orders.csv",
        ]

        for file_name in csv_files:
            csv_path = self.data_dir / file_name
            if not csv_path.exists():
                print(f"WARNING: {file_name} not found at {csv_path}. Skipping.")
                continue

            try:
                # Read CSV
                df = pd.read_csv(csv_path)
                
                # Table name: remove .csv extension
                table_name = file_name.replace(".csv", "").replace("-", "_")
                
                # SQLite doesn't like some special chars in table names
                table_name = table_name.replace(" ", "_")
                
                # Write to SQLite (preserves column names with spaces)
                df.to_sql(table_name, self.conn, index=False, if_exists="replace")
                print(f"✓ Loaded {file_name} → table '{table_name}' ({len(df)} rows)")
            except Exception as e:
                print(f"ERROR loading {file_name}: {e}")

        self._initialized = True
        print(f"\n✓ Database initialized with {len(csv_files)} tables\n")

    def execute_query(self, query: str) -> Dict:
        """
        Execute SQL with error capture.
        Converts PostgreSQL-specific syntax to SQLite where needed.

        Returns:
            {
                "success": bool,
                "data": List[Dict],
                "error": Optional[str],
                "suggestion": Optional[str]
            }
        """
        conn = self.connect()

        # Convert PostgreSQL date functions to SQLite
        # EXTRACT(DAY FROM (date1 - date2)) → (julianday(date1) - julianday(date2))
        # CURRENT_DATE → date('now')
        # INTERVAL '30 days' → handled via date arithmetic
        query = self._convert_postgres_to_sqlite(query)

        try:
            cursor = conn.cursor()
            cursor.execute(query)
            
            # Check if query returns rows
            if cursor.description is None:
                conn.commit()
                return {
                    "success": True,
                    "data": [],
                    "error": None,
                    "suggestion": None,
                    "message": "Query executed successfully (no result set).",
                }

            # Fetch results as dictionaries
            rows = cursor.fetchall()
            data = [dict(row) for row in rows]
            
            return {
                "success": True,
                "data": data,
                "error": None,
                "suggestion": None,
            }
        except sqlite3.Error as e:
            error_msg = str(e)
            suggestion: Optional[str] = None

            # Try to extract helpful suggestions from SQLite errors
            if "no such column" in error_msg.lower():
                suggestion = "Check column name spelling and ensure it's quoted if it contains spaces."

            return {
                "success": False,
                "data": [],
                "error": error_msg,
                "suggestion": suggestion,
            }

    def _convert_postgres_to_sqlite(self, query: str) -> str:
        """Convert PostgreSQL-specific syntax to SQLite-compatible syntax."""
        # Replace CURRENT_DATE with date('now')
        query = query.replace("CURRENT_DATE", "date('now')")
        
        # Replace EXTRACT(DAY FROM (date1 - date2)) with SQLite date difference
        # This is a simplified conversion - may need refinement for complex cases
        import re
        
        # Handle INTERVAL 'X days' in date arithmetic
        # PostgreSQL: date + INTERVAL '30 days'
        # SQLite: date(date, '+30 days')
        def replace_interval(match):
            days = match.group(1)
            date_expr = match.group(2)
            return f"date({date_expr}, '+{days} days')"
        
        # Pattern: date_expr + INTERVAL 'X days'
        query = re.sub(
            r"([^,\(\)]+)\s*\+\s*INTERVAL\s+'(\d+)\s+days'",
            lambda m: f"date({m.group(1).strip()}, '+{m.group(2)} days')",
            query,
            flags=re.IGNORECASE
        )
        
        # Pattern: date_expr - INTERVAL 'X days'
        query = re.sub(
            r"([^,\(\)]+)\s*-\s*INTERVAL\s+'(\d+)\s+days'",
            lambda m: f"date({m.group(1).strip()}, '-{m.group(2)} days')",
            query,
            flags=re.IGNORECASE
        )
        
        # Handle EXTRACT(DAY FROM (date1 - date2))
        def replace_extract_day(match):
            date1 = match.group(1)
            date2 = match.group(2)
            return f"ROUND((julianday({date1}) - julianday({date2})))"
        
        query = re.sub(
            r"EXTRACT\s*\(\s*DAY\s+FROM\s*\(\s*([^)]+)\s*-\s*([^)]+)\s*\)\s*\)",
            replace_extract_day,
            query,
            flags=re.IGNORECASE
        )
        
        return query

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


# Global DB instance used by tools - uses SQLite with CSV loading for local demo
db = DatabaseManager(data_dir="data")


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
    try:
        client = Client()
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction()

        # Load trial registry from DB (SQLite uses quoted column names)
        query = 'SELECT DISTINCT "Trial Name" FROM available_inventory_report'
        result = db.execute_query(query)

        if not result["success"]:
            return f"ERROR: Could not load trial registry - {result.get('error', 'Unknown error')}"

        trials: List[str] = [row["Trial Name"] for row in result["data"] if row.get("Trial Name")]
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
    except Exception as e:
        return f"ERROR: Vector search failed - {str(e)}"


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



