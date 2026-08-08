"""
MediLex India — SQLite logging module.

Stores anonymised case metadata and AI-generated protocol outputs.
No PII is persisted.  Session rows link to the hospital's own MLC
register via case_reference_number only.

Tech stack:  Python's built-in sqlite3 — chosen because:
  - Zero deployment overhead (no Postgres/MySQL server needed).
  - File-based, portable, works on every free-tier host.
  - Sufficient for a decision-support tool that logs tens of
    sessions per day, not thousands per second.
  - If the project outgrows SQLite, swapping to Postgres means
    changing this one file and the connection string.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import config


# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def _connection():
    """
    Context manager that commits on success, rolls back on error,
    and always closes.  sqlite3.Row lets us access columns by name.
    """
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Setup ─────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist.  Call once on server startup."""
    with _connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                case_reference_number   TEXT    NOT NULL,
                patient_age             INTEGER NOT NULL,
                sex_at_birth            TEXT,
                injury_types            TEXT,
                sexual_offense_suspected INTEGER,
                pregnancy_confirmed     INTEGER,
                is_minor                INTEGER,
                hospital_type           TEXT,
                laws_retrieved          TEXT,
                timestamp               TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS protocol_results (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       INTEGER NOT NULL,
                protocol_json    TEXT    NOT NULL,
                ai_model_used    TEXT,
                response_time_ms INTEGER,
                timestamp        TEXT    NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS error_logs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    INTEGER,
                error_message TEXT    NOT NULL,
                endpoint      TEXT,
                timestamp     TEXT    NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)
    print("✅ SQLite DB initialised at", config.DB_PATH)


# ── Write ─────────────────────────────────────────────────────────────────────

def log_session(context: dict, laws_retrieved: list[str]) -> int:
    """Persist anonymised session metadata.  Returns auto-generated session ID."""
    with _connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO sessions (
                case_reference_number, patient_age, sex_at_birth,
                injury_types, sexual_offense_suspected, pregnancy_confirmed,
                is_minor, hospital_type, laws_retrieved, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                context["case_reference_number"],
                context["patient_age"],
                context["sex_at_birth"],
                json.dumps(context["injury_types"]),
                int(context["sexual_offense_suspected"]),
                int(context["pregnancy_confirmed"]),
                int(context["is_minor"]),
                context["hospital_type"],
                json.dumps(laws_retrieved),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cursor.lastrowid


def log_protocol_result(
    session_id: int,
    protocol: dict,
    ai_model: str,
    response_time_ms: int,
) -> None:
    with _connection() as conn:
        conn.execute(
            """
            INSERT INTO protocol_results
                (session_id, protocol_json, ai_model_used, response_time_ms, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                json.dumps(protocol),
                ai_model,
                response_time_ms,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def log_error(error_message: str, endpoint: str, session_id: int | None = None) -> None:
    with _connection() as conn:
        conn.execute(
            """
            INSERT INTO error_logs (session_id, error_message, endpoint, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, error_message, endpoint, datetime.now(timezone.utc).isoformat()),
        )


# ── Read ──────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    with _connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        minor_count = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE is_minor = 1"
        ).fetchone()[0]
        errors = conn.execute("SELECT COUNT(*) FROM error_logs").fetchone()[0]
        by_injury = conn.execute(
            "SELECT injury_types, COUNT(*) as count FROM sessions GROUP BY injury_types"
        ).fetchall()

    return {
        "total_sessions": total,
        "minor_cases": minor_count,
        "total_errors": errors,
        "injury_type_breakdown": {row["injury_types"]: row["count"] for row in by_injury},
    }
