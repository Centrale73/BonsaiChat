"""
crm/db.py — SQLite data layer for the Cirkanime CRM.

Tables
------
organisations  — schools, camps, municipalities, festivals, etc.
events         — community events Leo wants to attend/participate in.
contacts_log   — interaction history + status tracking per organisation.
"""

import os
import sqlite3
from datetime import datetime, date
from typing import Optional, List, Dict, Any

# ---------------------------------------------------------------------------
# Database path — lives alongside the rest of Paramodus runtime data.
# ---------------------------------------------------------------------------

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_app_data = os.path.join(_base_dir, "memory_data")
os.makedirs(_app_data, exist_ok=True)

DB_PATH = os.path.join(_app_data, "cirkanime.db")


def _conn() -> sqlite3.Connection:
    """Return a new connection with row-factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS organisations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    org_type        TEXT,          -- Municipalité, Festival, Camp de jour, École, etc.
    city            TEXT,
    contact_person  TEXT,
    contact_email   TEXT,
    contact_phone   TEXT,
    activity_tags   TEXT,          -- comma-separated
    notes           TEXT,
    potential_value REAL DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    city            TEXT,
    event_name      TEXT NOT NULL,
    event_type      TEXT,          -- Festival, Fête de quartier, Marché, etc.
    period          TEXT,          -- e.g. "Juin 2026"
    best_contact    TEXT,          -- best month/period to reach out
    org_id          INTEGER REFERENCES organisations(id),
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS contacts_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id          INTEGER NOT NULL REFERENCES organisations(id),
    contact_date    TEXT NOT NULL,
    method          TEXT,          -- courriel, téléphone, en personne, messenger
    status          TEXT,          -- Contacté, Intéressé, Rencontre prévue, À relancer, Refus, Bon potentiel futur
    summary         TEXT,
    follow_up_date  TEXT,
    contract_value  REAL DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now'))
);
"""


def init_db() -> None:
    """Create tables if they don't exist yet."""
    with _conn() as conn:
        conn.executescript(_SCHEMA)
    print("[CRM] Database initialised.")


# ---------------------------------------------------------------------------
# Organisations
# ---------------------------------------------------------------------------

def add_organisation(
    name: str,
    org_type: str = "",
    city: str = "",
    contact_person: str = "",
    contact_email: str = "",
    contact_phone: str = "",
    activity_tags: str = "",
    notes: str = "",
    potential_value: float = 0,
) -> int:
    """Insert an organisation. Returns its new id."""
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO organisations
               (name, org_type, city, contact_person, contact_email,
                contact_phone, activity_tags, notes, potential_value)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, org_type, city, contact_person, contact_email,
             contact_phone, activity_tags, notes, potential_value),
        )
        return cur.lastrowid


def find_organisations(
    org_type: str = "",
    city: str = "",
    query: str = "",
) -> List[Dict[str, Any]]:
    """Search organisations by type, city, or free-text name query."""
    clauses: list[str] = []
    params: list[str] = []

    if org_type:
        clauses.append("org_type LIKE ?")
        params.append(f"%{org_type}%")
    if city:
        clauses.append("city LIKE ?")
        params.append(f"%{city}%")
    if query:
        clauses.append("(name LIKE ? OR notes LIKE ?)")
        params.extend([f"%{query}%", f"%{query}%"])

    where = " AND ".join(clauses) if clauses else "1=1"

    with _conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM organisations WHERE {where} ORDER BY name", params
        ).fetchall()
        return [dict(r) for r in rows]


def update_organisation(org_id: int, **fields) -> bool:
    """Update any number of fields on an existing organisation."""
    allowed = {
        "name", "org_type", "city", "contact_person", "contact_email",
        "contact_phone", "activity_tags", "notes", "potential_value",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False

    updates["updated_at"] = datetime.utcnow().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [org_id]

    with _conn() as conn:
        conn.execute(
            f"UPDATE organisations SET {set_clause} WHERE id = ?", values
        )
    return True


def get_organisation(org_id: int) -> Optional[Dict[str, Any]]:
    """Return a single organisation by id."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM organisations WHERE id = ?", (org_id,)
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Contacts log
# ---------------------------------------------------------------------------

def log_contact(
    org_id: int,
    contact_date: str = "",
    method: str = "",
    status: str = "",
    summary: str = "",
    follow_up_date: str = "",
    contract_value: float = 0,
) -> int:
    """Record an interaction with an organisation. Returns new log id."""
    if not contact_date:
        contact_date = date.today().isoformat()

    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO contacts_log
               (org_id, contact_date, method, status, summary,
                follow_up_date, contract_value)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (org_id, contact_date, method, status, summary,
             follow_up_date, contract_value),
        )
        return cur.lastrowid


def get_history(org_id: int) -> List[Dict[str, Any]]:
    """Full contact timeline for an organisation, newest first."""
    with _conn() as conn:
        rows = conn.execute(
            """SELECT cl.*, o.name AS org_name
               FROM contacts_log cl
               JOIN organisations o ON o.id = cl.org_id
               WHERE cl.org_id = ?
               ORDER BY cl.contact_date DESC""",
            (org_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_followups_due() -> List[Dict[str, Any]]:
    """Return all contacts whose follow_up_date is today or overdue."""
    today = date.today().isoformat()
    with _conn() as conn:
        rows = conn.execute(
            """SELECT cl.*, o.name AS org_name, o.city, o.org_type
               FROM contacts_log cl
               JOIN organisations o ON o.id = cl.org_id
               WHERE cl.follow_up_date <= ?
                 AND cl.follow_up_date != ''
                 AND cl.status NOT IN ('Refus')
               ORDER BY cl.follow_up_date ASC""",
            (today,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def add_event(
    event_name: str,
    city: str = "",
    event_type: str = "",
    period: str = "",
    best_contact: str = "",
    org_id: Optional[int] = None,
    notes: str = "",
) -> int:
    """Add a community event. Returns its new id."""
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO events
               (event_name, city, event_type, period, best_contact, org_id, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (event_name, city, event_type, period, best_contact, org_id, notes),
        )
        return cur.lastrowid


def find_events(
    city: str = "",
    event_type: str = "",
    query: str = "",
) -> List[Dict[str, Any]]:
    """Search events by city, type, or free-text query."""
    clauses: list[str] = []
    params: list[str] = []

    if city:
        clauses.append("city LIKE ?")
        params.append(f"%{city}%")
    if event_type:
        clauses.append("event_type LIKE ?")
        params.append(f"%{event_type}%")
    if query:
        clauses.append("(event_name LIKE ? OR notes LIKE ?)")
        params.extend([f"%{query}%", f"%{query}%"])

    where = " AND ".join(clauses) if clauses else "1=1"

    with _conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM events WHERE {where} ORDER BY event_name", params
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Seasonal intelligence
# ---------------------------------------------------------------------------

# Which org types are best contacted in which months (1-12).
# Keys are ASCII-friendly to match typical user/seed input.
SEASONAL_WINDOWS: Dict[str, List[int]] = {
    "Municipalite":         [9, 10, 11, 12, 1, 2],      # Automne + Hiver
    "Festival":             [1, 2, 3, 4],                # Hiver + Printemps
    "Camp de jour":         [1, 2, 3],                   # Janvier-Mars
    "Ecole":                [5, 11],                     # Mai + Novembre
    "Parascolaire":         [5, 11],
    "Maison des jeunes":    [1, 9],                      # Janvier + Septembre
    "Organisme":            [9, 10, 11, 12, 1, 2],      # Automne + Hiver
}


def get_seasonal_targets(month: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Return organisations whose org_type matches this month's seasonal
    contact window, along with the recommended action.
    """
    if month is None:
        month = date.today().month

    matching_types = [
        t for t, months in SEASONAL_WINDOWS.items() if month in months
    ]

    if not matching_types:
        return []

    # Use LIKE for accent-tolerant matching (e.g. "Ecole" matches "École").
    like_clauses = " OR ".join("org_type LIKE ?" for _ in matching_types)
    like_params = [f"%{t}%" for t in matching_types]

    with _conn() as conn:
        rows = conn.execute(
            f"""SELECT * FROM organisations
                WHERE {like_clauses}
                ORDER BY org_type, city""",
            like_params,
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Pipeline summary
# ---------------------------------------------------------------------------

def pipeline_summary() -> Dict[str, Any]:
    """Pipeline overview grouped by status with total $CAD value."""
    with _conn() as conn:
        rows = conn.execute(
            """SELECT status, COUNT(*) AS count,
                      COALESCE(SUM(contract_value), 0) AS total_value
               FROM contacts_log
               WHERE status IS NOT NULL AND status != ''
               GROUP BY status
               ORDER BY count DESC"""
        ).fetchall()

        pipeline = [dict(r) for r in rows]

        total_row = conn.execute(
            """SELECT COUNT(DISTINCT org_id) AS total_orgs,
                      COALESCE(SUM(contract_value), 0) AS total_pipeline_value
               FROM contacts_log"""
        ).fetchone()

        return {
            "by_status": pipeline,
            "total_orgs_contacted": total_row["total_orgs"],
            "total_pipeline_value_cad": total_row["total_pipeline_value"],
        }
