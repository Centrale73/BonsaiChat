import os
import sqlite3

app_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_data")
os.makedirs(app_data, exist_ok=True)

DB_PATH = os.path.join(app_data, "chat_history.db")


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS messages
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"""
        )
        # Add session_id column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE messages ADD COLUMN session_id TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        conn.execute(
            """CREATE TABLE IF NOT EXISTS spaces
               (id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                instructions TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"""
        )
        
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions
               (session_id TEXT PRIMARY KEY,
                space_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"""
        )


def save_msg(role, content, session_id=None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )


def get_history(session_id=None):
    with sqlite3.connect(DB_PATH) as conn:
        if session_id:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT role, content FROM messages ORDER BY id ASC"
            ).fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]


def clear_session(session_id):
    """Clear all messages for a specific session."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


def get_all_sessions():
    """
    Retrieve all unique session IDs with their first user message as title 
    and last activity timestamp.
    """
    with sqlite3.connect(DB_PATH) as conn:
        try:
            # Get sessions with their last activity time and first user message as title
            query = '''
                SELECT 
                    m.session_id,
                    MAX(m.timestamp) as last_active,
                    (SELECT content FROM messages m2 WHERE m2.session_id = m.session_id AND m2.role = 'user' ORDER BY m2.id ASC LIMIT 1) as title,
                    s.space_id
                FROM messages m
                LEFT JOIN sessions s ON m.session_id = s.session_id
                WHERE m.session_id IS NOT NULL
                GROUP BY m.session_id
                ORDER BY last_active DESC
            '''
            rows = conn.execute(query).fetchall()
            
            sessions = []
            for r in rows:
                sid, last_active, title, space_id = r
                if not title:
                    title = "New Chat"
                elif len(title) > 30:
                    title = title[:27] + "..."
                    
                sessions.append({
                    "id": sid,
                    "title": title,
                    "timestamp": last_active,
                    "space_id": space_id
                })
            return sessions
        except Exception as e:
            print(f"Error getting sessions: {e}")
            return []

# ------------------------------------------------------------------
# Spaces and Session Metadata Operations
# ------------------------------------------------------------------

def create_space(space_id, name, description, instructions):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO spaces (id, name, description, instructions) VALUES (?, ?, ?, ?)",
            (space_id, name, description, instructions)
        )

def get_spaces():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id, name, description, instructions FROM spaces ORDER BY created_at ASC").fetchall()
    return [{"id": r[0], "name": r[1], "description": r[2], "instructions": r[3]} for r in rows]

def get_space(space_id):
    with sqlite3.connect(DB_PATH) as conn:
        r = conn.execute("SELECT id, name, description, instructions FROM spaces WHERE id = ?", (space_id,)).fetchone()
    if r:
        return {"id": r[0], "name": r[1], "description": r[2], "instructions": r[3]}
    return None

def update_space(space_id, name, description, instructions):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE spaces SET name = ?, description = ?, instructions = ? WHERE id = ?",
            (name, description, instructions, space_id)
        )

def delete_space(space_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM spaces WHERE id = ?", (space_id,))
        # Orphan the sessions
        conn.execute("UPDATE sessions SET space_id = NULL WHERE space_id = ?", (space_id,))

def set_session_space(session_id, space_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, space_id) VALUES (?, ?) ON CONFLICT(session_id) DO UPDATE SET space_id=excluded.space_id",
            (session_id, space_id)
        )

def get_session_space(session_id):
    with sqlite3.connect(DB_PATH) as conn:
        r = conn.execute("SELECT space_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return r[0] if r else None

