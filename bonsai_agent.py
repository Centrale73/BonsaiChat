"""
bonsai_agent.py — Agno agent and RAG knowledge base for BonsaiChat.

Extracted from the original BonsaiChat.py monolith so that api/bridge.py
can import it lazily (after the llama-server is already running).
"""

import os
from typing import List, Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.llama_cpp import LlamaCpp
from agno.memory import MemoryManager
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.reader.csv_reader import CSVReader
from agno.knowledge.reader.text_reader import TextReader
from agno.knowledge.chunking.recursive import RecursiveChunking

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

_base_dir = os.path.dirname(os.path.abspath(__file__))
_app_data = os.path.join(_base_dir, "memory_data")
os.makedirs(_app_data, exist_ok=True)

LANCE_URI = os.path.join(_app_data, "lancedb")
DB_FILE   = os.path.join(_app_data, "bonsaichat_memory.db")

DEFAULT_CHUNKER = RecursiveChunking(chunk_size=1000, overlap=150)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_knowledge: Optional[Knowledge] = None
_db: Optional[SqliteDb] = None
_memory_manager: Optional[MemoryManager] = None
_agent: Optional[Agent] = None          # single agent reused across sessions


def _get_db() -> SqliteDb:
    global _db
    if _db is None:
        _db = SqliteDb(db_file=DB_FILE)
    return _db


def _get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(
            db=_get_db(),
            additional_instructions=(
                "Extract strictly factual statements about the user's preferences, "
                "projects, and constraints. Do not store conversational filler."
            ),
        )
    return _memory_manager


def _get_knowledge() -> Knowledge:
    global _knowledge
    if _knowledge is None:
        _knowledge = Knowledge(
            vector_db=LanceDb(
                table_name="user_documents",
                uri=LANCE_URI,
                embedder=FastEmbedEmbedder(
                    id="BAAI/bge-small-en-v1.5",
                    dimensions=384,
                ),
            ),
        )
    return _knowledge


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_agent() -> None:
    """
    Build the Agno agent.  Called once by the bridge after the llama-server
    reports it is ready.  Safe to call multiple times — subsequent calls are
    no-ops.
    """
    global _agent
    if _agent is not None:
        return

    _agent = Agent(
        model=LlamaCpp(
            id="bonsai-8b",
            base_url="http://127.0.0.1:8081/v1",
        ),
        db=_get_db(),
        memory_manager=_get_memory_manager(),
        update_memory_on_run=True,
        add_memories_to_context=True,
        add_history_to_context=True,
        instructions=(
            "You are a helpful and intelligent assistant with access to uploaded "
            "documents. Format all mathematical equations using proper LaTeX syntax "
            "(e.g., $$...$$ for block equations)."
        ),
        knowledge=_get_knowledge(),
        search_knowledge=True,
        markdown=True,
    )


def get_agent(session_id: str, language: str = 'en') -> Agent:
    """Return the global agent, initialising it if necessary, and update its language."""
    if _agent is None:
        init_agent()
    # Agno agents carry session context via the DB; just tag the run
    _agent.session_id = session_id
    
    # Update language instructions
    base_instructions = "You are a helpful and intelligent assistant with access to uploaded documents. Format all mathematical equations using proper LaTeX syntax (e.g., $$...$$ for block equations)."
    
    if language == 'fr':
        _agent.instructions = base_instructions + " You MUST reply entirely in French (Français)."
    elif language == 'es':
        _agent.instructions = base_instructions + " You MUST reply entirely in Spanish (Español)."
    else:
        _agent.instructions = base_instructions
        
    return _agent


def ingest_files(files: List[dict]) -> bool:
    """
    Accept a list of dicts: [{"name": str, "data": bytes}, ...]
    Writes each to a temp file, ingests into the vector DB, then removes it.
    """
    import tempfile
    ingested = 0
    for f in files:
        name: str = f["name"]
        data: bytes = f["data"]
        try:
            suffix = os.path.splitext(name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            if suffix == ".pdf":
                reader = PDFReader(chunking_strategy=DEFAULT_CHUNKER)
            elif suffix == ".csv":
                reader = CSVReader(chunking_strategy=DEFAULT_CHUNKER)
            elif suffix in (".txt", ".md", ".py", ".js", ".json"):
                reader = TextReader(chunking_strategy=DEFAULT_CHUNKER)
            else:
                print(f"[BonsaiAgent] Unsupported file type: {name}")
                os.remove(tmp_path)
                continue

            _get_knowledge().insert(
                path=tmp_path,
                name=name,
                reader=reader,
                metadata={"filename": name},
                upsert=True,
            )
            ingested += 1
        except Exception as e:
            print(f"[BonsaiAgent] Error ingesting {name}: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return ingested > 0


def clear_knowledge_base() -> bool:
    """Drop and recreate the vector DB table."""
    try:
        vdb = _get_knowledge().vector_db
        if vdb.exists():
            vdb.drop()
        vdb.create()
        return True
    except Exception as e:
        print(f"[BonsaiAgent] Error clearing knowledge base: {e}")
        return False
