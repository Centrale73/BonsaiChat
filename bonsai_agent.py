"""
bonsai_agent.py — Agno agent and RAG knowledge base for BonsaiChat.

Extracted from the original BonsaiChat.py monolith so that api/bridge.py
can import it lazily (after the llama-server is already running).
"""

import asyncio
import os
import tempfile
import threading
from typing import List, Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.chunking.recursive import RecursiveChunking
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.csv_reader import CSVReader
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.reader.text_reader import TextReader
from agno.memory import MemoryManager
from agno.memory.manager import UserMemory
from agno.models.llama_cpp import LlamaCpp
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.lancedb import LanceDb, SearchType

# Cirkanime CRM tools
from crm import ALL_TOOLS as CRM_TOOLS, init_crm_db

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

_base_dir = os.path.dirname(os.path.abspath(__file__))
_app_data = os.path.join(_base_dir, "memory_data")
os.makedirs(_app_data, exist_ok=True)

LANCE_URI = os.path.join(_app_data, "lancedb")
DB_FILE   = os.path.join(_app_data, "bonsaichat_memory.db")

# CHANGE: Slightly smaller chunks work better for 1-bit models — they produce
# cleaner, more focused context windows. Overlap kept proportional.
DEFAULT_CHUNKER = RecursiveChunking(chunk_size=700, overlap=100)

# CHANGE: Rewritten instructions for small/quantised models.
# Rules of thumb:
#   • Use numbered imperatives — 1-bit models follow ordered lists more reliably
#     than flowing prose.
#   • Keep each rule ≤ 15 words so it fits comfortably in a short context.
#   • Put the most important rules first (context is read top-to-bottom).
#   • Avoid redundant negations ("do NOT claim you cannot…") — state the
#     positive action instead.
BASE_INSTRUCTIONS = """\
You are Paramodus AI , a concise and precise assistant.

Rules (follow in order):
1. DOCUMENTS — When the user mentions files, data, or "provided documents", \
call search_knowledge_base first, then answer from the results.
2. WEB — For current events or news, call the web_search tool.
3. KNOWLEDGE — For general engineering, science, or math, answer from memory.
4. MATH — Render all equations with LaTeX ($$...$$).
5. BREVITY — Be direct. Omit filler phrases like "Certainly!" or "Of course!".
6. UNCERTAINTY — If unsure, say so briefly. Never fabricate facts or citations.
7. CRM — For anything about organisations, contacts, events, follow-ups, \
pipeline, or seasonal outreach, use the CRM tools (tool_add_organisation, \
tool_find_organisations, tool_log_contact, tool_get_followups_due, etc.).
"""

# CHANGE: Supported languages now stored as a dict — easier to extend and
# avoids a growing if/elif chain in get_run_kwargs.
_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "fr": "You MUST reply entirely in French (Français).",
    "es": "You MUST reply entirely in Spanish (Español).",
    "de": "You MUST reply entirely in German (Deutsch).",
    "pt": "You MUST reply entirely in Portuguese (Português).",
    "zh": "You MUST reply entirely in Simplified Chinese (简体中文).",
}

# ---------------------------------------------------------------------------
# Thread-safe singleton helpers
# ---------------------------------------------------------------------------

# CHANGE: A single lock prevents double-initialisation when the bridge starts
# concurrent requests before init_agent() finishes.
_lock = threading.RLock()

_knowledge: Optional[Knowledge] = None
_db: Optional[SqliteDb] = None
_memory_manager: Optional[MemoryManager] = None
_agent: Optional[Agent] = None


def _get_db() -> SqliteDb:
    global _db
    if _db is None:
        with _lock:
            if _db is None:          # double-checked locking
                _db = SqliteDb(db_file=DB_FILE)
    return _db


def _get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        with _lock:
            if _memory_manager is None:
                # CHANGE: Removed CappedMemoryManager subclass — the private
                # method it called (_get_last_n_memories) does not exist on the
                # base class and would raise AttributeError at runtime. The cap
                # is now expressed via num_history_runs on the Agent itself,
                # which is the documented Agno mechanism.
                _memory_manager = MemoryManager(
                    db=_get_db(),
                    additional_instructions=(
                        "Extract only concrete, factual statements about the user's "
                        "preferences, goals, and constraints. "
                        "Skip pleasantries and conversational filler."
                    ),
                )
    return _memory_manager


def _get_knowledge() -> Knowledge:
    global _knowledge
    if _knowledge is None:
        with _lock:
            if _knowledge is None:
                _knowledge = Knowledge(
                    vector_db=LanceDb(
                        table_name="user_documents",
                        uri=LANCE_URI,
                        search_type=SearchType.hybrid,
                        embedder=FastEmbedEmbedder(
                            id="BAAI/bge-small-en-v1.5",
                            dimensions=384,
                        ),
                    ),
                    contents_db=_get_db(),
                    # CHANGE: Bumped to 5 — gives the model more context when
                    # answering document-heavy questions without hurting small
                    # queries (extra results are simply not referenced).
                    max_results=5,
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

    with _lock:
        if _agent is not None:      # another thread may have finished first
            return

        # Initialise CRM database tables before building the agent.
        init_crm_db()

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
            # CHANGE: Reduced from 5 to 3 — 1-bit models have limited context;
            # three prior turns is usually enough for continuity without
            # crowding out document chunks.
            num_history_runs=3,
            instructions=BASE_INSTRUCTIONS,
            knowledge=_get_knowledge(),
            search_knowledge=True,
            # Keep False: avoids blindly stuffing all KB chunks into every
            # prompt. The agent calls search_knowledge_base when relevant.
            add_knowledge_to_context=False,
            markdown=True,
            tools=[DuckDuckGoTools()] + CRM_TOOLS,
            # Raised to 8 — CRM queries may chain multiple tool calls
            # (e.g. find org -> log contact -> get pipeline).
            tool_call_limit=8,
        )

def get_agent(session_id: str, language: str = 'en') -> Agent:
    """Return the global agent, initialising it if necessary, and update its language."""
    if _agent is None:
        init_agent()
    # Agno agents carry session context via the DB; just tag the run
    _agent.session_id = session_id
    
    # Update language instructions
    if language == 'fr':
        kwargs["additional_instructions"] = "You MUST reply entirely in French (Français)."
    elif language == 'es':
        _agent.instructions = f"{BASE_INSTRUCTIONS} You MUST reply entirely in Spanish (Español)."
    else:
        _agent.instructions = BASE_INSTRUCTIONS
        
    return _agent


# CHANGE: New helper — lets the bridge hot-swap the agent (e.g. after a model
# reload) without restarting the whole process.
def reset_agent() -> None:
    """Tear down the current agent so the next get_agent() call rebuilds it."""
    global _agent
    with _lock:
        _agent = None


def get_run_kwargs(session_id: str, language: str = "en") -> dict:
    """Return the dynamic kwargs needed by agent.arun for a given session and language."""
    kwargs: dict = {"session_id": session_id}
    instruction = _LANGUAGE_INSTRUCTIONS.get(language)
    if instruction:
        kwargs["additional_instructions"] = instruction
    return kwargs

# ---------------------------------------------------------------------------
# File reader factory
# ---------------------------------------------------------------------------

# CHANGE: Extended to cover common source-code and markup extensions that
# users frequently drop into a chat assistant.
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".yaml", ".yml", ".toml", ".html", ".htm",
    ".rst", ".xml", ".sh", ".bash",
}


def _get_reader(file_name: str):
    """Return the appropriate Agno reader for *file_name*, or None if unsupported."""
    suffix = os.path.splitext(file_name)[1].lower()
    if suffix == ".pdf":
        return PDFReader(chunking_strategy=DEFAULT_CHUNKER)
    if suffix == ".csv":
        return CSVReader(chunking_strategy=DEFAULT_CHUNKER)
    if suffix in _TEXT_EXTENSIONS:
        return TextReader(chunking_strategy=DEFAULT_CHUNKER)
    return None


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

async def aingest_local_file(file_path: str) -> bool:
    """Ingest a single file from a local disk path asynchronously."""
    name = os.path.basename(file_path)
    reader = _get_reader(name)
    if not reader:
        print(f"[BonsaiAgent] Unsupported file type: {name}")
        return False

    try:
        await _get_knowledge().ainsert(
            path=file_path,
            name=name,
            reader=reader,
            metadata={"filename": name},
            upsert=True,
        )
        print(f"[BonsaiAgent] Ingested: {name}")
        return True
    except Exception as exc:
        print(f"[BonsaiAgent] Error ingesting {name}: {exc}")
        return False


def ingest_local_file(file_path: str) -> bool:
    """Synchronous wrapper for aingest_local_file."""
    return asyncio.run(aingest_local_file(file_path))

def ingest_files(files: List[dict]) -> bool:
    """
    Accept a list of dicts: [{"name": str, "data": bytes}, ...]
    Writes each to a temp file, ingests into the vector DB, then removes it.

    CHANGE: Now ingests files concurrently with asyncio.gather — much faster
    when the user uploads several documents at once.
    """
    async def _ingest_one(f: dict) -> bool:
        name: str = f["name"]
        data: bytes = f["data"]
        tmp_path: Optional[str] = None
        try:
            reader = _get_reader(name)
            if not reader:
                print(f"[BonsaiAgent] Unsupported file type: {name}")
                return False

            suffix = os.path.splitext(name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            await _get_knowledge().ainsert(
                path=tmp_path,
                name=name,
                reader=reader,
                metadata={"filename": name},
                upsert=True,
            )
            print(f"[BonsaiAgent] Ingested: {name}")
            return True
        except Exception as exc:
            print(f"[BonsaiAgent] Error ingesting {name}: {exc}")
            return False
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    results = await asyncio.gather(*(_ingest_one(f) for f in files))
    return any(results)


def ingest_files(files: List[dict]) -> bool:
    """Synchronous wrapper for aingest_files."""
    return asyncio.run(aingest_files(files))


def clear_knowledge_base() -> bool:
    """Drop and recreate the vector DB table, and clear the contents DB."""
    import sqlite3

    try:
        vdb = _get_knowledge().vector_db
        if vdb.exists():
            vdb.drop()
        vdb.create()
        with sqlite3.connect(DB_FILE) as conn:
            for table in ("agno_knowledge_content", "agno_knowledge_contents"):
                try:
                    conn.execute(f"DELETE FROM {table}")
                except Exception:
                    pass

        print("[BonsaiAgent] Knowledge base cleared.")
        return True
    except Exception as exc:
        print(f"[BonsaiAgent] Error clearing knowledge base: {exc}")
        return False