"""
bonsai_agent.py — Agno agent + RAG knowledge base for BonsaiChat.

agent-zero-features branch additions
─────────────────────────────────────
• WebSearchTools  — DuckDuckGo web search (no API key required)
• CodeExecTools   — sandboxed Python execution
• SubAgentTools   — Agent Zero-style sub-task delegation
"""

from __future__ import annotations

import os
from typing import List, Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.memory import MemoryManager
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.reader.csv_reader import CSVReader
from agno.knowledge.reader.text_reader import TextReader
from agno.knowledge.chunking.recursive import RecursiveChunking
from agno.models.llama_cpp import LlamaCpp

from tools.web_search import WebSearchTools
from tools.code_exec import CodeExecTools
from tools.subordinate_agent import SubAgentTools

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_base_dir = os.path.dirname(os.path.abspath(__file__))
_app_data = os.path.join(_base_dir, "memory_data")
os.makedirs(_app_data, exist_ok=True)

LANCE_URI = os.path.join(_app_data, "lancedb")
DB_FILE   = os.path.join(_app_data, "bonsaichat_memory.db")

DEFAULT_CHUNKER = RecursiveChunking(chunk_size=1000, overlap=150)

# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────

_knowledge:      Optional[Knowledge]     = None
_db:             Optional[SqliteDb]      = None
_memory_manager: Optional[MemoryManager] = None
_agent:          Optional[Agent]         = None


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


# ─────────────────────────────────────────────────────────────────────────────
# Agent init
# ─────────────────────────────────────────────────────────────────────────────

def init_agent() -> None:
    """
    Build the Agno agent with all tools.  Called once by the bridge after
    llama-server is ready.  Subsequent calls are no-ops.
    """
    global _agent
    if _agent is not None:
        return

    model     = LlamaCpp(id="bonsai-8b", base_url="http://127.0.0.1:8081/v1")
    knowledge = _get_knowledge()

    _agent = Agent(
        model=model,
        db=_get_db(),
        memory_manager=_get_memory_manager(),
        update_memory_on_run=True,
        add_memories_to_context=True,
        add_history_to_context=True,
        instructions=(
            "You are an intelligent local assistant with access to:\n"
            "• Your uploaded knowledge base (RAG)\n"
            "• Web search via DuckDuckGo\n"
            "• Python code execution\n"
            "• Sub-agent delegation for focused sub-tasks\n\n"
            "Use tools proactively to give accurate answers.\n"
            "Format mathematical equations with LaTeX ($$...$$).\n"
            "When delegating to a sub-agent, give it a complete, self-contained task."
        ),
        knowledge=knowledge,
        search_knowledge=True,
        tools=[
            WebSearchTools(),
            CodeExecTools(),
            SubAgentTools(parent_model=model, db=_get_db(), knowledge=knowledge),
        ],
        markdown=True,
    )


def get_agent(session_id: str) -> Agent:
    if _agent is None:
        init_agent()
    _agent.session_id = session_id  # type: ignore[union-attr]
    return _agent


# ─────────────────────────────────────────────────────────────────────────────
# RAG helpers
# ─────────────────────────────────────────────────────────────────────────────

def ingest_files(files: List[dict]) -> bool:
    import tempfile
    ingested = 0
    for f in files:
        name: str   = f["name"]
        data: bytes = f["data"]
        tmp_path: Optional[str] = None
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
                continue

            _get_knowledge().insert(
                path=tmp_path,
                name=name,
                reader=reader,
                metadata={"filename": name},
                upsert=True,
            )
            ingested += 1
        except Exception as exc:
            print(f"[BonsaiAgent] Error ingesting {name}: {exc}")
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    return ingested > 0


def clear_knowledge_base() -> bool:
    try:
        vdb = _get_knowledge().vector_db
        if vdb.exists():
            vdb.drop()
        vdb.create()
        return True
    except Exception as exc:
        print(f"[BonsaiAgent] Error clearing knowledge base: {exc}")
        return False
