"""
bonsai_agent.py — Nexus-integrated Agno agent for Paramodus.

Integration order implemented here
-----------------------------------
1. Thread-safe singleton with RLock (from Paramodus)
2. Multi-provider model factory (from Paramodus_Base/agents/workspace_agent.py)
3. IntentGovernor hook injection (from Exp1-Intent/governor.py)
   - constitution_hook + logger_hook appended to tool_hooks
   - instructions replaced with a callable intent_retriever
   - Governor wraps AFTER agent construction — existing hooks preserved
4. get_run_kwargs() pattern (from Paramodus) — avoids singleton mutation
   on concurrent sessions
5. aingest_files() async concurrent ingestion (from Paramodus)

NOTE: The IntentGovernor is opt-in via env var INTENT_GOVERNANCE_ENABLED=true.
When disabled, the agent behaves exactly as the pre-nexus Paramodus bonsai_agent.
"""

import asyncio
import logging
import os
import tempfile
import threading
from typing import List, Optional, Dict

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.chunking.recursive import RecursiveChunking
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.csv_reader import CSVReader
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.reader.text_reader import TextReader
from agno.memory import MemoryManager
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.lancedb import LanceDb, SearchType

# Cirkanime CRM tools
from crm import ALL_TOOLS as CRM_TOOLS, init_crm_db
from crm.google_tools import ALL_GOOGLE_TOOLS

logger = logging.getLogger("paramodus.bonsai_agent")

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

_base_dir = os.path.dirname(os.path.abspath(__file__))
_app_data = os.path.join(_base_dir, "memory_data")
os.makedirs(_app_data, exist_ok=True)

LANCE_URI = os.path.join(_app_data, "lancedb")
DB_FILE   = os.path.join(_app_data, "paramodus_memory.db")

DEFAULT_CHUNKER = RecursiveChunking(chunk_size=700, overlap=100)

# ---------------------------------------------------------------------------
# Multi-provider model factory (ported from Paramodus_Base/agents/workspace_agent.py)
# ---------------------------------------------------------------------------
# In Paramodus the primary model is always local Bonsai (LlamaCpp).
# For governance evaluation (judge) we need an external provider.
# This factory is therefore used mainly by the judge — not the chat agent itself.
# ---------------------------------------------------------------------------

def _get_model(provider: str, api_key: Optional[str] = None, model_id: Optional[str] = None):
    """
    Return the correct Agno model instance for the given provider.
    Mirrors workspace_agent.get_model() but keeps LlamaCpp as the Bonsai path.
    """
    _DEFAULT_MODELS = {
        "openai":      "gpt-4o",
        "anthropic":   "claude-sonnet-4-5-20250929",
        "gemini":      "gemini-2.0-flash-001",
        "groq":        "llama-3.3-70b-versatile",
        "grok":        "grok-3",
        "openrouter":  "openai/gpt-4o-mini",
        "perplexity":  "sonar-pro",
        "bonsai":      "local-model",
    }
    mid = model_id or _DEFAULT_MODELS.get(provider, "gpt-4o")

    if provider == "openai":
        from agno.models.openai import OpenAIChat
        return OpenAIChat(id=mid, api_key=api_key)
    elif provider == "anthropic":
        from agno.models.anthropic import Claude
        return Claude(id=mid, api_key=api_key)
    elif provider == "gemini":
        from agno.models.google import Gemini
        return Gemini(id=mid, api_key=api_key)
    elif provider == "groq":
        from agno.models.groq import Groq
        return Groq(id=mid, api_key=api_key)
    elif provider == "grok":
        from agno.models.xai import xAI
        return xAI(id=mid, api_key=api_key)
    elif provider == "openrouter":
        from agno.models.openrouter import OpenRouter
        return OpenRouter(id=mid, api_key=api_key)
    elif provider == "perplexity":
        from agno.models.perplexity import Perplexity
        return Perplexity(id=mid, api_key=api_key)
    elif provider == "bonsai":
        from agno.models.llama_cpp import LlamaCpp
        base_url = os.environ.get("BONSAI_BASE_URL", "http://127.0.0.1:8081/v1")
        return LlamaCpp(id="bonsai", base_url=base_url)
    else:
        logger.warning("Unknown provider '%s', falling back to OpenAI.", provider)
        from agno.models.openai import OpenAIChat
        return OpenAIChat(id=mid, api_key=api_key)


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

BASE_INSTRUCTIONS = """\
You are Paramodus, a concise and precise assistant powered by the Bonsai model.

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

_LANG_SUFFIXES = {
    "fr": "\nRéponds toujours en français.",
    "es": "\nResponde siempre en español.",
}

SUPPORTED_LANGUAGES = ["en", "fr", "es"]

# ---------------------------------------------------------------------------
# Governance (IntentGovernor — from Exp1-Intent)
# ---------------------------------------------------------------------------
# Opt-in: set INTENT_GOVERNANCE_ENABLED=true to activate.
# The governor appends its hooks and replaces instructions AFTER init_agent()
# so it never clobbers the pre-built agent state during construction.
# ---------------------------------------------------------------------------

_GOVERNANCE_ENABLED = os.environ.get("INTENT_GOVERNANCE_ENABLED", "").lower() in ("1", "true", "yes")
_governor = None


def _build_governor():
    """
    Build the IntentGovernor lazily once — requires Exp1-Intent code
    to be co-located in a `governance/` package or on sys.path.
    The judge model is the cheapest available external provider to avoid
    running a second local LLM for evaluation.
    """
    try:
        from governance.governor import IntentGovernor

        judge_provider  = os.environ.get("JUDGE_PROVIDER", "groq")
        judge_api_key   = os.environ.get("JUDGE_API_KEY")
        judge_model_id  = os.environ.get("JUDGE_MODEL_ID")
        judge_model     = _get_model(judge_provider, judge_api_key, judge_model_id) if judge_api_key else None

        slack_hook = None
        if os.environ.get("SLACK_WEBHOOK_URL"):
            from governance.evals.judge_eval import SlackEscalationHook
            slack_hook = SlackEscalationHook()

        gov = IntentGovernor(
            constitution=os.environ.get(
                "CONSTITUTION_PATH",
                os.path.join(_base_dir, "governance", "constitutions", "paramodus.yaml"),
            ),
            judge_criteria=os.environ.get(
                "CRITERIA_PATH",
                os.path.join(_base_dir, "governance", "criteria", "brand_voice.txt"),
            ),
            judge_model=judge_model,
            escalation_hook=slack_hook,
            base_intent=BASE_INSTRUCTIONS,
            judge_threshold=int(os.environ.get("JUDGE_THRESHOLD", "7")),
        )
        logger.info("[Nexus] IntentGovernor built successfully.")
        return gov
    except Exception as exc:
        logger.warning("[Nexus] IntentGovernor build failed (%s) — running without governance.", exc)
        return None


# ---------------------------------------------------------------------------
# Singletons — thread-safe (from Paramodus)
# ---------------------------------------------------------------------------

_lock         = threading.RLock()
_agent:       Optional[Agent]     = None
_knowledge:   Optional[Knowledge] = None
_db:          Optional[SqliteDb]  = None


def _get_db() -> SqliteDb:
    global _db
    if _db is None:
        with _lock:
            if _db is None:
                _db = SqliteDb(db_file=DB_FILE)
    return _db


def _get_knowledge() -> Knowledge:
    global _knowledge
    if _knowledge is None:
        with _lock:
            if _knowledge is None:
                _knowledge = Knowledge(
                    vector_db=LanceDb(
                        table_name="bonsai_docs",
                        uri=LANCE_URI,
                        search_type=SearchType.hybrid,
                        embedder=FastEmbedEmbedder(
                            id="BAAI/bge-small-en-v1.5",
                            dimensions=384,
                        ),
                    ),
                    chunking_strategy=DEFAULT_CHUNKER,
                )
    return _knowledge


# ---------------------------------------------------------------------------
# Agent initialisation
# ---------------------------------------------------------------------------

def init_agent() -> None:
    """
    Build and cache the Bonsai Agno agent.
    Called once by bridge.py after llama-server reports it is ready.
    Safe to call multiple times (idempotent).

    Integration steps (in order):
      1. Build base Agent with LlamaCpp + CRM tools + RAG.
      2. If INTENT_GOVERNANCE_ENABLED, wrap with IntentGovernor.
         governor.wrap() appends tool_hooks and replaces instructions
         with the callable intent_retriever — existing hooks are preserved
         because wrap() does: agent.tool_hooks = [logger_hook, constitution_hook]
         + existing_hooks  (see Exp1-Intent/governor.py).
    """
    global _agent, _governor

    if _agent is not None:
        return

    with _lock:
        if _agent is not None:
            return

        base_url = os.environ.get("BONSAI_BASE_URL", "http://127.0.0.1:8081/v1")

        from agno.models.llama_cpp import LlamaCpp

        agent = Agent(
            model=LlamaCpp(id="bonsai", base_url=base_url),
            db=_get_db(),
            memory_manager=MemoryManager(
                db=_get_db(),
                additional_instructions="Remember user preferences and recurring topics.",
            ),
            update_memory_on_run=True,
            add_memories_to_context=True,
            add_history_to_context=True,
            num_history_runs=6,
            knowledge=_get_knowledge(),
            search_knowledge=True,
            tools=[
                DuckDuckGoTools(),
                *CRM_TOOLS,
                *ALL_GOOGLE_TOOLS,
            ],
            tool_call_limit=8,
            instructions=BASE_INSTRUCTIONS,
            markdown=True,
            stream=True,
        )

        # ── Step 2: Governance wrap (Exp1-Intent) ────────────────────────
        if _GOVERNANCE_ENABLED:
            if _governor is None:
                _governor = _build_governor()
            if _governor is not None:
                agent = _governor.wrap(agent)
                logger.info("[Nexus] Agent wrapped with IntentGovernor.")

        _agent = agent
        logger.info("[Nexus] Bonsai agent initialised.")


def get_agent(session_id: str, language: str = "en") -> Agent:
    """
    Return the singleton agent. Raises RuntimeError if init_agent() hasn't
    been called yet (i.e. llama-server is not ready).
    """
    if _agent is None:
        raise RuntimeError("Agent not initialised — call init_agent() after llama-server is ready.")
    return _agent


def get_run_kwargs(session_id: str, language: str = "en", space_instructions: str = "") -> dict:
    """
    Return per-run kwargs to pass to agent.arun().

    Critically: we do NOT mutate _agent.session_id or _agent.instructions here.
    That was the race condition in organizer_agent.py (get_organizer_agent mutated
    shared state). Instead we pass session_id and a dynamic instructions override
    as run-time kwargs, which Agno resolves per-call.
    """
    lang_suffix = _LANG_SUFFIXES.get(language, "")
    instructions = BASE_INSTRUCTIONS + lang_suffix
    if space_instructions:
        instructions += f"\n\nSpace context:\n{space_instructions}"

    return {
        "session_id": session_id,
        "instructions_override": instructions,
    }


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

async def aingest_files(files: List[dict]) -> bool:
    """
    Concurrently ingest a list of {name, data} dicts into the knowledge base.
    """
    async def _ingest_one(file_info: dict) -> bool:
        name = file_info["name"]
        data = file_info["data"]
        suffix = os.path.splitext(name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                reader = PDFReader(chunking_strategy=DEFAULT_CHUNKER)
            elif suffix == ".csv":
                reader = CSVReader(chunking_strategy=DEFAULT_CHUNKER)
            elif suffix in (".txt", ".md", ".py", ".js", ".json", ".yaml", ".yml"):
                reader = TextReader(chunking_strategy=DEFAULT_CHUNKER)
            else:
                logger.warning("[RAG] Unsupported file type: %s", name)
                return False

            _get_knowledge().insert(
                path=tmp_path,
                name=name,
                reader=reader,
                metadata={"filename": name},
                upsert=True,
            )
            logger.info("[RAG] Ingested: %s", name)
            return True
        except Exception as exc:
            logger.error("[RAG] Error ingesting %s: %s", name, exc)
            return False
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    results = await asyncio.gather(*[_ingest_one(f) for f in files])
    return any(results)


def clear_knowledge_base() -> bool:
    try:
        kb = _get_knowledge()
        if kb.vector_db.exists():
            kb.vector_db.drop()
        kb.vector_db.create()
        logger.info("[RAG] Knowledge base cleared.")
        return True
    except Exception as exc:
        logger.error("[RAG] Clear failed: %s", exc)
        return False
