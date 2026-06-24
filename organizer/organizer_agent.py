"""
organizer/organizer_agent.py — Nexus-integrated file organiser for Paramodus.

Fixes applied from the audit (relative to Organizer/organizer_agent.py)
-----------------------------------------------------------------------
1. Thread-safe double-checked locking on all singletons
   (copied from bonsai_agent.py — bare global was a race condition).

2. Shared DB path with bonsai_agent.py.
   Was: ../memory_data/bonsaichat_memory.db
   Now: memory_data/paramodus_memory.db  (same file as bonsai_agent.DB_FILE)
   This lets the chat agent answer "where did my invoice go?" using the
   same SqliteDb memory store without a separate file.

3. get_organizer_agent() no longer mutates the shared singleton.
   Was:
       _organizer_agent.session_id = session_id
       _organizer_agent.instructions = ...
   Now: session_id and language override are returned as run kwargs
   (same pattern as bonsai_agent.get_run_kwargs — caller passes them to arun()).
   The returned dict is: {"session_id": ..., "instructions_override": ...}

4. ingest_organized_manifest() uses bonsai_agent._get_knowledge() directly
   so both agents share a single LanceDB table — no dual-ingestion.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import shutil
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agno.agent import Agent
from agno.knowledge.chunking.recursive import RecursiveChunking
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.memory import MemoryManager
from agno.vectordb.lancedb import LanceDb

try:
    from agno.knowledge.reader.pdf_reader import PDFReader
    _PDF = True
except ImportError:
    _PDF = False

try:
    from agno.knowledge.reader.csv_reader import CSVReader
    _CSV = True
except ImportError:
    _CSV = False

try:
    from agno.knowledge.reader.text_reader import TextReader
    _TEXT = True
except ImportError:
    _TEXT = False

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

logger = logging.getLogger("paramodus.organizer")

# ---------------------------------------------------------------------------
# Paths — shared with bonsai_agent.py (Fix #2)
# ---------------------------------------------------------------------------

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_app_data = os.path.join(_base_dir, "memory_data")
os.makedirs(_app_data, exist_ok=True)

# SHARED: same DB file as bonsai_agent.py so memories are unified
DB_FILE   = os.path.join(_app_data, "paramodus_memory.db")
LANCE_URI = os.path.join(_app_data, "lancedb")

DEFAULT_CHUNKER = RecursiveChunking(chunk_size=1000, overlap=150)

# ---------------------------------------------------------------------------
# Extension rule map
# ---------------------------------------------------------------------------

EXT_RULES: Dict[str, List[str]] = {
    "Documents":     [".pdf", ".doc", ".docx", ".odt", ".rtf", ".tex"],
    "Spreadsheets":  [".xls", ".xlsx", ".csv", ".ods"],
    "Presentations": [".ppt", ".pptx", ".odp"],
    "Images":        [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".heic"],
    "Videos":        [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"],
    "Audio":         [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"],
    "Archives":      [".zip", ".tar", ".gz", ".rar", ".7z", ".bz2"],
    "Code":          [
        ".py", ".js", ".ts", ".html", ".css", ".java", ".cpp", ".c",
        ".h", ".rs", ".go", ".sh", ".bat", ".json", ".yaml", ".yml", ".toml",
    ],
    "Ebooks":        [".epub", ".mobi", ".azw"],
    "Data":          [".db", ".sqlite", ".parquet", ".feather"],
    "Fonts":         [".ttf", ".otf", ".woff", ".woff2"],
    "Executables":   [".exe", ".msi", ".dmg", ".deb", ".AppImage"],
}

AI_CATEGORIES = [
    "Financial", "Legal", "Medical", "Academic",
    "Business", "Personal", "Technical", "Research",
    "Creative", "Reference", "Correspondence",
]

_ORGANIZER_INSTRUCTIONS = (
    "You are a file organisation assistant integrated into Paramodus. "
    "You can scan folders, organise files by category, explain what was organised, "
    "and answer questions about file structure. "
    "When the user asks to organise a folder, call the appropriate function. "
    "Always confirm before moving files unless dry_run mode is active. "
    "Format file trees and category lists as markdown."
)

_LANG_SUFFIXES = {
    "fr": "\nRéponds toujours en français.",
    "es": "\nResponde siempre en español.",
}

# ---------------------------------------------------------------------------
# Thread-safe singletons (Fix #1 — copied from bonsai_agent.py pattern)
# ---------------------------------------------------------------------------

_lock:             threading.RLock  = threading.RLock()
_organizer_agent:  Optional[Agent]   = None
_watch_thread:     Optional[threading.Thread] = None
_watch_stop        = threading.Event()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_db():
    from agno.db.sqlite import SqliteDb
    return SqliteDb(db_file=DB_FILE)


def _get_knowledge() -> Knowledge:
    """
    Return the SHARED knowledge base (same LanceDB table as bonsai_agent).
    Importing bonsai_agent here would be a circular dep so we reconstruct
    the same LanceDb pointer. Both agents use table='bonsai_docs' so manifests
    ingested here are immediately searchable from the main chat agent.
    """
    return Knowledge(
        vector_db=LanceDb(
            table_name="bonsai_docs",
            uri=LANCE_URI,
            embedder=FastEmbedEmbedder(
                id="BAAI/bge-small-en-v1.5",
                dimensions=384,
            ),
        ),
    )


def _read_text_content(file_path: Path, max_chars: int = 4000) -> str:
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf" and _PDF:
            reader = PDFReader(chunking_strategy=DEFAULT_CHUNKER)
            docs = reader.read(str(file_path))
            return " ".join(d.content for d in docs if d.content)[:max_chars]
        if suffix == ".csv" and _CSV:
            reader = CSVReader(chunking_strategy=DEFAULT_CHUNKER)
            docs = reader.read(str(file_path))
            return " ".join(d.content for d in docs if d.content)[:max_chars]
        if suffix in (".txt", ".md", ".py", ".js", ".json", ".yaml",
                      ".yml", ".html", ".rst", ".toml", ".log"):
            return file_path.read_text(errors="ignore")[:max_chars]
        return file_path.stem.replace("_", " ").replace("-", " ")
    except Exception:
        return file_path.stem.replace("_", " ").replace("-", " ")


def _rule_classify(file_path: Path) -> Optional[str]:
    ext = file_path.suffix.lower()
    for category, extensions in EXT_RULES.items():
        if ext in extensions:
            return category
    return None


def _cluster_classify(files_with_content: List[tuple]) -> Dict[str, str]:
    if not _SKLEARN or len(files_with_content) < 3:
        return {}
    names = [p for p, _ in files_with_content]
    texts = [c for _, c in files_with_content]
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", min_df=1)
    try:
        X = vectorizer.fit_transform(texts).toarray()
    except ValueError:
        return {}
    n_clusters = max(2, min(int(len(files_with_content) ** 0.5), len(files_with_content) - 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    feature_names = vectorizer.get_feature_names_out()
    cluster_names: Dict[int, str] = {}
    for cid in range(n_clusters):
        mask = labels == cid
        if not mask.any():
            continue
        scores = X[mask].mean(axis=0)
        top_idx = scores.argsort()[-4:][::-1]
        words = [feature_names[i].capitalize() for i in top_idx if scores[i] > 0]
        cluster_names[cid] = "_".join(words) if words else f"Cluster_{cid}"
    return {name: cluster_names.get(label, "Misc") for name, label in zip(names, labels)}


# ---------------------------------------------------------------------------
# Public scanning API
# ---------------------------------------------------------------------------

def scan_folder(folder_path: str) -> List[Dict[str, Any]]:
    root = Path(folder_path)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    manifest = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            stat = f.stat()
            manifest.append({
                "filename":      f.name,
                "filepath":      str(f),
                "extension":     f.suffix.lower(),
                "size_bytes":    stat.st_size,
                "modified":      datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "category":      None,
                "confidence":    0,
                "strategy_used": None,
                "status":        "Pending",
            })
    return manifest


# ---------------------------------------------------------------------------
# Core organise pipeline
# ---------------------------------------------------------------------------

def organize_folder(
    source_path: str,
    target_path: str,
    strategy: str = "hybrid",
    dry_run: bool = False,
    preserve_originals: bool = True,
    progress_cb: Optional[Callable[[dict], None]] = None,
    ai_model_url: str = "http://127.0.0.1:8081/v1",
) -> List[Dict[str, Any]]:
    manifest = scan_folder(source_path)
    target = Path(target_path)

    # Step 1: Rule pass
    unclassified = []
    for item in manifest:
        fp = Path(item["filepath"])
        rule_cat = _rule_classify(fp)
        if rule_cat:
            item["category"]      = rule_cat
            item["confidence"]    = 95
            item["strategy_used"] = "rule"
        else:
            unclassified.append(item)

    # Step 2: Cluster pass
    if strategy in ("cluster", "hybrid") and unclassified and _SKLEARN:
        readable = []
        for item in unclassified:
            content = _read_text_content(Path(item["filepath"]))
            if content.strip():
                readable.append((item["filepath"], content))
        cluster_map = _cluster_classify(readable)
        for item in unclassified:
            if item["filepath"] in cluster_map:
                item["category"]      = cluster_map[item["filepath"]]
                item["confidence"]    = 72
                item["strategy_used"] = "cluster"

    still_unclassified = [i for i in manifest if i["category"] is None or i["confidence"] < 50]

    # Step 3: AI pass
    if strategy in ("ai", "hybrid") and still_unclassified:
        _ai_batch_classify(still_unclassified, ai_model_url)

    # Step 4: Move / Copy
    for item in manifest:
        if not item.get("category"):
            item["category"]  = "Uncategorized"
            item["confidence"] = 0
        if progress_cb:
            progress_cb(item)
        if dry_run:
            item["status"] = "DryRun"
            continue
        source_file = Path(item["filepath"])
        if not source_file.exists():
            item["status"] = "SourceMissing"
            continue
        dest_dir = target / item["category"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / source_file.name
        counter = 1
        while dest_file.exists():
            dest_file = dest_dir / f"{source_file.stem}_{counter}{source_file.suffix}"
            counter += 1
        try:
            if preserve_originals:
                shutil.copy2(source_file, dest_file)
                item["status"] = "Copied"
            else:
                shutil.move(str(source_file), str(dest_file))
                item["status"] = "Moved"
            item["organized_path"] = str(dest_file)
        except Exception as exc:
            item["status"] = f"Error: {exc}"

    return manifest


def _ai_batch_classify(items: List[Dict], model_url: str) -> None:
    try:
        from agno.models.llama_cpp import LlamaCpp
        classifier = Agent(
            model=LlamaCpp(id="bonsai", base_url=model_url),
            instructions=(
                f"You are a file classification expert. Given a filename and optional content snippet, "
                f"classify into exactly one of: {', '.join(AI_CATEGORIES)}\n\n"
                "Respond ONLY with valid JSON: "
                '{"category": "CategoryName", "confidence": 85, "reason": "brief explanation"}'
            ),
            markdown=False,
            stream=False,
        )
    except Exception:
        return

    for item in items:
        try:
            snippet = _read_text_content(Path(item["filepath"]), max_chars=800)
            prompt = f"Filename: {item['filename']}\nContent: {snippet}"
            response = classifier.run(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(raw[start:end])
                item["category"]      = result.get("category", "General")
                item["confidence"]    = max(0, min(100, int(result.get("confidence", 50))))
                item["strategy_used"] = "ai"
                item["reason"]        = result.get("reason", "")
        except Exception:
            item["category"]      = "General"
            item["confidence"]    = 20
            item["strategy_used"] = "ai_fallback"


# ---------------------------------------------------------------------------
# Watch daemon
# ---------------------------------------------------------------------------

def start_watch(
    source_path: str,
    target_path: str,
    strategy: str = "hybrid",
    interval_seconds: int = 30,
    progress_cb: Optional[Callable[[dict], None]] = None,
) -> None:
    global _watch_thread, _watch_stop
    _watch_stop.clear()
    seen: Dict[str, str] = {}

    def _hash(path: str) -> str:
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""

    def _loop() -> None:
        while not _watch_stop.is_set():
            try:
                manifest = scan_folder(source_path)
                new_items = []
                for item in manifest:
                    fp = item["filepath"]
                    h = _hash(fp)
                    if seen.get(fp) != h:
                        new_items.append(item)
                        seen[fp] = h
                if new_items:
                    organize_folder(
                        source_path=source_path,
                        target_path=target_path,
                        strategy=strategy,
                        dry_run=False,
                        progress_cb=progress_cb,
                    )
            except Exception as exc:
                logger.error("[OrganizerWatch] Error: %s", exc)
            _watch_stop.wait(timeout=interval_seconds)

    _watch_thread = threading.Thread(target=_loop, daemon=True, name="OrganizerWatch")
    _watch_thread.start()
    logger.info("[OrganizerWatch] Started watching: %s", source_path)


def stop_watch() -> None:
    _watch_stop.set()
    logger.info("[OrganizerWatch] Stopped.")


# ---------------------------------------------------------------------------
# Agno agent singleton — thread-safe (Fix #1)
# ---------------------------------------------------------------------------

def init_organizer_agent(model_url: str = "http://127.0.0.1:8081/v1") -> None:
    """
    Build and cache the organiser agent.
    Shares SqliteDb and LanceDb with bonsai_agent.py.
    Safe to call multiple times (idempotent).
    """
    global _organizer_agent
    if _organizer_agent is not None:
        return
    with _lock:
        if _organizer_agent is not None:
            return
        from agno.db.sqlite import SqliteDb
        from agno.models.llama_cpp import LlamaCpp
        _organizer_agent = Agent(
            model=LlamaCpp(id="bonsai", base_url=model_url),
            db=SqliteDb(db_file=DB_FILE),
            memory_manager=MemoryManager(
                db=SqliteDb(db_file=DB_FILE),
                additional_instructions="Remember user's preferred folder structures and naming conventions.",
            ),
            update_memory_on_run=True,
            add_memories_to_context=True,
            add_history_to_context=True,
            instructions=_ORGANIZER_INSTRUCTIONS,
            knowledge=_get_knowledge(),
            search_knowledge=True,
            markdown=True,
        )
        logger.info("[OrganizerAgent] Initialised.")


def get_organizer_agent(session_id: str, language: str = "en") -> tuple[Agent, dict]:
    """
    Return (agent_singleton, run_kwargs).

    CRITICAL FIX: We never mutate _organizer_agent.session_id or
    _organizer_agent.instructions directly (that was the race condition).
    Instead, pass the returned run_kwargs to agent.arun(**run_kwargs).
    """
    if _organizer_agent is None:
        init_organizer_agent()

    lang_suffix = _LANG_SUFFIXES.get(language, "")
    instructions = _ORGANIZER_INSTRUCTIONS + lang_suffix

    run_kwargs = {
        "session_id": session_id,
        "instructions_override": instructions,
    }
    return _organizer_agent, run_kwargs


def ingest_organized_manifest(manifest: List[Dict]) -> bool:
    """
    After organising, push the manifest summary into the SHARED LanceDB
    (same table as bonsai_agent) so the main chat agent can answer
    'where did my invoice go?' without a separate memory store.
    """
    summary_lines = ["# Organised File Manifest\n"]
    by_cat: Dict[str, list] = defaultdict(list)
    for item in manifest:
        by_cat[item.get("category", "Unknown")].append(item["filename"])
    for cat, files in sorted(by_cat.items()):
        summary_lines.append(f"## {cat}")
        for f in files:
            summary_lines.append(f"- {f}")
    summary_text = "\n".join(summary_lines)
    tmp_path = os.path.join(_app_data, "last_manifest.md")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(summary_text)
        kb = _get_knowledge()
        kb.insert(
            path=tmp_path,
            name="last_manifest.md",
            reader=TextReader(chunking_strategy=DEFAULT_CHUNKER) if _TEXT else None,
            metadata={"type": "organizer_manifest"},
            upsert=True,
        )
        logger.info("[OrganizerAgent] Manifest ingested into shared knowledge base.")
        return True
    except Exception as exc:
        logger.error("[OrganizerAgent] Manifest ingest failed: %s", exc)
        return False
