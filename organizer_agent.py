"""
organizer_agent.py — File Organizer backend for BonsaiChat.

Wires the PDF scan → LLM categorize → file organize pipeline
directly through the local Bonsai llama-server already running
on 127.0.0.1:8081. No cloud API keys required.
"""

import json
import os
import re
import shutil
import sys
import threading
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Agno imports (same stack as bonsai_agent.py)
# ---------------------------------------------------------------------------
from agno.agent import Agent
from agno.models.openai import OpenAIChat  # used in local-compat mode


# ---------------------------------------------------------------------------
# Lazy PDF reader — mirrors bonsai_agent.py's pattern
# ---------------------------------------------------------------------------

def _get_pdf_reader():
    try:
        from agno.document.reader.pdfreader import PDFReader
        return PDFReader()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model factory — reuses the llama-server endpoint BonsaiChat already started
# ---------------------------------------------------------------------------

def _local_model():
    """OpenAI-compat shim pointing at the Bonsai llama-server."""
    return OpenAIChat(
        id="bonsai-8b-native",
        api_key="local",
        base_url="http://127.0.0.1:8081/v1",
    )


# ---------------------------------------------------------------------------
# Categories (matches what the Organizer used — easy to extend)
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Financial",
    "Legal",
    "Medical",
    "Academic",
    "Business",
    "Personal",
    "Technical",
    "Other",
]

CATEGORIZE_INSTRUCTIONS = """
You are a document categorization engine. Analyze the document content and return ONLY a JSON object — no prose, no markdown fences.

Categories: Financial, Legal, Medical, Academic, Business, Personal, Technical, Other

Confidence scoring:
- 90-100: Very clear indicators
- 70-89: Strong indicators
- 50-69: Probable
- 30-49: Uncertain
- 0-29: No clear indicators

Required response format (strict JSON, nothing else):
{"category": "Financial", "confidence": 88, "subcategory": "Invoice", "reason": "Contains line-item pricing and vendor details"}
"""


# ---------------------------------------------------------------------------
# Module-level state (one agent instance reused across calls)
# ---------------------------------------------------------------------------

_agent: Optional[Agent] = None
_agent_lock = threading.Lock()


def get_categorizer_agent() -> Agent:
    global _agent
    with _agent_lock:
        if _agent is None:
            _agent = Agent(
                name="Document Categorizer",
                role="Classify PDF documents into structured categories",
                model=_local_model(),
                instructions=CATEGORIZE_INSTRUCTIONS,
                markdown=False,
                stream=False,
            )
    return _agent


# ---------------------------------------------------------------------------
# Core pipeline functions called from bridge.py
# ---------------------------------------------------------------------------

def scan_folder(source_folder: str) -> dict:
    """
    Scan a folder for PDFs and extract their text content.
    Returns a list of document dicts ready for categorization.
    """
    reader = _get_pdf_reader()
    source = Path(source_folder)

    if not source.exists() or not source.is_dir():
        return {"status": "error", "message": f"Folder not found: {source_folder}"}

    pdf_files = list(source.glob("*.pdf"))
    if not pdf_files:
        return {"status": "error", "message": "No PDF files found in the selected folder."}

    documents = []
    errors = []

    for pdf_file in pdf_files:
        try:
            if reader:
                docs = reader.read(str(pdf_file))
                content = " ".join(
                    d.content for d in docs if getattr(d, "content", None)
                )
            else:
                content = ""

            documents.append({
                "filename": pdf_file.name,
                "filepath": str(pdf_file),
                # First 3000 chars for categorization, mirroring the original
                "content_preview": content[:3000],
                "word_count": len(content.split()),
                "category": "Uncategorized",
                "subcategory": "",
                "confidence": 0,
                "reason": "",
                "status": "Scanned",
            })
        except Exception as e:
            errors.append({"filename": pdf_file.name, "error": str(e)})

    return {
        "status": "success",
        "documents": documents,
        "scan_errors": errors,
        "total": len(pdf_files),
        "readable": len(documents),
    }


def categorize_document(doc: dict) -> dict:
    """
    Run a single document dict through the Bonsai LLM and return an updated dict.
    Called per-document so the bridge can stream progress back to the UI.
    """
    agent = get_categorizer_agent()

    prompt = (
        f"Document filename: {doc['filename']}\n"
        f"Content:\n{doc['content_preview']}"
    )

    try:
        response = agent.run(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip()

        # Extract JSON robustly
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
            doc["category"] = parsed.get("category", "Other").strip()
            doc["subcategory"] = parsed.get("subcategory", "").strip()
            doc["confidence"] = max(0, min(100, int(parsed.get("confidence", 50))))
            doc["reason"] = parsed.get("reason", "").strip()
            doc["status"] = "Categorized"
        else:
            doc["category"] = "Other"
            doc["confidence"] = 20
            doc["reason"] = "Could not parse model response."
            doc["status"] = "Parse Error"

    except Exception as e:
        doc["category"] = "Other"
        doc["confidence"] = 0
        doc["reason"] = str(e)
        doc["status"] = "Error"

    return doc


def organize_documents(documents: list, target_folder: str, copy_files: bool = True) -> dict:
    """
    Move or copy categorized documents into target_folder/<Category>/<Subcategory>/.
    Returns a summary dict.
    """
    target = Path(target_folder)
    organized = []
    failed = []

    for doc in documents:
        category = doc.get("category", "Other")
        if category in ("Uncategorized", "Error", "Parse Error", ""):
            category = "Other"

        try:
            dest_dir = target / category
            subcategory = doc.get("subcategory", "").strip()
            if subcategory:
                dest_dir = dest_dir / subcategory

            dest_dir.mkdir(parents=True, exist_ok=True)

            source_file = Path(doc["filepath"])
            if not source_file.exists():
                failed.append({"filename": doc["filename"], "error": "Source file missing"})
                continue

            dest_file = dest_dir / source_file.name

            # Deduplicate names
            counter = 1
            orig = dest_file
            while dest_file.exists():
                dest_file = dest_dir / f"{orig.stem}_{counter}{orig.suffix}"
                counter += 1

            if copy_files:
                shutil.copy2(source_file, dest_file)
                action = "Copied"
            else:
                shutil.move(str(source_file), str(dest_file))
                action = "Moved"

            doc["status"] = action
            doc["organized_path"] = str(dest_file)
            organized.append(doc)

        except Exception as e:
            failed.append({"filename": doc["filename"], "error": str(e)})

    return {
        "status": "success",
        "organized": len(organized),
        "failed": len(failed),
        "details": organized,
        "errors": failed,
    }
