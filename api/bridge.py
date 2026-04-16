"""
api/bridge.py — BonsaiChat pywebview API bridge.

Exposes all methods the HTML/JS frontend calls via window.pywebview.api.
Adapted from Paramodus' bridge but targeting BonsaiChat's simpler, local-only
backend (llama-server + Agno + LanceDB RAG — no cloud provider switching).
"""

import base64
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from typing import Optional


# ---------------------------------------------------------------------------
# Lazy imports — same pattern as Paramodus so the window opens instantly
# ---------------------------------------------------------------------------

def _db_module():
    from database import save_msg, get_history, clear_session, get_all_sessions
    return save_msg, get_history, clear_session, get_all_sessions


def _agent_module():
    import bonsai_agent
    return bonsai_agent


# ---------------------------------------------------------------------------
# Server path helpers
# ---------------------------------------------------------------------------

def _get_base_dir() -> str:
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__ + "/.."))


def _get_server_paths():
    base = _get_base_dir()
    llama_bin = os.path.join(base, "bin", "llama-server.exe" if os.name == "nt" else "llama-server")
    model_path = os.path.join(base, "models", "Bonsai-8B.gguf")
    return llama_bin, model_path


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class ApiBridge:
    def __init__(self):
        self.window = None
        self.current_session_id = str(uuid.uuid4())
        self.uploaded_filenames: list[str] = []
        self._server_process: Optional[subprocess.Popen] = None
        self._server_ready = False
        self._agent = None
        self.current_language = "en"

    def set_window(self, window):
        self.window = window

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def new_session(self):
        self.current_session_id = str(uuid.uuid4())
        return {"status": "success", "session_id": self.current_session_id}

    def list_sessions(self):
        _, get_history, _, get_all_sessions = _db_module()
        return get_all_sessions()

    def switch_session(self, session_id: str):
        self.current_session_id = session_id
        return {"status": "success", "session_id": session_id}

    def get_current_session_id(self):
        return self.current_session_id

    def load_history(self):
        _, get_history, _, _ = _db_module()
        return get_history(self.current_session_id)

    # ------------------------------------------------------------------
    # Settings & Configuration
    # ------------------------------------------------------------------

    def set_language(self, language: str):
        self.current_language = language
        return f"Language set to: {language.upper()}"

    # ------------------------------------------------------------------
    # Local model / server lifecycle
    # ------------------------------------------------------------------

    def get_local_model_status(self, model_key: str = "bonsai-8b") -> dict:
        return {
            "server_running": self._server_ready,
            "downloaded": self._is_model_present(),
            "model_key": model_key,
        }

    def get_bonsai_models(self) -> list:
        return [
            {
                "key": "bonsai-8b",
                "name": "Bonsai 8B",
                "description": "1-bit quantized LLM — runs entirely on CPU",
                "downloaded": self._is_model_present(),
            }
        ]

    def _is_model_present(self) -> bool:
        _, model_path = _get_server_paths()
        return os.path.isfile(model_path)

    # begin_auto_setup — called by JS on pywebviewready
    def begin_auto_setup(self, model_key: str = "bonsai-8b") -> dict:
        def _report(phase: str, pct: float, msg: str):
            if self.window:
                self.window.evaluate_js(
                    f"onBonsaiSetupProgress({json.dumps(phase)}, {pct:.2f}, {json.dumps(msg)})"
                )

        def _worker():
            llama_bin, model_path = _get_server_paths()

            # 1. Binary check
            if not os.path.isfile(llama_bin):
                _report("error", -1,
                        f"llama-server not found at {llama_bin}. "
                        "Place it in the bin/ folder.")
                return

            # 2. Model check
            if not os.path.isfile(model_path):
                _report("error", -1,
                        f"Model not found at {model_path}. "
                        "Place Bonsai-8B.gguf in the models/ folder.")
                return

            # 3. Start server
            _report("starting", 0, "Loading model into memory…")

            command = [
                llama_bin,
                "-m", model_path,
                "--host", "127.0.0.1",
                "--port", "8081",
                "-ngl", "99",
            ]

            startupinfo = None
            if os.name == "nt":
                import subprocess as _sp
                startupinfo = _sp.STARTUPINFO()
                startupinfo.dwFlags |= _sp.STARTF_USESHOWWINDOW

            try:
                self._server_process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    startupinfo=startupinfo,
                    text=True,
                    bufsize=1,
                )

                for line in self._server_process.stdout:
                    stripped = line.strip()
                    if stripped:
                        _report("starting", 0, stripped[:90])

                    if "HTTP server error" in line:
                        _report("error", -1,
                                "Server failed to start — port 8081 may be in use.")
                        return

                    if "server is listening on" in line:
                        self._server_ready = True
                        # Init the Agno agent now that the server is up
                        try:
                            _agent_module().init_agent()
                        except Exception as e:
                            _report("error", -1, f"Agent init failed: {e}")
                            return
                        _report("ready", 100, "Bonsai is ready")
                        return

            except Exception as e:
                _report("error", -1, f"Failed to launch server: {e}")

        threading.Thread(target=_worker, daemon=True, name="bonsai-server").start()
        return {"status": "started"}

    def stop_bonsai(self) -> dict:
        if self._server_process and self._server_process.poll() is None:
            self._server_process.kill()
        self._server_ready = False
        return {"status": "stopped"}

    # Stub out download methods (model is shipped or user places it manually)
    def download_bonsai(self, model_key: str = "bonsai-8b") -> dict:
        return {"status": "not_applicable",
                "message": "BonsaiChat does not auto-download. Place the .gguf in models/."}

    def cancel_download_bonsai(self, model_key: str = "bonsai-8b") -> dict:
        return {"status": "not_applicable"}

    # ------------------------------------------------------------------
    # RAG / file handling
    # ------------------------------------------------------------------

    def clear_rag_context(self):
        try:
            success = _agent_module().clear_knowledge_base()
        except Exception as e:
            return f"Error clearing RAG: {e}"
        self.uploaded_filenames = []
        return "RAG context cleared" if success else "Error clearing RAG context"

    def upload_files(self, files_json):
        try:
            files_data = json.loads(files_json) if isinstance(files_json, str) else files_json
            processed = []
            for f in files_data:
                name = f["name"]
                content_b64 = f["content"]
                if "," in content_b64:
                    content_b64 = content_b64.split(",")[1]
                data = base64.b64decode(content_b64)
                processed.append({"name": name, "data": data})
                self.uploaded_filenames.append(name)

            success = _agent_module().ingest_files(processed)
            if success:
                return {"status": "success", "files": list(set(self.uploaded_filenames))}
            return {"status": "error", "message": "Failed to ingest files"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Chat streaming
    # ------------------------------------------------------------------

    def start_chat_stream(self, user_text: str, target_id: str = None):
        if not self._server_ready:
            if self.window:
                self.window.evaluate_js(
                    "receiveError('Bonsai is still starting up — please wait a moment.')"
                )
            return

        if not target_id:
            save_msg, _, _, _ = _db_module()
            save_msg("user", user_text, self.current_session_id)

        t = threading.Thread(
            target=self._run_chat,
            args=(user_text, target_id),
            daemon=True,
        )
        t.start()

    def _run_chat(self, user_text: str, target_id: str):
        try:
            agent = _agent_module().get_agent(self.current_session_id, self.current_language)
            full_response = ""

            run_response = agent.run(user_text, stream=True)

            if target_id and self.window:
                self.window.evaluate_js(f"clearBubble('{target_id}')")

            for chunk in run_response:
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    full_response += content
                    if self.window:
                        self.window.evaluate_js(
                            f"receiveChunk({json.dumps(content)}, '{target_id or ''}')"
                        )

            save_msg, _, _, _ = _db_module()
            save_msg("bot", full_response, self.current_session_id)

            tone = self._detect_tone(full_response)
            if self.window:
                self.window.evaluate_js(f"streamComplete({json.dumps(tone)})")

        except Exception as e:
            if self.window:
                self.window.evaluate_js(f"receiveError({json.dumps(str(e))})")

    # ------------------------------------------------------------------
    # File Organizer
    # ------------------------------------------------------------------

    def organizer_scan(self, source_folder: str) -> dict:
        """
        Scan a folder for PDFs and return document list.
        Called from the Organizer tab after the user picks a source folder.
        """
        try:
            import organizer_agent
            return organizer_agent.scan_folder(source_folder)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def organizer_categorize_all(self, documents_json: str) -> dict:
        """
        Categorize all scanned documents via the local Bonsai model.
        Streams per-document progress back to JS via onOrganizerProgress.
        documents_json: JSON string of the documents list from organizer_scan.
        """
        try:
            import organizer_agent
            documents = json.loads(documents_json) if isinstance(documents_json, str) else documents_json

            results = []
            total = len(documents)

            for i, doc in enumerate(documents):
                updated = organizer_agent.categorize_document(doc)
                results.append(updated)

                # Push progress to JS
                if self.window:
                    payload = json.dumps({
                        "index": i,
                        "total": total,
                        "doc": updated,
                    })
                    self.window.evaluate_js(f"onOrganizerProgress({payload})")

            return {"status": "success", "documents": results}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def organizer_categorize_all_async(self, documents_json: str):
        """
        Non-blocking version — runs categorization in a background thread
        so the UI stays responsive during long batch runs.
        """
        threading.Thread(
            target=self._organizer_categorize_worker,
            args=(documents_json,),
            daemon=True,
            name="organizer-categorize",
        ).start()
        return {"status": "started"}

    def _organizer_categorize_worker(self, documents_json: str):
        try:
            import organizer_agent
            documents = json.loads(documents_json) if isinstance(documents_json, str) else documents_json
            total = len(documents)

            for i, doc in enumerate(documents):
                updated = organizer_agent.categorize_document(doc)

                if self.window:
                    payload = json.dumps({"index": i, "total": total, "doc": updated})
                    self.window.evaluate_js(f"onOrganizerProgress({payload})")

            if self.window:
                self.window.evaluate_js("onOrganizerCategorizeComplete()")

        except Exception as e:
            if self.window:
                self.window.evaluate_js(
                    f"onOrganizerError({json.dumps(str(e))})"
                )

    def organizer_organize(self, documents_json: str, target_folder: str, copy_files: bool = True) -> dict:
        """
        Move/copy categorized documents into the target folder structure.
        """
        try:
            import organizer_agent
            documents = json.loads(documents_json) if isinstance(documents_json, str) else documents_json
            return organizer_agent.organize_documents(documents, target_folder, copy_files)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Tone detection (carried over from Paramodus)
    # ------------------------------------------------------------------

    def _detect_tone(self, text: str) -> str:
        t = text.lower()
        scores = {"excited": 0, "playful": 0, "serious": 0, "calm": 0}

        for w in ["!", "amazing", "awesome", "fantastic", "great", "excellent",
                  "wonderful", "exciting", "incredible", "brilliant", "love"]:
            scores["excited"] += t.count(w)

        for w in ["😊", "😄", "🎉", "haha", "fun", "enjoy", "play", "joke",
                  "funny", "silly", "cool", "👍", "✨"]:
            scores["playful"] += t.count(w)

        for w in ["important", "critical", "warning", "caution", "error",
                  "must", "should", "require", "necessary", "essential",
                  "security", "risk", "issue", "problem", "careful"]:
            scores["serious"] += t.count(w)

        for w in ["here", "let me", "simply", "just", "easy", "step", "guide",
                  "help", "explain", "understand", "note", "consider"]:
            scores["calm"] += t.count(w)

        return max(scores, key=scores.get) if max(scores.values()) > 0 else "calm"
