"""
api/bridge.py — BonsaiChat pywebview API bridge.

Exposes all methods the HTML/JS frontend calls via window.pywebview.api.
Adapted from Paramodus' bridge but targeting BonsaiChat's simpler, local-only
backend (llama-server + Agno + LanceDB RAG — no cloud provider switching).
"""

import base64
import json
import os
import requests
import subprocess
import sys
import threading
import uuid
from typing import Optional


def get_resource_path(relative_path: str) -> str:
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # For bridge.py located in api/, we go up one level
        base_path = os.path.dirname(os.path.abspath(__file__ + "/.."))
    return os.path.join(base_path, relative_path)



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

def _get_server_paths():
    """
    Finds the llama-server binary and the GGUF model.
    Checks:
    1. Local folder (for installed production apps)
    2. _MEIPASS (for bundled assets, though we exclude large ones in Option 1)
    3. Dev source folder
    """
    if getattr(sys, 'frozen', False):
        # Running as a compiled .exe
        exe_dir = os.path.dirname(sys.executable)
    else:
        # Running as a script
        exe_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))
    
    # --- BINARIES ---
    # Check next to the EXE first (typical for Option 1 install)
    local_bin = os.path.join(exe_dir, "bin", "llama-server.exe" if os.name == "nt" else "llama-server")
    # Check internal _MEIPASS (pyinstaller bundle)
    internal_bin = get_resource_path(os.path.join("bin", "llama-server.exe" if os.name == "nt" else "llama-server"))
    
    if os.path.exists(local_bin):
        llama_bin = local_bin
    else:
        llama_bin = internal_bin

    # --- MODELS ---
    # Check next to the EXE first
    local_model = os.path.join(exe_dir, "models", "Bonsai-8B.gguf")
    # Check internal _MEIPASS
    internal_model = get_resource_path(os.path.join("models", "Bonsai-8B.gguf"))
    # Also check AppData for previously downloaded models
    appdata_dir = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), "Paramodus", "models")
    appdata_model = os.path.join(appdata_dir, "Bonsai-8B.gguf")
    
    if os.path.exists(local_model):
        model_path = local_model
    elif os.path.exists(appdata_model):
        model_path = appdata_model
    else:
        model_path = internal_model
        
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
        # Update the agent if it exists
        try:
            agent = _agent_module().get_agent(self.current_session_id, language)
            # instructions are updated inside get_agent()
        except Exception:
            pass
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
                "name": "Paramodus 8B",
                "description": "1-bit quantized LLM — runs entirely on CPU",
                "downloaded": self._is_model_present(),
            }
        ]

    def _is_model_present(self) -> bool:
        _, model_path = _get_server_paths()
        return os.path.isfile(model_path)

    def _download_model(self, url: str, save_path: str, report_cb):
        """Downloads the model while reporting progress to the UI."""
        # Use APPDATA for downloads so they persist even if the app is re-installed or moved
        appdata_dir = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), "Paramodus", "models")
        os.makedirs(appdata_dir, exist_ok=True)
        persistent_path = os.path.join(appdata_dir, os.path.basename(save_path))
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(persistent_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            # Report progress every ~1% or at the end
                            if int(pct) % 2 == 0 or downloaded == total_size:
                                report_cb("downloading", pct, f"Downloading Model: {pct:.1f}%")

            return True
        except Exception as e:
            report_cb("error", -1, f"Download failed: {e}")
            return False

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
                _report("downloading", 0, "Model not found. Starting download from Hugging Face...")
                model_url = "https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B.gguf"
                success = self._download_model(model_url, model_path, _report)
                if not success:
                    return # Error already reported by helper

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
                        _report("ready", 100, "Paramodus is ready")
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
                "message": "Paramodus does not auto-download. Place the .gguf in models/."}

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
