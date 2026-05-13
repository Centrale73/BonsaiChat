"""
api/bridge.py — BonsaiChat pywebview API bridge.

Exposes all methods the HTML/JS frontend calls via window.pywebview.api.
Adapted from Paramodus' bridge but targeting BonsaiChat's simpler, local-only
backend (llama-server + Agno + LanceDB RAG — no cloud provider switching).
"""

import asyncio
import base64
import json
import os
import requests
import subprocess
import sys
import threading
import uuid
import asyncio
import queue
import time
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
        self._window = None
        self._server_process = None
        self._server_ready = False
        self.current_session_id = str(uuid.uuid4())
        self.current_language = 'en'
        self.current_space_id = None
        self.uploaded_filenames = []

        # Background Event Loop for async Agent runs
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True, name="api-loop")
        self._loop_thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def set_window(self, window):
        self._window = window

    def on_window_closed(self):
        """Called when window closes to stop the background loop."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self.stop_bonsai()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def new_session(self, space_id=None):
        self.current_session_id = str(uuid.uuid4())
        
        if space_id:
            import database
            database.set_session_space(self.current_session_id, space_id)
            self.current_space_id = space_id
            
        return {"status": "success", "session_id": self.current_session_id}

    def list_sessions(self):
        _, get_history, _, get_all_sessions = _db_module()
        return get_all_sessions()

    def switch_session(self, session_id: str):
        self.current_session_id = session_id
        import database
        self.current_space_id = database.get_session_space(session_id)
        return {"status": "success", "session_id": session_id}

    def get_current_session_id(self):
        return self.current_session_id

    def load_history(self):
        _, get_history, _, _ = _db_module()
        return get_history(self.current_session_id)

    def delete_session(self, session_id: str):
        _, _, clear_session, _ = _db_module()
        clear_session(session_id)
        if self.current_session_id == session_id:
            self.new_session()
        return {"status": "success", "session_id": session_id}

    # ------------------------------------------------------------------
    # Settings & Configuration
    # ------------------------------------------------------------------

    def set_language(self, language: str):
        self.current_language = language
        return {"status": "success"}

    # ------------------------------------------------------------------
    # Spaces management
    # ------------------------------------------------------------------

    def create_space(self, name, description, instructions):
        import database
        space_id = str(uuid.uuid4())
        database.create_space(space_id, name, description, instructions)
        return {"status": "success", "space_id": space_id}

    def list_spaces(self):
        import database
        return database.get_spaces()

    def delete_space(self, space_id):
        import database
        database.delete_space(space_id)
        if self.current_space_id == space_id:
            self.current_space_id = None
        return {"status": "success"}
        
    def get_current_space_id(self):
        return self.current_space_id

    def get_crm_settings(self):
        import crm.scheduler
        return {"google_calendar_enabled": crm.scheduler.GOOGLE_CALENDAR_ENABLED}

    def set_crm_settings(self, enabled: bool):
        try:
            import dotenv
            env_file = os.path.join(os.path.dirname(os.path.abspath(__file__ + "/..")), ".env")
            if not os.path.exists(env_file):
                open(env_file, 'a').close()
            dotenv.set_key(env_file, "GOOGLE_CALENDAR_ENABLED", "true" if enabled else "false")
            os.environ["GOOGLE_CALENDAR_ENABLED"] = "true" if enabled else "false"
            
            import crm.scheduler
            crm.scheduler.GOOGLE_CALENDAR_ENABLED = enabled
            
            # Restart scheduler if we're enabling it, so it triggers the OAuth flow or starts polling
            if enabled:
                from crm import start_scheduler
                start_scheduler()
                
            return {"status": "success", "enabled": enabled}
        except Exception as e:
            return {"status": "error", "message": str(e)}

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
            if self._window:
                self._window.evaluate_js(
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

            success = asyncio.run(_agent_module().aingest_files(processed))
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
            if self._window:
                self._window.evaluate_js(
                    "receiveError('Bonsai is still starting up — please wait a moment.')"
                )
            return

        if not target_id:
            save_msg, _, _, _ = _db_module()
            save_msg("user", user_text, self.current_session_id)

        # Schedule the async run in our background loop
        asyncio.run_coroutine_threadsafe(
            self._run_chat_async(user_text, target_id),
            self._loop
        )

    async def _run_chat_async(self, user_text: str, target_id: str):
        try:
            agent = _agent_module().get_agent(self.current_session_id, self.current_language)
            full_response = ""
            
            # Fetch space instructions if any
            import database
            space_instructions = ""
            if self.current_space_id:
                space = database.get_space(self.current_space_id)
                if space and space["instructions"]:
                    space_instructions = space["instructions"]
            
            run_kwargs = _agent_module().get_run_kwargs(self.current_session_id, self.current_language, space_instructions)
            run_response = agent.arun(user_text, stream=True, **run_kwargs)

            if target_id and self._window:
                self._window.evaluate_js(f"clearBubble('{target_id}')")

            # Batch chunks to avoid "Maximum recursion depth" in pywebview
            chunk_buffer = ""
            last_send_time = time.time()

            async for chunk in run_response:
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    full_response += content
                    chunk_buffer += content
                    
                    # Send every 50ms or if buffer is large
                    if time.time() - last_send_time > 0.05 or len(chunk_buffer) > 100:
                        if self._window:
                            self._window.evaluate_js(
                                f"receiveChunk({json.dumps(chunk_buffer)}, '{target_id or ''}')"
                            )
                        chunk_buffer = ""
                        last_send_time = time.time()

            # Send remaining buffer
            if chunk_buffer and self._window:
                self._window.evaluate_js(
                    f"receiveChunk({json.dumps(chunk_buffer)}, '{target_id or ''}')"
                )

            save_msg, _, _, _ = _db_module()
            save_msg("bot", full_response, self.current_session_id)

            tone = self._detect_tone(full_response)
            if self._window:
                self._window.evaluate_js(f"streamComplete({json.dumps(tone)})")

        except Exception as e:
            if self._window:
                self._window.evaluate_js(f"receiveError({json.dumps(str(e))})")

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
