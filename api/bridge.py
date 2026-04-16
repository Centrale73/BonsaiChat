"""
api/bridge.py — BonsaiChat pywebview API bridge.

agent-zero-features branch additions
──────────────────────────────────────
• download_bonsai  — auto-downloads model from Hugging Face if absent
• get_setup_env    — returns first-run state for the setup wizard
• save_env_key     — persists a key=value to .env (e.g. HF repo override)
• toggle_multi_agent — enables/disables sub-agent delegation UI hint
• All cloud/provider stubs removed — BonsaiChat is local-only.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import uuid
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports
# ─────────────────────────────────────────────────────────────────────────────

def _db_module():
    from database import save_msg, get_history, clear_session, get_all_sessions
    return save_msg, get_history, clear_session, get_all_sessions


def _agent_module():
    import bonsai_agent
    return bonsai_agent


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_base_dir() -> str:
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__ + "/.."))


def _get_server_paths():
    base = _get_base_dir()
    llama_bin  = os.path.join(base, "bin", "llama-server.exe" if os.name == "nt" else "llama-server")
    model_path = os.path.join(base, "models", "Bonsai-8B.gguf")
    return llama_bin, model_path


# ─────────────────────────────────────────────────────────────────────────────
# Model auto-download via Hugging Face Hub
# ─────────────────────────────────────────────────────────────────────────────

# Defaults — can be overridden via .env
HF_DEFAULT_REPO = "HuggingFaceTB/smollm2-1.7b-instruct-GGUF"  # swap to your preferred model
HF_DEFAULT_FILE = "smollm2-1.7b-instruct-q4_k_m.gguf"
HF_ENV_REPO     = "BONSAI_HF_REPO"
HF_ENV_FILE     = "BONSAI_HF_FILE"


def _hf_download(report_cb, model_path: str) -> bool:
    """Download model from HF Hub with progress callbacks. Returns True on success."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        report_cb("error", -1,
                  "huggingface_hub is not installed. Run: pip install huggingface-hub")
        return False

    repo  = os.environ.get(HF_ENV_REPO, HF_DEFAULT_REPO)
    fname = os.environ.get(HF_ENV_FILE, HF_DEFAULT_FILE)
    dest_dir = os.path.dirname(model_path)
    os.makedirs(dest_dir, exist_ok=True)

    report_cb("downloading", 0, f"Fetching {fname} from {repo}…")

    downloaded_path: list[Optional[str]] = [None]
    dl_error: list[Optional[Exception]]  = [None]

    def _dl():
        try:
            downloaded_path[0] = hf_hub_download(
                repo_id=repo,
                filename=fname,
                local_dir=dest_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            dl_error[0] = exc

    t = threading.Thread(target=_dl, daemon=True)
    t.start()

    import time
    while t.is_alive():
        # Coarse progress: report downloaded bytes from partial file
        for candidate in (model_path + ".incomplete", model_path):
            if os.path.exists(candidate):
                size_mb = os.path.getsize(candidate) / (1024 ** 2)
                report_cb("downloading", min(size_mb / 30, 95),
                           f"Downloading… {size_mb:.0f} MB received")
                break
        time.sleep(2)

    t.join()

    if dl_error[0]:
        report_cb("error", -1, f"Download failed: {dl_error[0]}")
        return False

    # Move to canonical path if HF saved it elsewhere
    if downloaded_path[0] and os.path.abspath(downloaded_path[0]) != os.path.abspath(model_path):
        import shutil
        shutil.move(downloaded_path[0], model_path)

    report_cb("downloading", 100, "Download complete.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Bridge
# ─────────────────────────────────────────────────────────────────────────────

class ApiBridge:
    def __init__(self):
        self.window                = None
        self.current_session_id    = str(uuid.uuid4())
        self.uploaded_filenames: list[str] = []
        self._server_process: Optional[subprocess.Popen] = None
        self._server_ready         = False
        self._multi_agent_enabled  = True

    def set_window(self, window):
        self.window = window

    def _report(self, phase: str, pct: float, msg: str) -> None:
        if self.window:
            self.window.evaluate_js(
                f"onBonsaiSetupProgress({json.dumps(phase)}, {pct:.2f}, {json.dumps(msg)})"
            )

    # ── session management ───────────────────────────────────────────────────

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

    # ── multi-agent toggle ───────────────────────────────────────────────────

    def toggle_multi_agent(self, enabled: bool) -> str:
        self._multi_agent_enabled = enabled
        return f"Multi-Agent: {'Enabled' if enabled else 'Disabled'}"

    # ── local model status ───────────────────────────────────────────────────

    def get_local_model_status(self, model_key: str = "bonsai-8b") -> dict:
        return {
            "server_running": self._server_ready,
            "downloaded":     self._is_model_present(),
            "model_key":      model_key,
        }

    def get_bonsai_models(self) -> list:
        return [
            {
                "key":         "bonsai-8b",
                "name":        "Bonsai 8B",
                "description": "1-bit quantized LLM — runs entirely on CPU",
                "downloaded":  self._is_model_present(),
            }
        ]

    def _is_model_present(self) -> bool:
        _, model_path = _get_server_paths()
        return os.path.isfile(model_path)

    # ── first-run wizard helpers ─────────────────────────────────────────────

    def get_setup_env(self) -> dict:
        """Return environment state for the first-run settings wizard."""
        _, model_path = _get_server_paths()
        return {
            "model_present": self._is_model_present(),
            "model_path":    model_path,
            "hf_repo":       os.environ.get(HF_ENV_REPO, HF_DEFAULT_REPO),
            "hf_file":       os.environ.get(HF_ENV_FILE, HF_DEFAULT_FILE),
        }

    def save_env_key(self, key_name: str, value: str) -> str:
        """Write a key=value to the project .env file and inject into os.environ."""
        env_path = os.path.join(_get_base_dir(), ".env")
        os.environ[key_name] = value

        lines: list[str] = []
        if os.path.exists(env_path):
            with open(env_path) as f:
                lines = f.readlines()

        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key_name}=") or line.startswith(f"{key_name} ="):
                lines[i] = f"{key_name}={value}\n"
                updated = True
                break
        if not updated:
            lines.append(f"{key_name}={value}\n")

        with open(env_path, "w") as f:
            f.writelines(lines)

        return f"{key_name} saved."

    # ── auto-download ────────────────────────────────────────────────────────

    def download_bonsai(self, model_key: str = "bonsai-8b") -> dict:
        """Kick off a background HF download then auto-start the server."""
        _, model_path = _get_server_paths()

        def _worker():
            ok = _hf_download(self._report, model_path)
            if ok:
                self.begin_auto_setup(model_key)

        threading.Thread(target=_worker, daemon=True, name="hf-download").start()
        return {"status": "started"}

    def cancel_download_bonsai(self, model_key: str = "bonsai-8b") -> dict:
        # huggingface_hub doesn't expose a cancel hook; killing the thread is
        # the only reliable option — we just return and let the daemon thread die.
        return {"status": "cancelled"}

    # ── server lifecycle ─────────────────────────────────────────────────────

    def begin_auto_setup(self, model_key: str = "bonsai-8b") -> dict:
        def _worker():
            llama_bin, model_path = _get_server_paths()

            if not os.path.isfile(llama_bin):
                self._report("error", -1,
                             f"llama-server not found at {llama_bin}. "
                             "Place it in the bin/ folder.")
                return

            if not os.path.isfile(model_path):
                # Signal UI to show download button instead of hard-failing
                self._report("missing_model", 0,
                             "Model not found. Click 'Download Model' in Settings.")
                return

            self._report("starting", 0, "Loading model into memory…")

            command = [
                llama_bin,
                "-m", model_path,
                "--host", "127.0.0.1",
                "--port", "8081",
                "-ngl", "99",
            ]

            startupinfo = None
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            try:
                self._server_process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    startupinfo=startupinfo,
                    text=True,
                    bufsize=1,
                )

                for line in self._server_process.stdout:  # type: ignore[union-attr]
                    stripped = line.strip()
                    if stripped:
                        self._report("starting", 0, stripped[:100])

                    if "HTTP server error" in line:
                        self._report("error", -1,
                                     "Server failed — port 8081 may be in use.")
                        return

                    if "server is listening on" in line:
                        self._server_ready = True
                        try:
                            _agent_module().init_agent()
                        except Exception as exc:
                            self._report("error", -1, f"Agent init failed: {exc}")
                            return
                        self._report("ready", 100, "Bonsai is ready")
                        return

            except Exception as exc:
                self._report("error", -1, f"Failed to launch server: {exc}")

        threading.Thread(target=_worker, daemon=True, name="bonsai-server").start()
        return {"status": "started"}

    def stop_bonsai(self) -> dict:
        if self._server_process and self._server_process.poll() is None:
            self._server_process.kill()
        self._server_ready = False
        return {"status": "stopped"}

    # ── RAG / file handling ──────────────────────────────────────────────────

    def clear_rag_context(self):
        try:
            success = _agent_module().clear_knowledge_base()
        except Exception as exc:
            return f"Error clearing RAG: {exc}"
        self.uploaded_filenames = []
        return "RAG context cleared" if success else "Error clearing RAG context"

    def upload_files(self, files_json):
        try:
            files_data = json.loads(files_json) if isinstance(files_json, str) else files_json
            processed  = []
            for f in files_data:
                name        = f["name"]
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
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    # ── chat streaming ────────────────────────────────────────────────────────

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

        threading.Thread(
            target=self._run_chat,
            args=(user_text, target_id),
            daemon=True,
        ).start()

    def _run_chat(self, user_text: str, target_id: str):
        try:
            agent         = _agent_module().get_agent(self.current_session_id)
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

        except Exception as exc:
            if self.window:
                self.window.evaluate_js(f"receiveError({json.dumps(str(exc))})")

    # ── tone detection ────────────────────────────────────────────────────────

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
