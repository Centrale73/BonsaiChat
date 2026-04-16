"""
tools/code_exec.py — Sandboxed Python code execution tool.

Inspired by Agent Zero's _code_execution plugin.  Executes arbitrary Python
in an isolated temporary directory with a configurable timeout.  stdout/stderr
are captured and returned to the LLM.

Security note: this is a LOCAL-only tool.  It executes code with the same
privileges as the BonsaiChat process.  Do not expose it over a network
without additional sandboxing (e.g. Docker, restricted user account).
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional

from agno.tools import Toolkit


TIMEOUT_SECONDS = 30


def _run_with_timeout(fn, timeout: int):
    """Run fn() in a thread; return (result, error) after at most `timeout` s."""
    result = [None]
    error  = [None]

    def _target():
        try:
            result[0] = fn()
        except Exception as exc:
            error[0] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        return None, TimeoutError(f"Code execution exceeded {timeout}s timeout.")
    return result[0], error[0]


class CodeExecTools(Toolkit):
    """Agno Toolkit for executing Python code in a sandboxed temp directory."""

    def __init__(self, timeout: int = TIMEOUT_SECONDS):
        super().__init__(name="code_exec")
        self.timeout = timeout
        self.register(self.run_python)

    def run_python(self, code: str) -> str:
        """
        Execute a Python code snippet and return its stdout/stderr output.

        The code runs in an isolated temporary directory.  Any files written
        during execution are accessible via relative paths and cleaned up
        automatically unless the code explicitly moves them elsewhere.

        Args:
            code: Valid Python source code to execute.

        Returns:
            Combined stdout + stderr output, or an error/traceback string.
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        orig_dir   = os.getcwd()

        def _exec():
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                try:
                    exec_globals: dict = {"__name__": "__main__"}
                    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                        exec(compile(code, "<bonsai_exec>", "exec"), exec_globals)
                finally:
                    os.chdir(orig_dir)

        _, err = _run_with_timeout(_exec, self.timeout)

        out = stdout_buf.getvalue()
        err_text = stderr_buf.getvalue()

        if err:
            tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
            return f"[EXECUTION ERROR]\n{tb}\n[STDOUT]\n{out}\n[STDERR]\n{err_text}"

        combined = ""
        if out:
            combined += f"[STDOUT]\n{out}"
        if err_text:
            combined += f"\n[STDERR]\n{err_text}"

        return combined.strip() or "[No output]"
