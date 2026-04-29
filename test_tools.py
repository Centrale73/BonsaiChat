# test_tools.py — v4: self-contained with server lifecycle
import json, time, os, sys, subprocess, requests
from agno.agent import Agent
from agno.models.llama_cpp import LlamaCpp
from agno.tools.duckduckgo import DuckDuckGoTools

# ── paths (relative to repo root, same logic as bridge.py) ──────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
LLAMA_BIN  = os.path.join(ROOT, "bin", "llama-server.exe" if os.name == "nt" else "llama-server")
MODEL_PATH = os.path.join(ROOT, "models", "Bonsai-8B.gguf")
HOST       = "127.0.0.1"
PORT       = 8081
BASE_URL   = f"http://{HOST}:{PORT}/v1"

# ── server helpers ───────────────────────────────────────────────────────────
def start_server() -> subprocess.Popen:
    print(f"[server] Starting llama-server on {HOST}:{PORT} ...")
    proc = subprocess.Popen(
        [LLAMA_BIN, "-m", MODEL_PATH,
         "--host", HOST, "--port", str(PORT),
         "-ngl", "99",
         "--log-disable"],          # suppress llama.cpp wall-of-text
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc

def wait_for_server(timeout: int = 120) -> bool:
    """Poll /v1/models until the server is ready or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{HOST}:{PORT}/v1/models", timeout=2)
            if r.status_code == 200:
                print(f"[server] Ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print(f"[server] Waiting... ({int(deadline - time.time())}s left)", end="\r")
        time.sleep(2)
    return False

def stop_server(proc: subprocess.Popen):
    print("\n[server] Shutting down...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("[server] Stopped.")

# ── main ─────────────────────────────────────────────────────────────────────
if not os.path.isfile(LLAMA_BIN):
    print(f"ERROR: llama-server not found at {LLAMA_BIN}")
    sys.exit(1)

if not os.path.isfile(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

server_proc = start_server()

if not wait_for_server(timeout=120):
    print("ERROR: Server failed to start within 120s")
    stop_server(server_proc)
    sys.exit(1)

# ── agent setup ──────────────────────────────────────────────────────────────
agent = Agent(
    model=LlamaCpp(
        id="bonsai-8b",
        base_url=BASE_URL,
    ),
    tools=[DuckDuckGoTools()],
    debug_mode=True,
    markdown=False,
)

TESTS = [
    ("L1-explicit",  "Use the search tool to find today's weather in Montreal."),
    ("L2-implicit",  "What is the current price of Bitcoin?"),
    ("L4-trap",      "What is the capital of France?"),
]

results = []

try:
    for label, prompt in TESTS:
        print(f"\n{'='*60}")
        print(f"TEST {label}")
        print(f"PROMPT: {prompt}")
        print("-" * 60)
        start = time.time()

        try:
            response = agent.run(prompt)
            elapsed = round(time.time() - start, 2)

            content    = getattr(response, "content", None)
            messages   = getattr(response, "messages", None) or []
            tools_used = getattr(response, "tools", None) or []
            tool_fired = False

            print(f"content    : {str(content)[:300]}")
            print(f"tools_used : {tools_used}")

            for i, msg in enumerate(messages):
                role = getattr(msg, "role", "?")
                tc   = getattr(msg, "tool_calls", None)
                mc   = str(getattr(msg, "content", ""))[:120]
                print(f"  msg[{i}] role={role}  tc={bool(tc)}  content={mc}")
                if tc or role == "tool":
                    tool_fired = True

            results.append({
                "test": label,
                "tool_fired": tool_fired,
                "elapsed_s": elapsed,
                "content": str(content)[:300] if content else None,
            })

        except Exception as e:
            import traceback
            elapsed = round(time.time() - start, 2)
            print(f"ERROR: {e}")
            traceback.print_exc()
            results.append({"test": label, "tool_fired": False,
                            "elapsed_s": elapsed, "error": str(e)})

finally:
    stop_server(server_proc)

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print("-" * 60)
for r in results:
    status = "✅ tool" if r.get("tool_fired") else ("❌ error" if "error" in r else "⚠️  no tool")
    print(f"  {r['test']:15} {status}  ({r.get('elapsed_s')}s)")
    if r.get("content"):
        print(f"    → {r['content'][:120]}")

with open("tool_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to tool_test_results.json")