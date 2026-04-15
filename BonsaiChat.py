import customtkinter as ctk
import subprocess
import threading
import sys
import os
import atexit
import time
import tkinter.filedialog as filedialog
from typing import List, Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.llama_cpp import LlamaCpp
from agno.memory import MemoryManager
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.reader.csv_reader import CSVReader
from agno.knowledge.reader.text_reader import TextReader
from agno.knowledge.chunking.recursive import RecursiveChunking

# --- RAG configuration ---
app_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_data")
os.makedirs(app_data, exist_ok=True)
LANCE_URI = os.path.join(app_data, "lancedb")
DEFAULT_CHUNKER = RecursiveChunking(chunk_size=1000, overlap=150)

_knowledge: Optional[Knowledge] = None

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

def ingest_files(file_paths: List[str]) -> bool:
    ingested_count = 0
    for path in file_paths:
        name = os.path.basename(path)
        try:
            if name.lower().endswith(".pdf"):
                reader = PDFReader(chunking_strategy=DEFAULT_CHUNKER)
            elif name.lower().endswith(".csv"):
                reader = CSVReader(chunking_strategy=DEFAULT_CHUNKER)
            elif name.lower().endswith((".txt", ".md", ".py", ".js", ".json")):
                reader = TextReader(chunking_strategy=DEFAULT_CHUNKER)
            else:
                print(f"Unsupported file type: {name}")
                continue

            _get_knowledge().insert(
                path=path,
                name=name,
                reader=reader,
                metadata={"filename": name},
                upsert=True,
            )
            ingested_count += 1
        except Exception as e:
            print(f"Error processing {name}: {e}")

    return ingested_count > 0

def clear_knowledge_base() -> bool:
    try:
        if _get_knowledge().vector_db.exists():
            _get_knowledge().vector_db.drop()
        _get_knowledge().vector_db.create()
        return True
    except Exception as e:
        print(f"Error clearing knowledge base: {e}")
        return False

# Configure UI theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

class BonsaiChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Bonsai Chat (1-bit LLM + RAG)")
        self.geometry("900x600")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0) # Sidebar should not expand
        self.grid_columnconfigure(1, weight=1) # Main frame should expand
        
        # Sidebar Frame for RAG Controls
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Knowledge Base", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 20))
        
        self.upload_btn = ctk.CTkButton(self.sidebar_frame, text="Upload File", command=self.upload_document)
        self.upload_btn.grid(row=1, column=0, padx=20, pady=10)
        
        self.clear_kb_btn = ctk.CTkButton(self.sidebar_frame, text="Clear Data", command=self.clear_knowledge, fg_color="#a83232", hover_color="#822020")
        self.clear_kb_btn.grid(row=2, column=0, padx=20, pady=10)
        
        # Main Frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Chat History Textbox
        self.chat_history = ctk.CTkTextbox(self.main_frame, wrap="word", font=("Segoe UI", 14))
        self.chat_history.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=(10, 0))
        self.chat_history.insert("0.0", "System: Starting background server. Please wait, this might take a few seconds...\n\n")
        self.chat_history.configure(state="disabled")
        
        # Input Frame
        self.input_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        self.input_frame.grid_columnconfigure(0, weight=1)
        
        # Entry field
        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Type your message...", height=40, font=("Segoe UI", 14))
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.entry.bind("<Return>", lambda event: self.send_message())
        
        # Send button
        self.send_btn = ctk.CTkButton(self.input_frame, text="Send", width=80, height=40, command=self.send_message)
        self.send_btn.grid(row=0, column=1)

        # State Variables
        self.server_process = None
        self.is_server_ready = False
        self.user_id = "local_dev_user"
        
        # Setup local SQLite database for persistence
        self.db = SqliteDb(db_file="paramodus_memory.db")

        # Configure the Memory Manager
        self.memory_manager = MemoryManager(
            db=self.db,
            additional_instructions="Extract strictly factual statements about the user's preferences, projects, and constraints. Do not store conversational filler."
        )

        # We defer Agent initialization to start_server thread to prevent GUI lockup
        self.agent = None

        # Launch server
        threading.Thread(target=self.start_server, daemon=True).start()

    def init_agent(self):
        """Initialize the agent with RAG after server loads."""
        self.agent = Agent(
            model=LlamaCpp(
                id="bonsai-8b", 
                base_url="http://127.0.0.1:8081/v1" 
            ),
            db=self.db,
            memory_manager=self.memory_manager,
            update_memory_on_run=True,
            add_memories_to_context=True,
            add_history_to_context=True,
            instructions="You are a helpful and intelligent assistant with access to uploaded documents. Format all mathematical equations using proper LaTeX syntax (e.g., $$...$$ for block equations).",
            knowledge=_get_knowledge(),
            search_knowledge=True,
            markdown=True
        )

    def determine_paths(self):
        if getattr(sys, 'frozen', False):
            # Running as compiled .exe
            base_dir = os.path.dirname(sys.executable)
        else:
            # Running as script
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
        llama_bin = os.path.join(base_dir, "bin", "llama-server.exe")
        model = os.path.join(base_dir, "models", "Bonsai-8B.gguf")
        return llama_bin, model

    def start_server(self):
        llama_bin, model_path = self.determine_paths()
        
        if not os.path.exists(llama_bin) or not os.path.exists(model_path):
            self.append_text(f"Error: Missing binary or model!\nBin expected at: {llama_bin}\nModel expected at: {model_path}\n", "System")
            return

        command = [
            llama_bin,
            "-m", model_path,
            "--host", "127.0.0.1",
            "--port", "8081",
            "-ngl", "99" # Full GPU offload
        ]

        # Use CREATE_NO_WINDOW so the terminal doesn't pop up even when it's an exe
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        try:
            self.server_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                startupinfo=startupinfo,
                text=True,
                bufsize=1
            )
            atexit.register(self.kill_server) # Safety net
            
            # Read output to detect readiness
            for line in self.server_process.stdout:
                if "HTTP server error" in line:
                    self.append_text(f"\nSystem Error: Failed to start server. Port 8081 might be in use.\n")
                    return
                if "server is listening on" in line:
                    self.is_server_ready = True
                    self.append_text("System: 🟢 Server is ready! Loading RAG engine...\n", "System")
                    self.init_agent()
                    self.append_text("System: 🟢 Knowledge Base linked. You can now chat and upload files.\n\n", "System")
                    break
        except Exception as e:
            self.append_text(f"\nSystem Error launching server: {e}\n")

    def kill_server(self):
        if self.server_process and self.server_process.poll() is None:
            self.server_process.kill()

    def append_text(self, text, sender=None):
        self.chat_history.configure(state="normal")
        if sender:
            self.chat_history.insert("end", f"{sender}: ", "sender")
        self.chat_history.insert("end", text)
        self.chat_history.see("end")
        self.chat_history.configure(state="disabled")

    def send_message(self):
        if not self.is_server_ready or self.agent is None:
            return
            
        user_input = self.entry.get().strip()
        if not user_input:
            return
            
        self.entry.delete(0, 'end')
        self.append_text(user_input + "\n", "You")
        
        self.send_btn.configure(state="disabled")
        self.append_text("", "Bonsai") # Start the AI message line
        
        # Thread for streaming response
        threading.Thread(target=self.generate_response, args=(user_input,), daemon=True).start()

    def generate_response(self, user_input):
        try:
            for chunk in self.agent.run(user_input, user_id=self.user_id, stream=True):
                if chunk.content:
                    text_chunk = chunk.content
                    # Safely push to UI thread
                    self.after(0, self.append_text, text_chunk)
            
            self.after(0, self.append_text, "\n\n")
            
        except Exception as e:
            self.after(0, self.append_text, f"[Error communicating with server: {e}]\n\n")
            
        finally:
            self.after(0, lambda: self.send_btn.configure(state="normal"))

    def upload_document(self):
        if not self.is_server_ready:
            self.append_text("System: Please wait for the knowledge base to load...\n", "System")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select a Document",
            filetypes=(("PDF Files", "*.pdf"), ("CSV Files", "*.csv"), ("Text Files", "*.txt;*.md;*.py;*.json"), ("All Files", "*.*"))
        )
        if file_path:
            self.append_text(f"System: Uploading {os.path.basename(file_path)} into Knowledge Base...\n", "System")
            threading.Thread(target=self._process_upload, args=(file_path,), daemon=True).start()

    def _process_upload(self, file_path):
        success = ingest_files([file_path])
        if success:
            self.after(0, self.append_text, f"System: ✓ Successfully ingested {os.path.basename(file_path)}\n", "System")
        else:
            self.after(0, self.append_text, f"System: ⚠ Failed to ingest {os.path.basename(file_path)}\n", "System")

    def clear_knowledge(self):
        if not self.is_server_ready:
            return
        self.append_text("System: Clearing knowledge base...\n", "System")
        threading.Thread(target=self._process_clear, daemon=True).start()

    def _process_clear(self):
        success = clear_knowledge_base()
        if success:
            self.after(0, self.append_text, "System: ✓ Knowledge base cleared successfully.\n", "System")
        else:
            self.after(0, self.append_text, "System: ⚠ Error clearing knowledge base.\n", "System")

    def destroy(self):
        # Cleanup when window shuts
        self.kill_server()
        super().destroy()

if __name__ == "__main__":
    app = BonsaiChatApp()
    app.mainloop()
