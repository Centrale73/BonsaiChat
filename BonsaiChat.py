import customtkinter as ctk
import subprocess
import threading
import sys
import os
import atexit
import time
from openai import OpenAI

# Configure UI theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

class BonsaiChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Bonsai Chat (1-bit LLM)")
        self.geometry("800x600")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Main Frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
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
        self.messages = [
            {"role": "system", "content": "You are a helpful and intelligent assistant."}
        ]
        self.is_server_ready = False
        
        # OpenAI Client (connected to local server once it's up)
        self.client = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="not-needed")

        # Launch server
        threading.Thread(target=self.start_server, daemon=True).start()

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
                    self.append_text("System: 🟢 Server is ready! You can now chat.\n\n", "System")
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
        if not self.is_server_ready:
            return
            
        user_input = self.entry.get().strip()
        if not user_input:
            return
            
        self.entry.delete(0, 'end')
        self.append_text(user_input + "\n", "You")
        self.messages.append({"role": "user", "content": user_input})
        
        self.send_btn.configure(state="disabled")
        self.append_text("", "Bonsai") # Start the AI message line
        
        # Thread for streaming response
        threading.Thread(target=self.generate_response, daemon=True).start()

    def generate_response(self):
        try:
            response = self.client.chat.completions.create(
                model="Bonsai-8B",
                messages=self.messages,
                stream=True,
                temperature=0.7
            )
            
            full_reply = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    text_chunk = chunk.choices[0].delta.content
                    full_reply += text_chunk
                    # Safely push to UI thread
                    self.after(0, self.append_text, text_chunk)
            
            self.messages.append({"role": "assistant", "content": full_reply})
            self.after(0, self.append_text, "\n\n")
            
        except Exception as e:
            self.after(0, self.append_text, f"[Error communicating with server: {e}]\n\n")
            
        finally:
            self.after(0, lambda: self.send_btn.configure(state="normal"))

    def destroy(self):
        # Cleanup when window shuts
        self.kill_server()
        super().destroy()

if __name__ == "__main__":
    app = BonsaiChatApp()
    app.mainloop()
