import customtkinter as ctk
import threading
import requests

class SearchBar(ctk.CTkFrame):
    def __init__(self, master, on_select):
        super().__init__(master)
        self.on_select = on_select
        self.entry = ctk.CTkEntry(self, placeholder_text="Recherche Ticker")
        self.entry.pack(fill="x", padx=10, pady=5)
        self.entry.bind("<KeyRelease>", self._on_key)
        
        self.suggestions = ctk.CTkTextbox(self, height=100)
        self.suggestions.pack(fill="x", padx=10)
        self.suggestions.configure(state="disabled")

    def _on_key(self, event):
        threading.Thread(target=self.fetch_suggestions).start()

    def fetch_suggestions(self):
        query = self.entry.get()
        try:
            res = requests.get(f"http://127.0.0.1:8000/autocomplete?query={query}")
            if res.status_code == 200:
                tickers = res.json()
                self.update_suggestions(tickers)
        except Exception as e:
            pass

    def update_suggestions(self, tickers):
        self.suggestions.configure(state="normal")
        self.suggestions.delete("1.0", "end")
        for t in tickers:
            self.suggestions.insert("end", f"{t}\n")
        self.suggestions.configure(state="disabled")
