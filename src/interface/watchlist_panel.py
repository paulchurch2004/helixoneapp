import customtkinter as ctk  # Ajout de l'import manquant
import json
import threading
import time
import yfinance as yf
from datetime import datetime
from plyer import notification

class WatchlistPanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.tickers = []
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(header, text="📋 Watchlist", font=("Arial", 18, "bold")).pack(side="left")
        
        add_btn = ctk.CTkButton(
            header, text="➕ Ajouter", 
            width=100,
            command=self.add_ticker_dialog
        )
        add_btn.pack(side="right")
        
        # Liste scrollable
        self.list_frame = ctk.CTkScrollableFrame(self)
        self.list_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    def add_ticker(self, ticker):
        # Frame pour chaque ticker
        ticker_frame = ctk.CTkFrame(self.list_frame)
        ticker_frame.pack(fill="x", pady=5)
        
        # Récupérer les infos
        stock = yf.Ticker(ticker)
        info = stock.history(period="1d")
        current_price = info["Close"].iloc[-1]
        change = ((info["Close"].iloc[-1] - info["Open"].iloc[0]) / info["Open"].iloc[0]) * 100
        
        # Affichage
        ctk.CTkLabel(ticker_frame, text=ticker, font=("Arial", 14, "bold")).pack(side="left", padx=10)
        ctk.CTkLabel(ticker_frame, text=f"{current_price:.2f}€").pack(side="left", padx=10)
        
        color = "#00FF00" if change > 0 else "#FF0000"
        ctk.CTkLabel(ticker_frame, text=f"{change:+.2f}%", text_color=color).pack(side="left")
        
        # Boutons d'action
        ctk.CTkButton(ticker_frame, text="Analyser", width=80, 
                     command=lambda: self.analyze_ticker(ticker)).pack(side="right", padx=5)
        ctk.CTkButton(ticker_frame, text="🔔", width=30,
                     command=lambda: self.set_alert(ticker)).pack(side="right")