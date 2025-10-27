# chart_component.py - Nouveau composant pour afficher des graphiques

import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class ChartPanel(ctk.CTkFrame):
    """
    Panneau de graphiques interactifs pour HelixOne
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(fg_color="#1c1f26")
        
        # Toolbar avec boutons pour changer le type de graphique
        self.toolbar = ctk.CTkFrame(self)
        self.toolbar.pack(fill="x", padx=10, pady=5)
        
        self.chart_types = ["Prix", "RSI", "MACD", "Volume", "Bollinger"]
        self.current_chart = "Prix"
        
        for chart_type in self.chart_types:
            btn = ctk.CTkButton(
                self.toolbar, 
                text=chart_type,
                width=80,
                height=28,
                command=lambda t=chart_type: self.switch_chart(t),
                fg_color="#2a2d35" if chart_type != self.current_chart else "#00BFFF"
            )
            btn.pack(side="left", padx=2)
        
        # Zone de graphique
        self.chart_frame = ctk.CTkFrame(self, fg_color="#101418")
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.figure = None
        self.canvas = None
        self.current_data = None
        self.current_ticker = None
    
    def switch_chart(self, chart_type):
        """Change le type de graphique affiché"""
        self.current_chart = chart_type
        if self.current_data is not None:
            self.update_chart()
    
    def load_ticker_data(self, ticker: str, period: str = "6mo"):
        """Charge les données du ticker"""
        try:
            stock = yf.Ticker(ticker)
            self.current_data = stock.history(period=period)
            self.current_ticker = ticker
            
            # Calculer les indicateurs
            self._calculate_indicators()
            
            # Afficher le graphique
            self.update_chart()
            
        except Exception as e:
            print(f"Erreur chargement {ticker}: {e}")
    
    def _calculate_indicators(self):
        """Calcule tous les indicateurs techniques"""
        if self.current_data is None:
            return
        
        df = self.current_data
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Moving Averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
    
    def update_chart(self):
        """Met à jour le graphique selon le type sélectionné"""
        if self.current_data is None:
            return
        
        # Nettoyer le canvas précédent
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        # Créer une nouvelle figure
        self.figure = Figure(figsize=(10, 5), dpi=80, facecolor='#101418')
        
        if self.current_chart == "Prix":
            self._plot_price_chart()
        elif self.current_chart == "RSI":
            self._plot_rsi_chart()
        elif self.current_chart == "MACD":
            self._plot_macd_chart()
        elif self.current_chart == "Volume":
            self._plot_volume_chart()
        elif self.current_chart == "Bollinger":
            self._plot_bollinger_chart()
        
        # Afficher le canvas
        self.canvas = FigureCanvasTkAgg(self.figure, self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _plot_price_chart(self):
        """Graphique des prix avec moyennes mobiles"""
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#1c1f26')
        
        df = self.current_data
        
        # Prix de clôture
        ax.plot(df.index, df['Close'], label='Prix', color='#00BFFF', linewidth=2)
        
        # Moyennes mobiles
        if 'MA20' in df.columns:
            ax.plot(df.index, df['MA20'], label='MA20', color='#FFA500', alpha=0.7)
        if 'MA50' in df.columns:
            ax.plot(df.index, df['MA50'], label='MA50', color='#FF69B4', alpha=0.7)
        if 'MA200' in df.columns:
            ax.plot(df.index, df['MA200'], label='MA200', color='#32CD32', alpha=0.7)
        
        ax.set_title(f'{self.current_ticker} - Prix et Moyennes Mobiles', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Prix ($)', color='white')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        
    def _plot_rsi_chart(self):
        """Graphique RSI avec zones de surachat/survente"""
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#1c1f26')
        
        df = self.current_data
        
        # RSI
        ax.plot(df.index, df['RSI'], label='RSI', color='#00BFFF', linewidth=2)
        
        # Zones
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Surachat (70)')
        ax.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Survente (30)')
        ax.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
        
        ax.set_title(f'{self.current_ticker} - RSI (14)', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('RSI', color='white')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
    
    def _plot_macd_chart(self):
        """Graphique MACD avec signal et histogramme"""
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#1c1f26')
        
        df = self.current_data
        
        # MACD et Signal
        ax.plot(df.index, df['MACD'], label='MACD', color='#00BFFF', linewidth=2)
        ax.plot(df.index, df['Signal'], label='Signal', color='#FFA500', linewidth=2)
        
        # Histogramme
        colors = ['g' if val >= 0 else 'r' for val in df['MACD_Histogram']]
        ax.bar(df.index, df['MACD_Histogram'], label='Histogram', color=colors, alpha=0.3)
        
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.3)
        ax.set_title(f'{self.current_ticker} - MACD', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('MACD', color='white')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
    
    def _plot_volume_chart(self):
        """Graphique des volumes avec moyenne"""
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        
        ax1.set_facecolor('#1c1f26')
        ax2.set_facecolor('#1c1f26')
        
        df = self.current_data
        
        # Prix sur le graphique du haut
        ax1.plot(df.index, df['Close'], color='#00BFFF', linewidth=1)
        ax1.set_ylabel('Prix ($)', color='white')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.2)
        
        # Volume sur le graphique du bas
        colors = ['g' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'r' 
                 for i in range(len(df))]
        ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7)
        
        # Moyenne des volumes
        vol_mean = df['Volume'].rolling(window=20).mean()
        ax2.plot(df.index, vol_mean, color='#FFA500', linewidth=2, label='MA Volume (20)')
        
        ax2.set_title(f'{self.current_ticker} - Volume', color='white')
        ax2.set_xlabel('Date', color='white')
        ax2.set_ylabel('Volume', color='white')
        ax2.legend(loc='upper left')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.2)
    
    def _plot_bollinger_chart(self):
        """Graphique Bollinger Bands"""
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#1c1f26')
        
        df = self.current_data
        
        # Prix et Bollinger Bands
        ax.plot(df.index, df['Close'], label='Prix', color='#00BFFF', linewidth=2)
        ax.plot(df.index, df['BB_Upper'], label='BB Supérieure', color='r', alpha=0.5)
        ax.plot(df.index, df['BB_Middle'], label='BB Moyenne', color='orange', alpha=0.5)
        ax.plot(df.index, df['BB_Lower'], label='BB Inférieure', color='g', alpha=0.5)
        
        # Remplissage entre les bandes
        ax.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='gray')
        
        ax.set_title(f'{self.current_ticker} - Bollinger Bands', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Prix ($)', color='white')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
    
    def export_chart(self, filename=None):
        """Exporte le graphique actuel"""
        if self.figure:
            if filename is None:
                filename = f"{self.current_ticker}_{self.current_chart}_{datetime.now().strftime('%Y%m%d')}.png"
            self.figure.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#101418')
            return filename
        return None


# Fonction pour intégrer dans main_app.py
def add_chart_to_interface(main_frame, ticker):
    """Ajoute le panneau de graphiques à l'interface"""
    # Créer le panneau
    chart_panel = ChartPanel(main_frame)
    chart_panel.pack(fill="both", expand=True, padx=20, pady=10)
    
    # Charger les données
    chart_panel.load_ticker_data(ticker)
    
    return chart_panel