"""
Collecteur de données principal
"""

import logging
from typing import Dict, Any
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    yf = None

from ..core.config import EngineConfig
from .scrapers import SimpleScraper

logger = logging.getLogger(__name__)

class DataCollector:
    """Collecte toutes les données nécessaires à l'analyse"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.scraper = SimpleScraper(config) if config.use_scraping else None
    
    def collect_all_data(self, ticker: str) -> Dict[str, Any]:
        """Collecte toutes les données pour un ticker"""
        data = {}
        
        # 1. Yahoo Finance (données principales)
        if self.config.use_yahoo and yf is not None:
            data['yahoo'] = self._collect_yahoo_data(ticker)
        else:
            data['yahoo'] = {'error': 'yfinance disabled or not installed'}
        
        # 2. Données scrapées (complément)
        if self.config.use_scraping and self.scraper:
            data['scraped'] = self._collect_scraped_data(ticker)
        else:
            data['scraped'] = {'error': 'scraping disabled'}
        
        # 3. Métadonnées
        data['_meta'] = {
            'collected_at': datetime.utcnow().isoformat(),
            'ticker': ticker,
            'sources_used': list(data.keys())
        }
        
        return data
    
    def _collect_yahoo_data(self, ticker: str) -> Dict[str, Any]:
        """Collecte les données Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            
            # Informations générales
            info = stock.info
            
            # Historique des prix (1 an)
            history = stock.history(period="1y", auto_adjust=False)
            
            # Données financières (optionnel)
            financials = None
            balance_sheet = None
            cashflow = None
            
            try:
                financials = stock.financials
                balance_sheet = stock.balance_sheet
                cashflow = stock.cashflow
            except:
                logger.debug(f"Données financières détaillées non disponibles pour {ticker}")
            
            return {
                'info': info,
                'history': history,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cashflow': cashflow,
                'success': True
            }
            
        except Exception as e:
            logger.warning(f"Erreur collecte Yahoo {ticker}: {e}")
            return {'error': str(e), 'success': False}
    
    def _collect_scraped_data(self, ticker: str) -> Dict[str, Any]:
        """Collecte les données scrapées"""
        try:
            scraped_data = self.scraper.scrape_all(ticker)
            scraped_data['success'] = True
            return scraped_data
        except Exception as e:
            logger.warning(f"Erreur scraping {ticker}: {e}")
            return {'error': str(e), 'success': False}