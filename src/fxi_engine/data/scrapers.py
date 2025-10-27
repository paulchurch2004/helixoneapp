"""
Scraper simple et robuste
"""

import time
import random
import logging
from typing import Dict, Optional
import requests
from bs4 import BeautifulSoup

from ..core.config import EngineConfig

logger = logging.getLogger(__name__)

class SimpleScraper:
    """Scraper simple pour données complémentaires"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def scrape_all(self, ticker: str) -> Dict:
        """Scrape toutes les sources disponibles"""
        results = {}
        
        # MarketWatch
        try:
            marketwatch_data = self._scrape_marketwatch(ticker)
            results.update(marketwatch_data)
        except Exception as e:
            results['marketwatch_error'] = str(e)
        
        # FINVIZ
        try:
            finviz_data = self._scrape_finviz(ticker)
            results.update(finviz_data)
        except Exception as e:
            results['finviz_error'] = str(e)
        
        return results
    
    def _scrape_marketwatch(self, ticker: str) -> Dict:
        """Scrape MarketWatch pour données de sentiment"""
        url = f"https://www.marketwatch.com/investing/stock/{ticker}"
        
        try:
            # Respecter le rate limiting
            time.sleep(random.uniform(1.0, 2.0))
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            data = {}
            
            # Compter les actualités récentes
            news_headlines = soup.find_all('h3', class_='article__headline')
            data['marketwatch_recent_news_count'] = len(news_headlines)
            data['marketwatch_recent_headlines'] = [
                headline.get_text(strip=True) for headline in news_headlines[:5]
            ]
            
            # Changement de prix (si disponible)
            try:
                change_element = soup.find('span', class_='change--percent--q')
                if change_element:
                    change_text = change_element.get_text(strip=True)
                    data['marketwatch_change_percent'] = change_text
            except:
                pass
            
            return data
            
        except Exception as e:
            logger.debug(f"Erreur scraping MarketWatch {ticker}: {e}")
            return {'marketwatch_error': str(e)}
    
    def _scrape_finviz(self, ticker: str) -> Dict:
        """Scrape FINVIZ pour métriques financières"""
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        
        try:
            # Respecter le rate limiting plus strict pour FINVIZ
            time.sleep(random.uniform(2.0, 3.0))
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            data = {}
            
            # Chercher le tableau de snapshot
            snapshot_table = soup.find('table', class_='snapshot-table2')
            if snapshot_table:
                cells = snapshot_table.find_all('td')
                
                # Parser les paires clé-valeur
                for i in range(0, len(cells) - 1, 2):
                    if i + 1 < len(cells):
                        label = cells[i].get_text(strip=True)
                        value = cells[i + 1].get_text(strip=True)
                        
                        # Capturer quelques métriques importantes
                        if label == 'P/E':
                            data['finviz_pe'] = self._parse_number(value)
                        elif label == 'PEG':
                            data['finviz_peg'] = self._parse_number(value)
                        elif label == 'Debt/Eq':
                            data['finviz_debt_equity'] = self._parse_number(value)
                        elif label == 'ROE':
                            data['finviz_roe'] = self._parse_percentage(value)
                        elif label == 'ROI':
                            data['finviz_roi'] = self._parse_percentage(value)
                        elif label == 'Recom':
                            data['finviz_recommendation'] = self._parse_number(value)
            
            return data
            
        except Exception as e:
            logger.debug(f"Erreur scraping FINVIZ {ticker}: {e}")
            return {'finviz_error': str(e)}
    
    def _parse_number(self, text: str) -> Optional[float]:
        """Parse un nombre depuis du texte"""
        if not text or text in ['-', 'N/A', '']:
            return None
        
        try:
            # Nettoyer le texte (garder seulement chiffres, point et signe moins)
            clean_text = ''.join(c for c in text if c.isdigit() or c in '.-')
            if clean_text and clean_text != '-':
                return float(clean_text)
        except:
            pass
        
        return None
    
    def _parse_percentage(self, text: str) -> Optional[float]:
        """Parse un pourcentage depuis du texte"""
        if not text:
            return None
        
        try:
            # Supprimer le symbole % et convertir
            clean_text = text.replace('%', '').strip()
            number = self._parse_number(clean_text)
            if number is not None:
                return number / 100.0  # Convertir en décimal
        except:
            pass
        
        return None