"""
Analyseur macro-économique avancé
Intègre inflation, calendrier économique, événements géopolitiques
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from bs4 import BeautifulSoup

from ..core.config import EngineConfig

logger = logging.getLogger(__name__)

class MacroEconomicAnalyzer:
    """Analyse macro-économique complète (0-100 points)"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.fed_data_sources = {
            'inflation': 'https://fred.stlouisfed.org/series/CPIAUCSL',
            'interest_rates': 'https://fred.stlouisfed.org/series/FEDFUNDS',
            'unemployment': 'https://fred.stlouisfed.org/series/UNRATE',
            'gdp_growth': 'https://fred.stlouisfed.org/series/GDPC1',
            'consumer_confidence': 'https://fred.stlouisfed.org/series/CSCICP03USM665S'
        }
        
        # Sources calendrier économique
        self.economic_calendar_sources = {
            'investing': 'https://www.investing.com/economic-calendar/',
            'forexfactory': 'https://www.forexfactory.com/calendar.php',
            'tradingeconomics': 'https://tradingeconomics.com/calendar'
        }
    
    def analyze(self, data: Dict, ticker: str) -> float:
        """
        Analyse macro-économique avec 6 dimensions :
        1. Conditions monétaires (20 points)
        2. Inflation et pouvoir d'achat (20 points)  
        3. Croissance économique (20 points)
        4. Calendrier économique à venir (15 points)
        5. Sentiment géopolitique (15 points)
        6. Corrélations sectorielles (10 points)
        """
        try:
            # Récupérer le secteur de l'entreprise pour analyse sectorielle
            sector = data.get('yahoo', {}).get('info', {}).get('sector', 'Unknown')
            
            score = 0.0
            
            # 1. CONDITIONS MONÉTAIRES
            monetary_score = self._analyze_monetary_conditions()
            
            # 2. INFLATION ET POUVOIR D'ACHAT
            inflation_score = self._analyze_inflation_impact(sector)
            
            # 3. CROISSANCE ÉCONOMIQUE
            growth_score = self._analyze_economic_growth()
            
            # 4. CALENDRIER ÉCONOMIQUE
            calendar_score = self._analyze_economic_calendar(sector, ticker)
            
            # 5. SENTIMENT GÉOPOLITIQUE
            geopolitical_score = self._analyze_geopolitical_sentiment()
            
            # 6. CORRÉLATIONS SECTORIELLES
            sector_correlation_score = self._analyze_sector_correlations(sector)
            
            total_score = (monetary_score + inflation_score + growth_score + 
                          calendar_score + geopolitical_score + sector_correlation_score)
            
            final_score = min(100.0, max(0.0, total_score))
            
            logger.debug(f"Scores macro {ticker}: Monetary={monetary_score:.1f}, "
                        f"Inflation={inflation_score:.1f}, Growth={growth_score:.1f}, "
                        f"Calendar={calendar_score:.1f}, Geo={geopolitical_score:.1f}, "
                        f"Sector={sector_correlation_score:.1f}, Total={final_score:.1f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Erreur analyse macro {ticker}: {e}")
            return 50.0
    
    def _analyze_monetary_conditions(self) -> float:
        """Analyse des conditions monétaires (taux Fed, liquidité)"""
        score = 10.0  # Score de base
        
        try:
            # Récupérer les taux actuels de la Fed
            current_fed_rate = self._get_current_fed_rate()
            
            # Analyser la tendance des taux
            if current_fed_rate is not None:
                if current_fed_rate < 2.0:
                    score += 10.0  # Taux très bas = stimulation économique
                elif current_fed_rate < 4.0:
                    score += 7.0   # Taux modérés = conditions favorables
                elif current_fed_rate < 6.0:
                    score += 4.0   # Taux élevés = resserrement
                else:
                    score += 1.0   # Taux très élevés = conditions restrictives
            
            # Analyser la pente de la courbe des taux
            yield_curve_slope = self._get_yield_curve_slope()
            if yield_curve_slope is not None:
                if yield_curve_slope > 1.0:
                    score += 0.0   # Courbe normale = neutre
                elif yield_curve_slope > -0.5:
                    score -= 3.0   # Courbe aplatie = signal d'alerte
                else:
                    score -= 7.0   # Courbe inversée = récession potentielle
            
        except Exception as e:
            logger.debug(f"Erreur conditions monétaires: {e}")
        
        return min(20.0, max(0.0, score))
    
    def _analyze_inflation_impact(self, sector: str) -> float:
        """Analyse de l'impact de l'inflation par secteur"""
        score = 10.0  # Score de base
        
        try:
            # Récupérer l'inflation actuelle
            current_inflation = self._get_current_inflation()
            
            if current_inflation is not None:
                # Impact sectoriel de l'inflation
                inflation_sensitive_sectors = {
                    'Consumer Staples': 0.8,      # Moins sensible
                    'Utilities': 0.7,             # Moins sensible
                    'Healthcare': 0.6,            # Moins sensible
                    'Technology': -0.3,           # Négativement impacté (taux)
                    'Real Estate': -0.8,          # Très négativement impacté
                    'Consumer Discretionary': -0.5, # Négativement impacté
                    'Financials': 0.5,            # Positivement impacté (marges)
                    'Energy': 0.9,                # Très positivement impacté
                    'Materials': 0.7,             # Positivement impacté
                    'Industrials': 0.2            # Légèrement positif
                }
                
                sector_multiplier = inflation_sensitive_sectors.get(sector, 0.0)
                
                # Score basé sur l'inflation et la sensibilité sectorielle
                if current_inflation < 2.0:
                    base_inflation_score = 8.0   # Inflation faible = bon
                elif current_inflation < 4.0:
                    base_inflation_score = 5.0   # Inflation modérée
                elif current_inflation < 6.0:
                    base_inflation_score = 2.0   # Inflation élevée
                else:
                    base_inflation_score = 0.0   # Inflation très élevée
                
                # Ajuster selon le secteur
                sector_adjustment = sector_multiplier * current_inflation
                score += base_inflation_score + sector_adjustment
            
        except Exception as e:
            logger.debug(f"Erreur analyse inflation: {e}")
        
        return min(20.0, max(0.0, score))
    
    def _analyze_economic_growth(self) -> float:
        """Analyse de la croissance économique (PIB, emploi)"""
        score = 10.0  # Score de base
        
        try:
            # Croissance du PIB
            gdp_growth = self._get_gdp_growth()
            if gdp_growth is not None:
                if gdp_growth > 3.0:
                    score += 8.0   # Forte croissance
                elif gdp_growth > 2.0:
                    score += 5.0   # Croissance modérée
                elif gdp_growth > 0.0:
                    score += 2.0   # Croissance faible
                else:
                    score -= 5.0   # Récession
            
            # Taux de chômage
            unemployment_rate = self._get_unemployment_rate()
            if unemployment_rate is not None:
                if unemployment_rate < 4.0:
                    score += 5.0   # Plein emploi
                elif unemployment_rate < 6.0:
                    score += 3.0   # Chômage modéré
                elif unemployment_rate < 8.0:
                    score += 1.0   # Chômage élevé
                else:
                    score -= 2.0   # Chômage très élevé
            
            # Confiance des consommateurs
            consumer_confidence = self._get_consumer_confidence()
            if consumer_confidence is not None:
                if consumer_confidence > 100:
                    score += 5.0   # Confiance élevée
                elif consumer_confidence > 90:
                    score += 3.0   # Confiance modérée
                else:
                    score += 1.0   # Confiance faible
            
        except Exception as e:
            logger.debug(f"Erreur analyse croissance: {e}")
        
        return min(20.0, max(0.0, score))
    
    def _analyze_economic_calendar(self, sector: str, ticker: str) -> float:
        """Analyse du calendrier économique à venir"""
        score = 7.0  # Score de base
        
        try:
            # Récupérer les événements économiques des 7 prochains jours
            upcoming_events = self._get_upcoming_economic_events()
            
            if upcoming_events:
                # Événements à fort impact
                high_impact_events = [
                    'FOMC Meeting', 'Interest Rate Decision', 'NFP', 
                    'CPI', 'GDP', 'Unemployment Rate'
                ]
                
                # Événements spécifiques au secteur
                sector_specific_events = {
                    'Energy': ['Oil Inventory', 'OPEC Meeting'],
                    'Technology': ['Tech Earnings Season'],
                    'Healthcare': ['FDA Approval'],
                    'Financials': ['Banking Stress Test'],
                    'Consumer Discretionary': ['Retail Sales']
                }
                
                high_impact_count = 0
                sector_specific_count = 0
                
                for event in upcoming_events:
                    event_name = event.get('name', '').upper()
                    
                    # Compter les événements à fort impact
                    if any(high_event.upper() in event_name for high_event in high_impact_events):
                        high_impact_count += 1
                    
                    # Compter les événements spécifiques au secteur
                    sector_events = sector_specific_events.get(sector, [])
                    if any(sector_event.upper() in event_name for sector_event in sector_events):
                        sector_specific_count += 1
                
                # Ajuster le score selon les événements à venir
                if high_impact_count == 0:
                    score += 8.0   # Pas d'événements majeurs = stabilité
                elif high_impact_count <= 2:
                    score += 5.0   # Quelques événements = attention
                else:
                    score += 2.0   # Beaucoup d'événements = volatilité
                
                # Bonus pour événements sectoriels favorables
                score += min(3.0, sector_specific_count * 1.0)
            
        except Exception as e:
            logger.debug(f"Erreur calendrier économique: {e}")
        
        return min(15.0, max(0.0, score))
    
    def _analyze_geopolitical_sentiment(self) -> float:
        """Analyse du sentiment géopolitique"""
        score = 8.0  # Score de base
        
        try:
            # Analyser les nouvelles géopolitiques récentes
            geopolitical_news = self._get_geopolitical_news()
            
            if geopolitical_news:
                # Mots-clés négatifs
                negative_keywords = [
                    'war', 'conflict', 'sanctions', 'tension', 'crisis',
                    'embargo', 'recession', 'default', 'bankruptcy'
                ]
                
                # Mots-clés positifs
                positive_keywords = [
                    'agreement', 'deal', 'recovery', 'growth', 'cooperation',
                    'stimulus', 'support', 'alliance', 'stability'
                ]
                
                negative_count = 0
                positive_count = 0
                
                for news in geopolitical_news[:20]:  # Analyser les 20 dernières news
                    content = (news.get('title', '') + ' ' + news.get('summary', '')).lower()
                    
                    negative_count += sum(1 for keyword in negative_keywords if keyword in content)
                    positive_count += sum(1 for keyword in positive_keywords if keyword in content)
                
                # Calculer le sentiment net
                net_sentiment = positive_count - negative_count
                
                if net_sentiment > 5:
                    score += 7.0   # Sentiment très positif
                elif net_sentiment > 0:
                    score += 4.0   # Sentiment positif
                elif net_sentiment > -5:
                    score += 1.0   # Sentiment neutre
                else:
                    score -= 3.0   # Sentiment négatif
            
        except Exception as e:
            logger.debug(f"Erreur sentiment géopolitique: {e}")
        
        return min(15.0, max(0.0, score))
    
    def _analyze_sector_correlations(self, sector: str) -> float:
        """Analyse des corrélations sectorielles avec conditions macro"""
        score = 5.0  # Score de base
        
        try:
            # Corrélations secteur/macro historiques
            sector_macro_correlations = {
                'Technology': {'interest_rates': -0.7, 'inflation': -0.5, 'gdp_growth': 0.6},
                'Financials': {'interest_rates': 0.8, 'inflation': 0.3, 'gdp_growth': 0.7},
                'Real Estate': {'interest_rates': -0.9, 'inflation': -0.6, 'gdp_growth': 0.4},
                'Energy': {'inflation': 0.8, 'gdp_growth': 0.5, 'oil_price': 0.9},
                'Consumer Staples': {'inflation': -0.3, 'unemployment': -0.4, 'gdp_growth': 0.2},
                'Healthcare': {'interest_rates': -0.2, 'inflation': -0.1, 'gdp_growth': 0.3},
                'Materials': {'inflation': 0.6, 'gdp_growth': 0.8, 'dollar_strength': -0.5},
                'Industrials': {'gdp_growth': 0.9, 'unemployment': -0.6, 'trade_policy': 0.4},
                'Consumer Discretionary': {'unemployment': -0.7, 'consumer_confidence': 0.8},
                'Utilities': {'interest_rates': -0.6, 'inflation': -0.3}
            }
            
            sector_correlations = sector_macro_correlations.get(sector, {})
            
            if sector_correlations:
                # Récupérer les conditions macro actuelles
                current_conditions = self._get_current_macro_conditions()
                
                correlation_score = 0.0
                for macro_var, correlation in sector_correlations.items():
                    condition_trend = current_conditions.get(macro_var, 0)  # -1, 0, 1
                    correlation_score += correlation * condition_trend
                
                # Normaliser le score
                score += max(-5.0, min(5.0, correlation_score * 2))
            
        except Exception as e:
            logger.debug(f"Erreur corrélations sectorielles: {e}")
        
        return min(10.0, max(0.0, score))
    
    # Méthodes de collecte de données
    def _get_current_fed_rate(self) -> Optional[float]:
        """Récupère le taux directeur actuel de la Fed"""
        try:
            url = "https://fred.stlouisfed.org/series/FEDFUNDS"
            # API FRED nécessite une clé, simulation ici
            # En production, utiliser l'API officielle FRED
            return 5.25  # Taux exemple
        except:
            return None
    
    def _get_yield_curve_slope(self) -> Optional[float]:
        """Récupère la pente de la courbe des taux (10Y - 2Y)"""
        try:
            # Simulation - en production utiliser FRED API
            return 0.3  # Pente exemple
        except:
            return None
    
    def _get_current_inflation(self) -> Optional[float]:
        """Récupère l'inflation actuelle (CPI YoY)"""
        try:
            # Scraping ou API
            url = "https://www.bls.gov/news.release/cpi.htm"
            # Simulation
            return 3.2  # Inflation exemple
        except:
            return None
    
    def _get_gdp_growth(self) -> Optional[float]:
        """Récupère la croissance du PIB"""
        try:
            return 2.1  # Croissance PIB exemple
        except:
            return None
    
    def _get_unemployment_rate(self) -> Optional[float]:
        """Récupère le taux de chômage"""
        try:
            return 3.8  # Chômage exemple
        except:
            return None
    
    def _get_consumer_confidence(self) -> Optional[float]:
        """Récupère l'indice de confiance des consommateurs"""
        try:
            return 102.5  # Confiance exemple
        except:
            return None
    
    def _get_upcoming_economic_events(self) -> List[Dict]:
        """Récupère les événements économiques à venir"""
        try:
            # Scraping du calendrier économique
            url = "https://www.investing.com/economic-calendar/"
            # Simulation
            return [
                {'name': 'FOMC Meeting', 'date': '2024-01-15', 'impact': 'high'},
                {'name': 'CPI Release', 'date': '2024-01-12', 'impact': 'high'}
            ]
        except:
            return []
    
    def _get_geopolitical_news(self) -> List[Dict]:
        """Récupère les nouvelles géopolitiques"""
        try:
            # Scraping de sources d'actualités
            # Reuters, Bloomberg, etc.
            return [
                {'title': 'Economic cooperation agreement signed', 'summary': 'Positive development'},
                {'title': 'Trade tensions ease', 'summary': 'Markets optimistic'}
            ]
        except:
            return []
    
    def _get_current_macro_conditions(self) -> Dict[str, int]:
        """Récupère les tendances macro actuelles (-1: baisse, 0: stable, 1: hausse)"""
        try:
            return {
                'interest_rates': 0,    # Stable
                'inflation': -1,        # En baisse
                'gdp_growth': 1,        # En hausse
                'unemployment': -1,     # En baisse
                'consumer_confidence': 1, # En hausse
                'oil_price': 0,         # Stable
                'dollar_strength': 1    # En hausse
            }
        except:
            return {}