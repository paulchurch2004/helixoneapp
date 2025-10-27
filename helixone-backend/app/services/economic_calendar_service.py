"""
Service de Calendrier Économique
Récupère et analyse les événements économiques à venir qui peuvent impacter le marché

Sources:
- Finnhub Economic Calendar API
- Alpha Vantage Earnings Calendar
- FRED (Fed meetings, economic releases)
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import finnhub
import asyncio
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EconomicEvent:
    """Événement économique à venir"""
    event_id: str
    event_type: str  # 'fed_meeting', 'earnings', 'cpi', 'gdp', 'nfp', etc.
    title: str
    description: Optional[str]
    date: datetime
    country: str  # 'US', 'EU', 'CN', etc.

    # Impact
    impact_level: str  # 'low', 'medium', 'high', 'critical'
    affected_sectors: List[str]  # ['Technology', 'Financials', ...]

    # Données
    actual: Optional[float] = None  # Valeur réelle (après l'événement)
    forecast: Optional[float] = None  # Valeur prévue
    previous: Optional[float] = None  # Valeur précédente
    unit: Optional[str] = None  # '%', 'K', 'M', 'B', etc.

    # Métadonnées
    source: str = "finnhub"
    ticker: Optional[str] = None  # Pour earnings, le ticker concerné


@dataclass
class FedMeeting:
    """Réunion de la Fed (FOMC)"""
    date: datetime
    type: str  # 'regular', 'emergency'
    expected_rate_change: Optional[float]  # Changement de taux attendu (bps)
    minutes_release_date: Optional[datetime]
    press_conference: bool


@dataclass
class EarningsEvent:
    """Publication de résultats d'entreprise"""
    ticker: str
    company_name: str
    date: datetime
    fiscal_period: str  # 'Q1', 'Q2', 'Q3', 'Q4', 'FY'

    # Estimates
    eps_estimate: Optional[float]
    revenue_estimate: Optional[float]

    # Historique
    eps_last_quarter: Optional[float]
    beat_rate: Optional[float]  # % de fois où l'entreprise bat les attentes

    # Timing
    time: Optional[str]  # 'bmo' (before market open), 'amc' (after market close)


# ============================================================================
# SERVICE PRINCIPAL
# ============================================================================

class EconomicCalendarService:
    """
    Service de calendrier économique

    Fournit :
    - Événements économiques à venir
    - Prédictions d'impact sur les marchés
    - Alertes avant événements critiques
    """

    def __init__(self):
        self.finnhub_client = None

        # Initialiser Finnhub si API key disponible
        if settings.FINNHUB_API_KEY:
            try:
                self.finnhub_client = finnhub.Client(api_key=settings.FINNHUB_API_KEY)
                logger.info("EconomicCalendarService initialisé avec Finnhub")
            except Exception as e:
                logger.error(f"Erreur init Finnhub: {e}")

        # Mapping des événements macro importants
        self.critical_events = {
            'Interest Rate Decision': 'critical',
            'FOMC Meeting': 'critical',
            'Non-Farm Payrolls': 'high',
            'CPI': 'high',
            'Core CPI': 'high',
            'GDP': 'high',
            'Unemployment Rate': 'high',
            'Retail Sales': 'medium',
            'PPI': 'medium',
            'Consumer Confidence': 'medium',
            'Manufacturing PMI': 'medium',
            'Services PMI': 'medium',
            'Housing Starts': 'low',
            'Initial Jobless Claims': 'low'
        }

        # Secteurs affectés par type d'événement
        self.event_sector_impact = {
            'Interest Rate Decision': ['Financials', 'Real Estate', 'Utilities'],
            'CPI': ['Consumer Discretionary', 'Consumer Staples'],
            'Non-Farm Payrolls': ['All'],
            'GDP': ['All'],
            'Retail Sales': ['Consumer Discretionary', 'Consumer Staples'],
            'Housing Starts': ['Real Estate', 'Materials', 'Industrials']
        }

    # ========================================================================
    # MÉTHODES PRINCIPALES
    # ========================================================================

    async def get_upcoming_events(
        self,
        days: int = 30,
        min_impact: str = 'low'
    ) -> List[EconomicEvent]:
        """
        Récupère tous les événements économiques à venir

        Args:
            days: Nombre de jours à regarder dans le futur
            min_impact: Impact minimum ('low', 'medium', 'high', 'critical')

        Returns:
            Liste d'événements économiques triés par date
        """
        logger.info(f"📅 Récupération événements économiques ({days} jours)")

        all_events = []

        # Récupérer événements macro
        macro_events = await self._get_macro_events(days)
        all_events.extend(macro_events)

        # Récupérer Fed meetings
        fed_meetings = await self._get_fed_meetings(days)
        all_events.extend(fed_meetings)

        # Récupérer earnings (si demandé)
        # earnings_events = await self._get_earnings_events(days)
        # all_events.extend(earnings_events)

        # Filtrer par impact minimum
        impact_levels = ['low', 'medium', 'high', 'critical']
        min_index = impact_levels.index(min_impact)

        filtered_events = [
            event for event in all_events
            if impact_levels.index(event.impact_level) >= min_index
        ]

        # Trier par date
        filtered_events.sort(key=lambda x: x.date)

        logger.info(f"✅ {len(filtered_events)} événements trouvés")
        return filtered_events

    async def get_upcoming_earnings(
        self,
        tickers: Optional[List[str]] = None,
        days: int = 30
    ) -> List[EarningsEvent]:
        """
        Récupère les earnings à venir

        Args:
            tickers: Liste de tickers (ou None pour tous)
            days: Nombre de jours

        Returns:
            Liste d'événements earnings
        """
        if not self.finnhub_client:
            logger.warning("Finnhub non disponible pour earnings")
            return []

        logger.info(f"📊 Récupération earnings calendar ({days} jours)")

        try:
            # Calculer dates
            start_date = datetime.now()
            end_date = start_date + timedelta(days=days)

            # Finnhub earnings calendar
            earnings_data = await asyncio.get_event_loop().run_in_executor(
                None,
                self.finnhub_client.earnings_calendar,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                '',  # symbol (vide = tous)
                False  # international
            )

            earnings_events = []

            for item in earnings_data.get('earningsCalendar', []):
                # Filtrer par tickers si spécifié
                if tickers and item.get('symbol') not in tickers:
                    continue

                event = EarningsEvent(
                    ticker=item.get('symbol'),
                    company_name=item.get('name', item.get('symbol')),
                    date=datetime.fromisoformat(item.get('date')),
                    fiscal_period=item.get('quarter', 'Unknown'),
                    eps_estimate=item.get('epsEstimate'),
                    revenue_estimate=item.get('revenueEstimate'),
                    eps_last_quarter=None,  # TODO: Récupérer historique
                    beat_rate=None,  # TODO: Calculer
                    time=None  # Finnhub ne fournit pas le timing
                )

                earnings_events.append(event)

            logger.info(f"✅ {len(earnings_events)} earnings trouvés")
            return earnings_events

        except Exception as e:
            logger.error(f"❌ Erreur récupération earnings: {e}")
            return []

    async def get_events_for_ticker(
        self,
        ticker: str,
        days: int = 30
    ) -> Dict[str, List]:
        """
        Récupère tous les événements concernant un ticker spécifique

        Returns:
            Dict avec 'earnings', 'macro_events', 'sector_events'
        """
        logger.info(f"🔍 Événements pour {ticker}")

        # Earnings du ticker
        earnings = await self.get_upcoming_earnings([ticker], days)

        # Événements macro globaux
        macro_events = await self.get_upcoming_events(days, min_impact='medium')

        # TODO: Identifier secteur du ticker et filtrer événements pertinents

        return {
            'earnings': earnings,
            'macro_events': macro_events,
            'sector_events': []
        }

    # ========================================================================
    # MÉTHODES PRIVÉES - RÉCUPÉRATION DONNÉES
    # ========================================================================

    async def _get_macro_events(self, days: int) -> List[EconomicEvent]:
        """Récupère événements macro via Finnhub"""
        if not self.finnhub_client:
            return []

        try:
            # Finnhub Economic Calendar
            # https://finnhub.io/docs/api/economic-calendar
            calendar_data = await asyncio.get_event_loop().run_in_executor(
                None,
                self.finnhub_client.economic_calendar
            )

            events = []
            now = datetime.now()
            end_date = now + timedelta(days=days)

            for item in calendar_data.get('economicCalendar', []):
                try:
                    event_date_str = item.get('time')
                    if not event_date_str:
                        continue

                    # Parse date (format: '2024-10-24 08:30:00')
                    event_date = datetime.strptime(event_date_str, '%Y-%m-%d %H:%M:%S')

                    # Filtrer par date
                    if event_date < now or event_date > end_date:
                        continue

                    # Déterminer importance
                    event_name = item.get('event', '')
                    impact_level = self._determine_impact_level(event_name)

                    # Secteurs affectés
                    affected_sectors = self.event_sector_impact.get(
                        event_name,
                        ['All'] if impact_level in ['high', 'critical'] else []
                    )

                    event = EconomicEvent(
                        event_id=f"finnhub_{item.get('time')}_{event_name}",
                        event_type=self._classify_event_type(event_name),
                        title=event_name,
                        description=item.get('country', ''),
                        date=event_date,
                        country=item.get('country', 'US'),
                        impact_level=impact_level,
                        affected_sectors=affected_sectors,
                        actual=item.get('actual'),
                        forecast=item.get('estimate'),
                        previous=item.get('prev'),
                        unit=item.get('unit', ''),
                        source='finnhub'
                    )

                    events.append(event)

                except Exception as e:
                    logger.warning(f"Erreur parsing événement: {e}")
                    continue

            return events

        except Exception as e:
            logger.error(f"❌ Erreur _get_macro_events: {e}")
            return []

    async def _get_fed_meetings(self, days: int) -> List[EconomicEvent]:
        """
        Récupère les prochaines réunions de la Fed

        Note: Les dates des réunions FOMC sont publiées à l'avance
        https://www.federalreserve.gov/monetarypolicy/fomccalend.htm
        """
        # Dates FOMC 2024-2025 (source: Fed)
        fomc_dates_2024 = [
            '2024-10-30',  # 30-31 octobre
            '2024-12-11'   # 10-11 décembre
        ]

        fomc_dates_2025 = [
            '2025-01-29',  # 28-29 janvier
            '2025-03-19',  # 18-19 mars
            '2025-04-30',  # 29-30 avril
            '2025-06-18',  # 17-18 juin
            '2025-07-30',  # 29-30 juillet
            '2025-09-17',  # 16-17 septembre
            '2025-10-29',  # 28-29 octobre
            '2025-12-10'   # 9-10 décembre
        ]

        all_fomc_dates = fomc_dates_2024 + fomc_dates_2025

        events = []
        now = datetime.now()
        end_date = now + timedelta(days=days)

        for date_str in all_fomc_dates:
            meeting_date = datetime.strptime(date_str, '%Y-%m-%d')

            # Filtrer par période
            if meeting_date < now or meeting_date > end_date:
                continue

            event = EconomicEvent(
                event_id=f"fed_meeting_{date_str}",
                event_type='fed_meeting',
                title='FOMC Meeting - Interest Rate Decision',
                description='Federal Open Market Committee meeting',
                date=meeting_date,
                country='US',
                impact_level='critical',
                affected_sectors=['All'],
                source='fed'
            )

            events.append(event)

        return events

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _determine_impact_level(self, event_name: str) -> str:
        """Détermine le niveau d'impact d'un événement"""
        for name, impact in self.critical_events.items():
            if name.lower() in event_name.lower():
                return impact
        return 'low'

    def _classify_event_type(self, event_name: str) -> str:
        """Classifie le type d'événement"""
        name_lower = event_name.lower()

        if 'interest rate' in name_lower or 'fomc' in name_lower:
            return 'fed_meeting'
        elif 'cpi' in name_lower:
            return 'cpi'
        elif 'gdp' in name_lower:
            return 'gdp'
        elif 'payroll' in name_lower or 'nfp' in name_lower:
            return 'nfp'
        elif 'unemployment' in name_lower:
            return 'unemployment'
        elif 'retail sales' in name_lower:
            return 'retail_sales'
        elif 'ppi' in name_lower:
            return 'ppi'
        elif 'housing' in name_lower:
            return 'housing'
        elif 'pmi' in name_lower:
            return 'pmi'
        else:
            return 'other'

    def get_impact_summary(
        self,
        events: List[EconomicEvent],
        portfolio_sectors: List[str]
    ) -> Dict:
        """
        Résume l'impact potentiel des événements sur un portefeuille

        Args:
            events: Liste d'événements
            portfolio_sectors: Secteurs présents dans le portefeuille

        Returns:
            Dict avec résumé d'impact
        """
        impact_summary = {
            'total_events': len(events),
            'critical_events': 0,
            'high_impact_events': 0,
            'affected_sectors': set(),
            'high_risk_days': [],
            'events_by_type': {}
        }

        for event in events:
            # Compter par niveau
            if event.impact_level == 'critical':
                impact_summary['critical_events'] += 1
            elif event.impact_level == 'high':
                impact_summary['high_impact_events'] += 1

            # Secteurs affectés
            for sector in event.affected_sectors:
                if sector in portfolio_sectors or sector == 'All':
                    impact_summary['affected_sectors'].add(sector)

            # Grouper par type
            event_type = event.event_type
            if event_type not in impact_summary['events_by_type']:
                impact_summary['events_by_type'][event_type] = []
            impact_summary['events_by_type'][event_type].append(event)

        impact_summary['affected_sectors'] = list(impact_summary['affected_sectors'])

        return impact_summary


# ============================================================================
# SINGLETON
# ============================================================================

_calendar_service_instance = None

def get_economic_calendar_service() -> EconomicCalendarService:
    """Retourne l'instance singleton du service"""
    global _calendar_service_instance
    if _calendar_service_instance is None:
        _calendar_service_instance = EconomicCalendarService()
    return _calendar_service_instance
