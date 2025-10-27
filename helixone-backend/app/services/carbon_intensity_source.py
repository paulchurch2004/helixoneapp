"""
Carbon Intensity API Data Source
Documentation: https://carbon-intensity.github.io/api-definitions/

Free Tier Limits:
- UNLIMITED and FREE
- No API key required
- No rate limiting (reasonable use expected)

Coverage:
- UK electricity carbon intensity (gCO2/kWh)
- Real-time data and forecasts
- Regional breakdowns
- Generation mix (wind, solar, gas, nuclear, etc.)
- Useful for ESG scoring and carbon footprint calculation

API Provider: UK National Grid ESO
"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class CarbonIntensitySource:
    """
    Carbon Intensity API collector for ESG data

    Free: Unlimited, no API key required
    Use case: ESG scoring, carbon footprint analysis, renewable energy tracking
    """

    def __init__(self):
        """Initialize Carbon Intensity API source"""
        self.base_url = "https://api.carbonintensity.org.uk"

        # No rate limiting needed (unlimited), but be courteous
        self.min_request_interval = 1.0
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce courtesy rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str) -> Dict:
        """
        Make API request

        Args:
            endpoint: API endpoint path

        Returns:
            JSON response as dictionary
        """
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"

        headers = {
            'Accept': 'application/json'
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Carbon Intensity API request failed: {str(e)}")

    def get_current_intensity(self) -> Dict:
        """
        Get current carbon intensity for UK

        Returns:
            Dictionary with:
            {
                'from': '2025-10-22T12:00Z',
                'to': '2025-10-22T12:30Z',
                'intensity': {
                    'forecast': 150,
                    'actual': 145,
                    'index': 'moderate'  # very low, low, moderate, high, very high
                }
            }

        Example:
            >>> ci = carbon.get_current_intensity()
            >>> print(f"Current intensity: {ci['intensity']['actual']} gCO2/kWh")
        """
        result = self._make_request('intensity')

        if result.get('data') and len(result['data']) > 0:
            return result['data'][0]
        else:
            raise Exception("No current intensity data available")

    def get_intensity_factors(self) -> List[Dict]:
        """
        Get carbon intensity factors by fuel type

        Returns:
            List of dictionaries with fuel types and their carbon intensity

        Example:
            >>> factors = carbon.get_intensity_factors()
            >>> for fuel in factors:
            ...     print(f"{fuel['fuel']}: {fuel['intensity']} gCO2/kWh")
        """
        result = self._make_request('intensity/factors')

        if result.get('data'):
            return result['data']
        else:
            raise Exception("No intensity factors available")

    def get_generation_mix(self) -> Dict:
        """
        Get current electricity generation mix (% by source)

        Returns:
            Dictionary with generation mix:
            {
                'from': '2025-10-22T12:00Z',
                'to': '2025-10-22T12:30Z',
                'generationmix': [
                    {'fuel': 'wind', 'perc': 25.5},
                    {'fuel': 'solar', 'perc': 10.2},
                    {'fuel': 'gas', 'perc': 30.0},
                    ...
                ]
            }
        """
        result = self._make_request('generation')

        if result.get('data') and len(result['data']) > 0:
            return result['data'][0]
        else:
            raise Exception("No generation mix data available")

    def get_intensity_statistics(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> Dict:
        """
        Get intensity statistics for date range

        Args:
            from_date: Start date (YYYY-MM-DDTHH:MMZ format) - defaults to 24h ago
            to_date: End date (YYYY-MM-DDTHH:MMZ format) - defaults to now

        Returns:
            Dictionary with statistics (min, max, average, etc.)

        Example:
            >>> # Last 24 hours stats
            >>> stats = carbon.get_intensity_statistics()
            >>> print(f"Average: {stats['data'][0]['intensity']['average']} gCO2/kWh")
        """
        if not from_date:
            # Default to 24 hours ago
            from_dt = datetime.utcnow() - timedelta(hours=24)
            from_date = from_dt.strftime('%Y-%m-%dT%H:%MZ')

        if not to_date:
            # Default to now
            to_dt = datetime.utcnow()
            to_date = to_dt.strftime('%Y-%m-%dT%H:%MZ')

        endpoint = f"intensity/stats/{from_date}/{to_date}"

        return self._make_request(endpoint)

    def get_regional_intensity(self, postcode: Optional[str] = None) -> List[Dict]:
        """
        Get carbon intensity by UK region

        Args:
            postcode: Optional UK postcode to get specific region (e.g., 'SW1' or 'OX1')

        Returns:
            List of dictionaries with regional intensities

        Example:
            >>> # All regions
            >>> regions = carbon.get_regional_intensity()
            >>> for region in regions:
            ...     print(f"{region['shortname']}: {region['data'][0]['intensity']['forecast']}")
        """
        if postcode:
            # Get intensity for specific postcode
            endpoint = f"regional/postcode/{postcode}"
        else:
            # Get all regions
            endpoint = "regional"

        result = self._make_request(endpoint)

        if result.get('data'):
            return result['data'] if isinstance(result['data'], list) else [result['data']]
        else:
            raise Exception("No regional data available")

    def get_renewable_percentage(self) -> float:
        """
        Calculate current percentage of renewable energy in the mix

        Returns:
            Float percentage of renewable energy (wind + solar + hydro + biomass)

        Example:
            >>> renewable_pct = carbon.get_renewable_percentage()
            >>> print(f"Renewable energy: {renewable_pct:.1f}%")
        """
        mix = self.get_generation_mix()

        renewable_sources = ['wind', 'solar', 'hydro', 'biomass']
        renewable_total = 0.0

        for source in mix.get('generationmix', []):
            if source['fuel'] in renewable_sources:
                renewable_total += source['perc']

        return round(renewable_total, 2)

    def get_fossil_fuel_percentage(self) -> float:
        """
        Calculate current percentage of fossil fuel energy in the mix

        Returns:
            Float percentage of fossil fuels (gas + coal + oil)
        """
        mix = self.get_generation_mix()

        fossil_sources = ['gas', 'coal', 'oil']
        fossil_total = 0.0

        for source in mix.get('generationmix', []):
            if source['fuel'] in fossil_sources:
                fossil_total += source['perc']

        return round(fossil_total, 2)

    def is_clean_energy_period(self, threshold: float = 50.0) -> Dict:
        """
        Check if current period has high renewable energy

        Args:
            threshold: Minimum percentage of renewable energy to be considered "clean" (default 50%)

        Returns:
            Dictionary with:
            {
                'is_clean': True,
                'renewable_pct': 65.5,
                'carbon_intensity': 120,
                'message': 'Clean energy period - good time to run energy-intensive tasks'
            }
        """
        renewable_pct = self.get_renewable_percentage()
        current = self.get_current_intensity()
        intensity = current['intensity'].get('actual') or current['intensity'].get('forecast')

        is_clean = renewable_pct >= threshold

        if is_clean:
            message = f"Clean energy period ({renewable_pct:.1f}% renewable) - good time for energy-intensive tasks"
        else:
            message = f"High carbon period ({renewable_pct:.1f}% renewable) - consider delaying non-urgent energy use"

        return {
            'is_clean': is_clean,
            'renewable_pct': renewable_pct,
            'carbon_intensity': intensity,
            'index': current['intensity']['index'],
            'message': message
        }

    def get_esg_score(self) -> Dict:
        """
        Calculate simplified ESG score based on current carbon intensity and renewables

        Returns:
            Dictionary with ESG score (0-100) and components

        Note:
            Simplified scoring for demonstration:
            - 50% weight on renewable percentage
            - 50% weight on carbon intensity (inverted, lower is better)
        """
        renewable_pct = self.get_renewable_percentage()
        current = self.get_current_intensity()
        intensity = current['intensity'].get('actual') or current['intensity'].get('forecast')

        # Score renewable percentage (0-100 maps to 0-50 points)
        renewable_score = (renewable_pct / 100) * 50

        # Score carbon intensity (lower is better, typical range 50-400 gCO2/kWh)
        # Invert and normalize: 400 = 0 points, 50 = 50 points
        max_intensity = 400
        min_intensity = 50
        normalized_intensity = max(0, min(1, (max_intensity - intensity) / (max_intensity - min_intensity)))
        intensity_score = normalized_intensity * 50

        total_score = round(renewable_score + intensity_score, 1)

        # Grade
        if total_score >= 80:
            grade = 'A'
        elif total_score >= 70:
            grade = 'B'
        elif total_score >= 60:
            grade = 'C'
        elif total_score >= 50:
            grade = 'D'
        else:
            grade = 'F'

        return {
            'esg_score': total_score,
            'grade': grade,
            'renewable_pct': renewable_pct,
            'carbon_intensity': intensity,
            'components': {
                'renewable_score': round(renewable_score, 1),
                'intensity_score': round(intensity_score, 1)
            }
        }


# === Singleton Pattern ===

_carbon_instance = None

def get_carbon_intensity_collector() -> CarbonIntensitySource:
    """
    Get or create Carbon Intensity collector instance (singleton pattern)

    Returns:
        CarbonIntensitySource instance
    """
    global _carbon_instance

    if _carbon_instance is None:
        _carbon_instance = CarbonIntensitySource()

    return _carbon_instance
