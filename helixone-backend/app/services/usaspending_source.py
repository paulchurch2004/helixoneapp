"""
USAspending.gov API Data Source
Documentation: https://api.usaspending.gov/

Free Tier Limits:
- UNLIMITED and FREE
- No API key required
- Reasonable use rate limiting expected

Coverage:
- US Federal government spending data
- Contracts, grants, loans, direct payments
- Agency spending
- Company/recipient information
- Useful for analyzing government sector exposure

API Provider: US Department of the Treasury
"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class USAspendingSource:
    """
    USAspending.gov API collector for federal spending data

    Free: Unlimited, no API key required
    Use case: Government contracts analysis, company sector exposure
    """

    def __init__(self):
        """Initialize USAspending.gov API source"""
        self.base_url = "https://api.usaspending.gov/api/v2"

        # Courtesy rate limiting
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

    def _make_request(self, endpoint: str, method: str = 'POST', data: Optional[Dict] = None) -> Dict:
        """
        Make API request

        Args:
            endpoint: API endpoint path
            method: HTTP method ('GET' or 'POST')
            data: Request body for POST requests

        Returns:
            JSON response as dictionary
        """
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"

        headers = {
            'Content-Type': 'application/json'
        }

        try:
            if method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)
            else:
                response = requests.get(url, headers=headers, timeout=30)

            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"USAspending.gov request failed: {str(e)}")

    def search_spending_by_recipient(
        self,
        recipient_name: str,
        fiscal_year: Optional[int] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search federal spending by recipient company name

        Args:
            recipient_name: Company/recipient name (e.g., "Lockheed Martin", "Boeing")
            fiscal_year: Fiscal year (optional, e.g., 2024)
            limit: Maximum number of results

        Returns:
            List of spending transactions

        Example:
            >>> contracts = usa.search_spending_by_recipient("Lockheed Martin", limit=5)
            >>> for contract in contracts:
            ...     print(f"{contract['Award Amount']}: {contract['Description']}")
        """
        endpoint = "search/spending_by_award"

        # Build filters
        filters = {
            "recipient_search_text": [recipient_name],
            "award_type_codes": ["A", "B", "C", "D"]  # Contracts
        }

        if fiscal_year:
            filters["time_period"] = [{
                "start_date": f"{fiscal_year}-10-01",
                "end_date": f"{fiscal_year + 1}-09-30"
            }]

        data = {
            "filters": filters,
            "fields": [
                "Award ID",
                "Recipient Name",
                "Award Amount",
                "Description",
                "Start Date",
                "End Date",
                "Awarding Agency",
                "Award Type"
            ],
            "limit": limit,
            "page": 1
        }

        result = self._make_request(endpoint, method='POST', data=data)

        return result.get('results', [])

    def get_agency_spending(
        self,
        agency_code: str,
        fiscal_year: Optional[int] = None
    ) -> Dict:
        """
        Get spending totals for a federal agency

        Args:
            agency_code: Agency code (e.g., "097" for DOD, "068" for NASA)
            fiscal_year: Fiscal year (optional)

        Returns:
            Dictionary with agency spending data

        Example:
            >>> dod_spending = usa.get_agency_spending("097", fiscal_year=2024)
        """
        endpoint = f"agency/{agency_code}"

        result = self._make_request(endpoint, method='GET')

        return result

    def search_contracts_by_naics(
        self,
        naics_code: str,
        fiscal_year: Optional[int] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search contracts by NAICS industry code

        Args:
            naics_code: NAICS code (e.g., "336411" for Aircraft Manufacturing)
            fiscal_year: Fiscal year
            limit: Maximum results

        Returns:
            List of contracts in that industry

        Example:
            >>> # Aircraft manufacturing contracts
            >>> contracts = usa.search_contracts_by_naics("336411", limit=5)
        """
        endpoint = "search/spending_by_award"

        filters = {
            "naics_codes": [naics_code],
            "award_type_codes": ["A", "B", "C", "D"]
        }

        if fiscal_year:
            filters["time_period"] = [{
                "start_date": f"{fiscal_year}-10-01",
                "end_date": f"{fiscal_year + 1}-09-30"
            }]

        data = {
            "filters": filters,
            "limit": limit
        }

        result = self._make_request(endpoint, method='POST', data=data)

        return result.get('results', [])

    def get_top_contractors(
        self,
        fiscal_year: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get top federal contractors by award amount

        Args:
            fiscal_year: Fiscal year (defaults to current FY)
            limit: Number of top contractors to return

        Returns:
            List of top contractors with total amounts

        Example:
            >>> top_100 = usa.get_top_contractors(fiscal_year=2024, limit=100)
            >>> for i, contractor in enumerate(top_100, 1):
            ...     print(f"{i}. {contractor['name']}: ${contractor['amount']:,.0f}")
        """
        endpoint = "search/spending_by_award"

        if not fiscal_year:
            # Default to current fiscal year
            now = datetime.now()
            fiscal_year = now.year if now.month >= 10 else now.year - 1

        filters = {
            "time_period": [{
                "start_date": f"{fiscal_year}-10-01",
                "end_date": f"{fiscal_year + 1}-09-30"
            }],
            "award_type_codes": ["A", "B", "C", "D"]  # All contract types
        }

        data = {
            "filters": filters,
            "fields": [
                "Recipient Name",
                "Award Amount",
                "Award ID",
                "Description"
            ],
            "limit": limit,
            "sort": "Award Amount",
            "order": "desc"
        }

        result = self._make_request(endpoint, method='POST', data=data)

        # Extract and format results
        contractors = []
        seen_recipients = {}

        for award in result.get('results', []):
            recipient = award.get('Recipient Name', 'Unknown')
            amount = award.get('Award Amount', 0)

            if recipient in seen_recipients:
                seen_recipients[recipient] += amount
            else:
                seen_recipients[recipient] = amount

        # Convert to list and sort
        for name, amount in seen_recipients.items():
            contractors.append({
                'name': name,
                'amount': amount
            })

        contractors.sort(key=lambda x: x['amount'], reverse=True)

        return contractors[:limit]

    def get_company_contract_summary(
        self,
        company_name: str,
        years: int = 3
    ) -> Dict:
        """
        Get summary of company's federal contracts over recent years

        Args:
            company_name: Company name
            years: Number of recent fiscal years to analyze

        Returns:
            Dictionary with summary statistics

        Example:
            >>> summary = usa.get_company_contract_summary("Boeing", years=3)
            >>> print(f"Total contracts: {summary['total_amount']:,.0f}")
            >>> print(f"Contract count: {summary['contract_count']}")
        """
        current_year = datetime.now().year
        current_month = datetime.now().month

        # Determine current fiscal year
        if current_month >= 10:
            current_fy = current_year + 1
        else:
            current_fy = current_year

        total_amount = 0
        total_count = 0
        yearly_breakdown = []

        for i in range(years):
            fy = current_fy - i - 1

            contracts = self.search_spending_by_recipient(
                company_name,
                fiscal_year=fy,
                limit=100
            )

            year_amount = sum(c.get('Award Amount', 0) for c in contracts)
            year_count = len(contracts)

            total_amount += year_amount
            total_count += year_count

            yearly_breakdown.append({
                'fiscal_year': fy,
                'amount': year_amount,
                'count': year_count
            })

        return {
            'company_name': company_name,
            'total_amount': total_amount,
            'contract_count': total_count,
            'years_analyzed': years,
            'yearly_breakdown': yearly_breakdown,
            'average_annual_amount': total_amount / years if years > 0 else 0
        }

    def get_spending_trends(
        self,
        agency_code: Optional[str] = None,
        years: int = 5
    ) -> List[Dict]:
        """
        Get spending trends over multiple fiscal years

        Args:
            agency_code: Specific agency (optional, None for total federal)
            years: Number of years to analyze

        Returns:
            List of yearly spending totals

        Example:
            >>> trends = usa.get_spending_trends(years=5)
            >>> for year in trends:
            ...     print(f"FY{year['fiscal_year']}: ${year['total_spent']:,.0f}")
        """
        # Note: This is a simplified version. Full implementation would
        # require aggregating spending data across all awards.

        current_year = datetime.now().year
        current_month = datetime.now().month

        if current_month >= 10:
            current_fy = current_year + 1
        else:
            current_fy = current_year

        trends = []

        for i in range(years):
            fy = current_fy - i - 1

            # For demonstration, using a simple search
            endpoint = "search/spending_by_award"

            filters = {
                "time_period": [{
                    "start_date": f"{fy}-10-01",
                    "end_date": f"{fy + 1}-09-30"
                }],
                "award_type_codes": ["A", "B", "C", "D"]
            }

            if agency_code:
                filters["awarding_agency_codes"] = [agency_code]

            data = {
                "filters": filters,
                "limit": 1  # We just want the total
            }

            try:
                result = self._make_request(endpoint, method='POST', data=data)

                total = result.get('total_obligation', 0)

                trends.append({
                    'fiscal_year': fy,
                    'total_spent': total
                })

            except:
                # If request fails, append with None
                trends.append({
                    'fiscal_year': fy,
                    'total_spent': None
                })

        return trends


# === Singleton Pattern ===

_usaspending_instance = None

def get_usaspending_collector() -> USAspendingSource:
    """
    Get or create USAspending collector instance (singleton pattern)

    Returns:
        USAspendingSource instance
    """
    global _usaspending_instance

    if _usaspending_instance is None:
        _usaspending_instance = USAspendingSource()

    return _usaspending_instance
