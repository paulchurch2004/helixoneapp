
import requests
from bs4 import BeautifulSoup
from typing import Optional
from data_sources.schemas import FinvizStockData

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

def parse_float(val: str) -> Optional[float]:
    try:
        val = val.replace('%', '').replace(',', '').replace('B', 'e9').replace('M', 'e6').replace('K', 'e3').strip()
        return float(eval(val))
    except:
        return None

def get_finviz_data(ticker: str) -> Optional[FinvizStockData]:
    ticker = ticker.split('.')[0]
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    print(f"[Finviz+] Scraping {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find("table", class_="snapshot-table2")
        if not table:
            print(f"[Finviz] ❌ Tableau non trouvé pour {ticker}")
            return None

        data = {}
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            for i in range(0, len(cells), 2):
                key = cells[i].text.strip()
                val = cells[i + 1].text.strip()
                data[key] = val

        return FinvizStockData(
            ticker=ticker,
            secteur=data.get("Sector"),
            industrie=data.get("Industry"),
            cours=parse_float(data.get("Price", "")),
            prev_close=parse_float(data.get("Prev Close", "")),
            capitalisation=data.get("Market Cap"),
            price_to_earnings=parse_float(data.get("P/E", "")),
            forward_pe=parse_float(data.get("Forward P/E", "")),
            peg=parse_float(data.get("PEG", "")),
            pb=parse_float(data.get("P/B", "")),
            pfcf=parse_float(data.get("P/FCF", "")),
            ps=parse_float(data.get("P/S", "")),
            roe=parse_float(data.get("ROE", "")),
            roa=parse_float(data.get("ROA", "")),
            roi=parse_float(data.get("ROI", "")),
            eps_ttm=parse_float(data.get("EPS (ttm)", "")),
            eps_next_y=parse_float(data.get("EPS next Y", "")),
            eps_next_q=parse_float(data.get("EPS next Q", "")),
            eps_past_5y=parse_float(data.get("EPS past 5Y", "")),
            profit_margin=parse_float(data.get("Profit Margin", "")),
            gross_margin=parse_float(data.get("Gross Margin", "")),
            operating_margin=parse_float(data.get("Oper. Margin", "")),
            dividend_yield=parse_float(data.get("Dividend", "")),
            dividend_ttm=data.get("Dividend TTM"),
            dividend_ex_date=data.get("Dividend Ex-Date"),
            payout=parse_float(data.get("Payout", "")),
            debt_to_equity=parse_float(data.get("Debt/Eq", "")),
            lt_debt_to_equity=parse_float(data.get("LT Debt/Eq", "")),
            current_ratio=parse_float(data.get("Current Ratio", "")),
            quick_ratio=parse_float(data.get("Quick Ratio", "")),
            insider_own=parse_float(data.get("Insider Own", "")),
            inst_own=parse_float(data.get("Inst Own", "")),
            short_float=parse_float(data.get("Short Float", "")),
            beta=parse_float(data.get("Beta", "")),
            rsi=parse_float(data.get("RSI (14)", "")),
            sma20=parse_float(data.get("SMA20", "")),
            sma50=parse_float(data.get("SMA50", "")),
            sma200=parse_float(data.get("SMA200", "")),
            atr=parse_float(data.get("ATR (14)", "")),
            perf_week=parse_float(data.get("Perf Week", "")),
            perf_month=parse_float(data.get("Perf Month", "")),
            perf_quarter=parse_float(data.get("Perf Quarter", "")),
            perf_year=parse_float(data.get("Perf Year", "")),
            perf_ytd=parse_float(data.get("Perf YTD", "")),
            target_price=parse_float(data.get("Target Price", "")),
            analyst_recom=parse_float(data.get("Recom", "")),
            employees=parse_float(data.get("Employees", ""))
        )

    except Exception as e:
        print(f"[Finviz ❌] Erreur pour {ticker} : {e}")
        return None
