
from typing import Optional
from pydantic import BaseModel

class BoursoramaStockData(BaseModel):
    nom: Optional[str]
    variation: Optional[str]
    cours: Optional[float]
    per: Optional[float]
    dividende: Optional[float]
    volume: Optional[str]
    capitalisation: Optional[str]
    objectif_cours: Optional[str]
    recommandation: Optional[str]


class FinvizStockData(BaseModel):
    # Identification
    ticker: str
    secteur: Optional[str]
    industrie: Optional[str]
    
    # Prix & Capitalisation
    cours: Optional[float]
    prev_close: Optional[float]
    capitalisation: Optional[str]
    price_to_earnings: Optional[float]
    forward_pe: Optional[float]
    peg: Optional[float]
    pb: Optional[float]
    pfcf: Optional[float]
    ps: Optional[float]
    
    # Rentabilité
    roe: Optional[float]
    roa: Optional[float]
    roi: Optional[float]
    eps_ttm: Optional[float]
    eps_next_y: Optional[float]
    eps_next_q: Optional[float]
    eps_past_5y: Optional[float]
    profit_margin: Optional[float]
    gross_margin: Optional[float]
    operating_margin: Optional[float]

    # Dividendes
    dividend_yield: Optional[float]
    dividend_ttm: Optional[str]
    dividend_ex_date: Optional[str]
    payout: Optional[float]

    # Endettement & Liquidité
    debt_to_equity: Optional[float]
    lt_debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    quick_ratio: Optional[float]

    # Actionnariat
    insider_own: Optional[float]
    inst_own: Optional[float]
    short_float: Optional[float]

    # Technique
    beta: Optional[float]
    rsi: Optional[float]
    sma20: Optional[float]
    sma50: Optional[float]
    sma200: Optional[float]
    atr: Optional[float]

    # Performance
    perf_week: Optional[float]
    perf_month: Optional[float]
    perf_quarter: Optional[float]
    perf_year: Optional[float]
    perf_ytd: Optional[float]

    # Cibles
    target_price: Optional[float]
    analyst_recom: Optional[float]

    # Divers
    employees: Optional[int]

