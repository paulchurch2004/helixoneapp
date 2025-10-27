import os
import json
import time
import logging
import asyncio
from typing import Optional, Dict, List

import pandas as pd
import aiohttp

from data_sources import source_manager
from data_sources.twelvedata_source import get_history as get_twelvedata_history
from data_sources.eod_source import get_history as get_eod_history
from data_sources.scraper_helpers import get_boursorama_data
from data_sources.scraper_finviz import get_finviz_data
from data_sources.scraper_source import get_data as get_scraper_data
from data_sources.schemas import BoursoramaStockData, FinvizStockData

# 📦 Paramètres de cache
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
CACHE_TTL = 300  # secondes

# 📋 Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HelixOne")


def load_cache(ticker: str) -> Optional[dict]:
    path = os.path.join(CACHE_DIR, f"{ticker}.json")
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > CACHE_TTL:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(ticker: str, data: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f"{ticker}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


async def get_primary_data(ticker: str) -> Dict:
    logger.info(f"🔍 Scraping Boursorama & Finviz pour {ticker}")
    result = {}

    try:
        async with aiohttp.ClientSession() as session:
            boursorama = await get_boursorama_data(ticker, session)
        if boursorama:
            result.update(boursorama.model_dump(exclude_none=True))
    except Exception as e:
        logger.warning(f"❌ Erreur Boursorama : {e}")

    try:
        finviz = get_finviz_data(ticker)
        if finviz:
            result.update(finviz.model_dump(exclude_none=True))
    except Exception as e:
        logger.warning(f"❌ Erreur Finviz : {e}")

    try:
        df_hist, _ = await get_eod_history(ticker)
        if not df_hist.empty:
            df_hist.index = df_hist.index.astype(str)
            result["historique"] = df_hist.reset_index().to_dict(orient="records")
    except Exception as e:
        logger.warning(f"⚠️ Historique indisponible via EOD : {e}")

    return result


def merge_sources(data_list: List[dict]) -> dict:
    merged = {}
    for source in data_list:
        for key, value in source.items():
            if key not in merged or merged[key] is None:
                merged[key] = value
    return merged


async def enrich_data(ticker: str, current_data: dict) -> dict:
    logger.info("♻️ Enrichissement via Scraper générique")
    try:
        enriched = await get_scraper_data(ticker)
        return merge_sources([current_data, enriched])
    except Exception as e:
        logger.warning(f"⚠️ Enrichissement échoué : {e}")
        return current_data


async def get_best_history(ticker: str) -> Optional[pd.DataFrame]:
    logger.info(f"📈 Récupération historique pour {ticker}")
    history_sources = [get_twelvedata_history, get_eod_history]
    for source in history_sources:
        try:
            df = await source(ticker)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"⚠️ Historique échoué via {source.__name__}: {e}")
    return None


async def get_stock_data(ticker: str) -> dict:
    logger.info(f"🚀 Récupération des données pour {ticker}")
    cache = load_cache(ticker)
    if cache:
        logger.info("✅ Données chargées depuis le cache")
        return cache

    data = await get_primary_data(ticker)
    data = await enrich_data(ticker, data)

    history = await get_best_history(ticker)
    if history is not None:
        history.index = history.index.astype(str)
        data["historique"] = history.to_dict(orient="records")

    save_cache(ticker, data)
    logger.info("✅ Données sauvegardées dans le cache")
    return data
