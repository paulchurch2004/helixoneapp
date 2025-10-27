import yfinance as yf
import pandas as pd
import ta

def analyse_stock_for_alert(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="30d", interval="1h")

        if df.empty or df.shape[0] < 20:
            return None

        # Ajout des indicateurs techniques
        df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        macd = ta.trend.MACD(df["Close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["ema50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
        df["ema200"] = ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator()
        df["volume_sma"] = df["Volume"].rolling(10).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        score = 0
        if last["rsi"] < 30:
            score += 1
        if prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]:
            score += 1
        if last["ema50"] > last["ema200"]:
            score += 1
        if last["Volume"] > last["volume_sma"]:
            score += 1

        if score >= 3:
            return {
                "ticker": ticker,
                "price": last["Close"],
                "score": score,
                "rsi": last["rsi"],
                "macd": last["macd"]
            }

    except Exception as e:
        print(f"❌ Erreur analyse {ticker} : {e}")
        return None
