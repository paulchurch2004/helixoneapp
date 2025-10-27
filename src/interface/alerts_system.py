import json
import threading
import time
import yfinance as yf
from datetime import datetime
from plyer import notification  # pip install plyer

class AlertSystem:
    def __init__(self):
        self.alerts_file = "data/alerts.json"
        self.active_alerts = []
        self.monitoring = False
        self.load_alerts()
    
    def add_alert(self, ticker, target_price, condition="above"):
        alert = {
            "ticker": ticker,
            "target": target_price,
            "condition": condition,  # "above" ou "below"
            "created": datetime.now().isoformat(),
            "triggered": False
        }
        self.active_alerts.append(alert)
        self.save_alerts()
        
    def check_alerts(self):
        while self.monitoring:
            for alert in self.active_alerts:
                if not alert["triggered"]:
                    try:
                        stock = yf.Ticker(alert["ticker"])
                        current_price = stock.history(period="1d")["Close"].iloc[-1]
                        
                        if alert["condition"] == "above" and current_price >= alert["target"]:
                            self.trigger_alert(alert, current_price)
                        elif alert["condition"] == "below" and current_price <= alert["target"]:
                            self.trigger_alert(alert, current_price)
                    except:
                        pass
            time.sleep(60)  # Check every minute
    
    def trigger_alert(self, alert, current_price):
        notification.notify(
            title=f"🔔 Alerte Prix {alert['ticker']}",
            message=f"{alert['ticker']} a atteint {current_price:.2f}€ (cible: {alert['target']})",
            timeout=10
        )
        alert["triggered"] = True
        self.save_alerts()