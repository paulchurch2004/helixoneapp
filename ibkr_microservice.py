"""
Microservice IBKR - Serveur Flask séparé pour éviter les conflits d'event loop
Ce service tourne en parallèle du backend principal et gère uniquement IBKR
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from ib_insync import IB, util
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional
import sys
import os
import threading
import time

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ibkr_microservice')

# Créer l'application Flask
app = Flask(__name__)
CORS(app)  # Permettre les requêtes cross-origin

# Variables globales
ib = IB()
connected = False
connection_config = {
    'host': '127.0.0.1',
    'port': 7496,
    'client_id': 2,
    'account_id': None
}
event_loop = None
loop_thread = None

def start_event_loop():
    """Démarrer un event loop dans un thread dédié"""
    global event_loop
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    logger.info("🔄 Event loop démarré dans le thread dédié")
    event_loop.run_forever()

def init_ibkr_thread():
    """Initialiser le thread IBKR avec son event loop"""
    global loop_thread
    loop_thread = threading.Thread(target=start_event_loop, daemon=True)
    loop_thread.start()
    time.sleep(1)  # Attendre que le loop soit prêt
    logger.info("✅ Thread IBKR initialisé")

@app.route('/health', methods=['GET'])
def health_check():
    """Vérifier que le microservice est en ligne"""
    return jsonify({
        'status': 'healthy',
        'service': 'IBKR Microservice',
        'connected': ib.isConnected(),
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/connect', methods=['POST'])
def connect_ibkr():
    """
    Connecter à IBKR avec les paramètres fournis

    Body JSON:
    {
        "host": "127.0.0.1",
        "port": 7496,
        "client_id": 2,
        "account_id": "U17421384"
    }
    """
    global connected, connection_config

    try:
        data = request.get_json()

        # Mettre à jour la configuration
        connection_config['host'] = data.get('host', '127.0.0.1')
        connection_config['port'] = int(data.get('port', 7496))
        connection_config['client_id'] = int(data.get('client_id', 2))
        connection_config['account_id'] = data.get('account_id')

        # Fonction async pour connecter
        async def do_connect():
            try:
                # Déconnecter si déjà connecté
                if ib.isConnected():
                    ib.disconnect()
                    await asyncio.sleep(1)

                # Connecter
                logger.info(f"Connexion à IBKR: {connection_config['host']}:{connection_config['port']}")
                await ib.connectAsync(
                    connection_config['host'],
                    connection_config['port'],
                    clientId=connection_config['client_id'],
                    timeout=20
                )
                return True
            except Exception as e:
                logger.error(f"Erreur dans do_connect: {e}")
                raise

        # Exécuter dans l'event loop dédié
        future = asyncio.run_coroutine_threadsafe(do_connect(), event_loop)
        result = future.result(timeout=30)  # Attendre max 30 secondes

        connected = True
        logger.info("✅ Connecté à IBKR")

        return jsonify({
            'success': True,
            'connected': True,
            'message': 'Connexion IBKR réussie',
            'config': connection_config
        })

    except Exception as e:
        logger.error(f"❌ Erreur connexion IBKR: {e}")
        connected = False
        return jsonify({
            'success': False,
            'connected': False,
            'error': str(e)
        })

@app.route('/disconnect', methods=['POST'])
def disconnect_ibkr():
    """Déconnecter d'IBKR"""
    global connected

    try:
        if ib.isConnected():
            ib.disconnect()
            connected = False
            logger.info("Déconnecté d'IBKR")

        return jsonify({
            'success': True,
            'connected': False,
            'message': 'Déconnecté d\'IBKR'
        })

    except Exception as e:
        logger.error(f"Erreur déconnexion: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Obtenir le statut de connexion"""
    return jsonify({
        'connected': ib.isConnected(),
        'config': connection_config if ib.isConnected() else None
    })

@app.route('/portfolio', methods=['GET'])
def get_portfolio():
    """Récupérer le portefeuille"""
    if not ib.isConnected():
        return jsonify({
            'connected': False,
            'error': 'Non connecté à IBKR'
        }), 503

    try:
        # Récupérer le résumé du compte
        account_id = connection_config.get('account_id') or ib.managedAccounts()[0]

        # Fonction async pour récupérer les données
        async def get_data():
            summary = await ib.accountSummaryAsync(account_id)
            positions = ib.positions()
            return summary, positions

        # Exécuter dans l'event loop dédié
        future = asyncio.run_coroutine_threadsafe(get_data(), event_loop)
        summary, positions = future.result(timeout=20)

        # Convertir en dict
        summary_dict = {}
        for item in summary:
            try:
                summary_dict[item.tag] = float(item.value) if item.value else 0
            except (ValueError, TypeError):
                summary_dict[item.tag] = item.value if item.value else ""

        # Traiter les positions
        positions_list = []
        for pos in positions:
            try:
                position_dict = {
                    'symbol': pos.contract.symbol,
                    'position': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_price': pos.avgCost,
                    'market_value': pos.position * pos.avgCost,
                    'unrealized_pnl': 0
                }
                positions_list.append(position_dict)
            except Exception as e:
                logger.error(f"Erreur position: {e}")

        # Construire la réponse
        portfolio = {
            'account_id': account_id,
            'net_liquidation': summary_dict.get('NetLiquidation', 0),
            'total_cash': summary_dict.get('TotalCashValue', 0),
            'stock_value': summary_dict.get('GrossPositionValue', 0),
            'unrealized_pnl': summary_dict.get('UnrealizedPnL', 0),
            'realized_pnl': summary_dict.get('RealizedPnL', 0),
            'daily_pnl': summary_dict.get('DailyPnL', 0),
            'buying_power': summary_dict.get('BuyingPower', 0),
            'available_funds': summary_dict.get('AvailableFunds', 0),
            'currency': summary_dict.get('Currency', 'EUR'),
            'positions': positions_list,
            'timestamp': datetime.utcnow().isoformat()
        }

        logger.info(f"Portfolio récupéré: {len(positions_list)} positions")

        return jsonify({
            'connected': True,
            'portfolio': portfolio
        })

    except Exception as e:
        logger.error(f"Erreur récupération portfolio: {e}")
        return jsonify({
            'connected': True,
            'error': str(e)
        }), 500

@app.route('/dashboard', methods=['GET'])
def get_dashboard():
    """
    Endpoint principal pour le dashboard
    Retourne toutes les données nécessaires
    """
    if not ib.isConnected():
        return jsonify({
            'connected': False,
            'message': 'Non connecté à IBKR. Veuillez configurer la connexion.',
            'portfolio': None,
            'alerts': [],
            'performance': []
        })

    try:
        # Récupérer le portfolio directement
        account_id = connection_config.get('account_id') or ib.managedAccounts()[0]

        # Fonction async pour récupérer les données
        async def get_data():
            summary = await ib.accountSummaryAsync(account_id)
            positions = ib.positions()
            return summary, positions

        # Exécuter dans l'event loop dédié
        future = asyncio.run_coroutine_threadsafe(get_data(), event_loop)
        summary, positions = future.result(timeout=20)

        # Convertir en dict
        summary_dict = {}
        for item in summary:
            try:
                summary_dict[item.tag] = float(item.value) if item.value else 0
            except (ValueError, TypeError):
                summary_dict[item.tag] = item.value if item.value else ""

        # Traiter les positions
        positions_list = []
        for pos in positions:
            try:
                position_dict = {
                    'symbol': pos.contract.symbol,
                    'position': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_price': pos.avgCost,
                    'market_value': pos.position * pos.avgCost,
                    'unrealized_pnl': 0
                }
                positions_list.append(position_dict)
            except Exception as e:
                logger.error(f"Erreur position: {e}")

        # Construire le portfolio
        portfolio = {
            'account_id': account_id,
            'net_liquidation': summary_dict.get('NetLiquidation', 0),
            'total_cash': summary_dict.get('TotalCashValue', 0),
            'stock_value': summary_dict.get('GrossPositionValue', 0),
            'unrealized_pnl': summary_dict.get('UnrealizedPnL', 0),
            'realized_pnl': summary_dict.get('RealizedPnL', 0),
            'daily_pnl': summary_dict.get('DailyPnL', 0),
            'buying_power': summary_dict.get('BuyingPower', 0),
            'available_funds': summary_dict.get('AvailableFunds', 0),
            'currency': summary_dict.get('Currency', 'EUR'),
            'positions': positions_list,
            'timestamp': datetime.utcnow().isoformat()
        }

        logger.info(f"Portfolio récupéré: {len(positions_list)} positions")

        # Pour l'instant, pas d'alertes ni de performance (à implémenter plus tard)
        return jsonify({
            'connected': True,
            'portfolio': portfolio,
            'alerts': [],
            'performance': [],
            'last_update': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Erreur dashboard: {e}")
        return jsonify({
            'connected': False,
            'message': str(e),
            'portfolio': None,
            'alerts': [],
            'performance': []
        }), 500


if __name__ == '__main__':
    logger.info("🚀 Démarrage du microservice IBKR...")

    # Initialiser le thread IBKR avec event loop
    init_ibkr_thread()

    logger.info("📡 API disponible sur http://127.0.0.1:8001")
    logger.info("📊 Health check: http://127.0.0.1:8001/health")
    logger.info("💼 Dashboard: http://127.0.0.1:8001/dashboard")

    # Démarrer le serveur Flask
    app.run(
        host='127.0.0.1',
        port=8001,
        debug=False,
        threaded=True
    )
