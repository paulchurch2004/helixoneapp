"""
Moteur d'analyse financière FXI v2.0 avec analyse macro-économique
Architecture modulaire professionnelle
"""

from .core.engine import FXIEngine, analyze_ticker, AnalysisResult
from .core.config import EngineConfig, DEFAULT_CONFIG
import requests
import os

__version__ = "2.0.0-macro"
__all__ = ["FXIEngine", "analyze_ticker", "AnalysisResult", "EngineConfig", "DEFAULT_CONFIG"]

# Interface de compatibilité pour l'ancien code
def get_analysis(ticker: str, mode: str = "Standard", auth_token: str = None):
    """
    Interface rétrocompatible qui appelle l'API backend HelixOne
    Retourne un dict au lieu d'un AnalysisResult

    Args:
        ticker: Symbole de l'action
        mode: Mode d'analyse (Standard, Conservative, Aggressive)
        auth_token: Token d'authentification (depuis auth_manager)
    """
    # URL du backend API
    backend_url = os.environ.get("HELIXONE_API_URL", "http://127.0.0.1:8000")

    # Token d'authentification (depuis paramètre ou environnement)
    token = auth_token or os.environ.get("HELIXONE_API_TOKEN")

    try:
        # Appeler l'API backend
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.post(
            f"{backend_url}/api/analysis/complete",
            json={"ticker": ticker, "mode": mode},
            headers=headers,
            timeout=120  # 2 minutes timeout pour les analyses
        )

        if response.status_code == 200:
            result = response.json()

            # Convertir au format attendu par l'interface
            return {
                'score_fxi': result.get('final_score', 0),
                'score': result.get('final_score', 0),
                'recommandation': result.get('recommendation', 'N/A'),
                'status': 'success',
                'indicateurs': {
                    'RSI': 'calculé',
                    'MACD': 'calculé',
                    'trend': 'analysé',
                    'volatilite': 'calculée'
                },
                'fondamentaux': {
                    'PE': result.get('details', {}).get('pe_ratio'),
                    'prix': result.get('details', {}).get('current_price'),
                    'secteur': result.get('details', {}).get('sector'),
                    'industrie': result.get('details', {}).get('industry')
                },
                'scores_detailles': {
                    'technique': result.get('technical_score', 0),
                    'fondamental': result.get('fundamental_score', 0),
                    'sentiment': result.get('sentiment_score', 0),
                    'risque': result.get('risk_score', 0),
                    'macro': result.get('macro_score', 0)
                },
                'esg': {
                    'total': 'N/A',
                    'environment': 'N/A',
                    'social': 'N/A',
                    'governance': 'N/A',
                    'grade': 'N/A'
                },
                'macro_environment': result.get('details', {}).get('macro_environment', 'N/A'),
                'timestamp': result.get('timestamp'),
                'execution_time': result.get('execution_time', 0),
                'confidence': result.get('confidence', 0),
                'data_quality': result.get('data_quality', 0)
            }
        else:
            # En cas d'erreur API, retourner un résultat d'erreur
            error_detail = response.json().get('detail', 'Erreur inconnue') if response.text else response.reason
            return {
                'score_fxi': 0,
                'score': 0,
                'recommandation': 'ERREUR',
                'status': 'error',
                'error': f"API Error {response.status_code}: {error_detail}",
                'indicateurs': {},
                'fondamentaux': {},
                'scores_detailles': {
                    'technique': 0,
                    'fondamental': 0,
                    'sentiment': 0,
                    'risque': 0,
                    'macro': 0
                },
                'esg': {},
                'macro_environment': 'N/A',
                'timestamp': None,
                'execution_time': 0,
                'confidence': 0,
                'data_quality': 0
            }

    except requests.exceptions.ConnectionError:
        # Serveur backend non accessible - fallback sur moteur local
        print(f"⚠️  Backend API non accessible à {backend_url}, utilisation du moteur FXI local")

        try:
            result = analyze_ticker(ticker, mode)

            return {
                'score_fxi': result.final_score,
                'score': result.final_score,
                'recommandation': result.recommendation,
                'status': 'success',
                'indicateurs': {
                    'RSI': 'calculé',
                    'MACD': 'calculé',
                    'trend': 'analysé',
                    'volatilite': 'calculée'
                },
                'fondamentaux': {
                    'PE': result.details.get('pe_ratio'),
                    'prix': result.details.get('current_price'),
                    'secteur': result.details.get('sector'),
                    'industrie': result.details.get('industry')
                },
                'scores_detailles': {
                    'technique': result.technical_score,
                    'fondamental': result.fundamental_score,
                    'sentiment': result.sentiment_score,
                    'risque': result.risk_score,
                    'macro': result.macro_score
                },
                'esg': {
                    'total': 'N/A',
                    'environment': 'N/A',
                    'social': 'N/A',
                    'governance': 'N/A',
                    'grade': 'N/A'
                },
                'macro_environment': result.details.get('macro_environment', 'N/A'),
                'timestamp': result.timestamp,
                'execution_time': result.execution_time,
                'confidence': result.confidence,
                'data_quality': result.data_quality
            }
        except Exception as local_error:
            return {
                'score_fxi': 0,
                'score': 0,
                'recommandation': 'ERREUR',
                'status': 'error',
                'error': f"Erreur moteur local: {str(local_error)}",
                'indicateurs': {},
                'fondamentaux': {},
                'scores_detailles': {'technique': 0, 'fondamental': 0, 'sentiment': 0, 'risque': 0, 'macro': 0},
                'esg': {},
                'macro_environment': 'N/A',
                'timestamp': None,
                'execution_time': 0,
                'confidence': 0,
                'data_quality': 0
            }

    except Exception as e:
        # Erreur générale
        return {
            'score_fxi': 0,
            'score': 0,
            'recommandation': 'ERREUR',
            'status': 'error',
            'error': str(e),
            'indicateurs': {},
            'fondamentaux': {},
            'scores_detailles': {'technique': 0, 'fondamental': 0, 'sentiment': 0, 'risque': 0, 'macro': 0},
            'esg': {},
            'macro_environment': 'N/A',
            'timestamp': None,
            'execution_time': 0,
            'confidence': 0,
            'data_quality': 0
        }