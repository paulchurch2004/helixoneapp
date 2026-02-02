"""
Rate Limiter global pour l'application HelixOne
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

# Instance globale du rate limiter
limiter = Limiter(key_func=get_remote_address)
