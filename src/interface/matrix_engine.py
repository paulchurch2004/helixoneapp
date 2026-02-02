import pygame
import threading
from src.asset_path import get_asset_path

def play_fx_sound():
    def _loop():
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(get_asset_path("chiffresound.mp3"))
            pygame.mixer.music.play(-1)
        except Exception as e:
            print(f"[⚠️ FX Sound] {e}")
    threading.Thread(target=_loop, daemon=True).start()
