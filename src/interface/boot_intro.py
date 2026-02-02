import tkinter as tk
import random
import time
from PIL import Image, ImageTk
import pygame
import threading
from src.asset_path import get_asset_path


def play_start_sound():
    """Joue le son de démarrage"""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(get_asset_path("son_start.mp3"))
        pygame.mixer.music.play()
    except Exception as e:
        print(f"[Warning] Erreur audio : {e}")


def show_boot_intro():
    """Affiche l'intro de boot avec effet Matrix"""
    root = tk.Tk()
    root.overrideredirect(True)

    # Dimensions de l'écran
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    width = 600
    height = 400

    # Centrer la fenêtre
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.configure(bg="black")

    # Canvas pour l'animation
    canvas = tk.Canvas(root, width=width, height=height, bg="black", highlightthickness=0)
    canvas.pack()

    # Variables pour l'effet Matrix
    num_drops = width // 15
    drops = [random.randint(0, height // 15) for _ in range(num_drops)]

    def draw_matrix():
        """Dessine l'effet Matrix"""
        canvas.delete("all")
        for i in range(len(drops)):
            char = chr(0x30A0 + int(time.time() * 1000 + i * 10) % 96)
            x_pos = i * 15
            y_pos = drops[i] * 15
            canvas.create_text(x_pos, y_pos, text=char, fill="#00FF00", font=("Courier", 14))
            drops[i] += 1
            if y_pos > height or random.random() > 0.95:
                drops[i] = 0
        root.update()

    def show_logo():
        """Affiche le logo"""
        try:
            img = Image.open(get_asset_path("logo.png"))
            img.thumbnail((300, 150))
            logo = ImageTk.PhotoImage(img)
            canvas.create_image(width // 2, height // 2 - 40, image=logo)
            root.logo_ref = logo  # Garder une référence
            root.update()
        except Exception as e:
            print(f"[Warning] Erreur logo : {e}")

    def loading_bar():
        """Affiche la barre de chargement"""
        bar_width = 300
        step = 20
        x0 = (width - bar_width) // 2
        y0 = height - 60

        canvas.create_text(width // 2, y0 - 20, text="Chargement moteur FXI...", fill="white", font=("Courier", 12))
        for i in range(0, bar_width + 1, step):
            canvas.create_rectangle(x0, y0, x0 + i, y0 + 20, fill="#00FF00", outline="")
            root.update()
            time.sleep(0.07)

    # Jouer le son dans un thread séparé
    threading.Thread(target=play_start_sound, daemon=True).start()

    # Animation Matrix
    for _ in range(35):
        draw_matrix()
        time.sleep(0.05)

    # Afficher logo et barre de chargement
    show_logo()
    loading_bar()

    # Fermer la fenêtre d'intro
    root.destroy()


# Point d'entrée si exécuté directement
if __name__ == "__main__":
    show_boot_intro()
