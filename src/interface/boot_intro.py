import tkinter as tk
import random
import time
from PIL import Image, ImageTk
import pygame
import threading

def play_start_sound():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("assets/son_start.mp3")
        pygame.mixer.music.play()
    except Exception as e:
        print(f"[⚠️] Erreur audio : {e}")


    def draw_matrix():
        canvas.delete("all")
        for i in range(len(drops)):
            char = chr(0x30A0 + int(time.time() * 1000 + i * 10) % 96)
            x = i * 15
            y = drops[i] * 15
            canvas.create_text(x, y, text=char, fill="#00FF00", font=("Courier", 14))
            drops[i] += 1
            if y > height or random.random() > 0.95:
                drops[i] = 0
        root.update()

    def show_logo():
        try:
            img = Image.open("assets/logo.png")  # ✅ Chemin correct (racine projet)
            img.thumbnail((300, 150))
            logo = ImageTk.PhotoImage(img)
            canvas.create_image(width // 2, height // 2 - 40, image=logo)
            root.logo_ref = logo
            root.update()
        except Exception as e:
            print(f"[⚠️] Erreur logo : {e}")

    def loading_bar():
        bar_width = 300
        step = 20
        x0 = (width - bar_width) // 2
        y0 = height - 60

        canvas.create_text(width // 2, y0 - 20, text="Chargement moteur FXI...", fill="white", font=("Courier", 12))
        for i in range(0, bar_width + 1, step):
            canvas.create_rectangle(x0, y0, x0 + i, y0 + 20, fill="#00FF00", outline="")
            root.update()
            time.sleep(0.07)

    for _ in range(35):
        draw_matrix()
        time.sleep(0.05)

    show_logo()
    loading_bar()

    root.destroy()
