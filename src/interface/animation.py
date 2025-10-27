import customtkinter as ctk
import time

def matrix_intro_animation(root):
    splash = ctk.CTkToplevel(root)
    splash.geometry("600x300")
    splash.title("HelixOne - Initialisation")
    splash.configure(fg_color="#000000")

    label = ctk.CTkLabel(splash, text="Chargement HelixOne...", font=("Consolas", 26), text_color="#00FF00")
    label.pack(expand=True)

    splash.update()
    time.sleep(2.2)
    splash.destroy()
