import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import pygame
from src.interface.main_window import launch_helixone_ui as lancer_interface
from src.asset_path import get_asset_path

# === Compatibilité Pillow : remplace ANTIALIAS si supprimé ===
try:
    resample_mode = Image.Resampling.LANCZOS
except AttributeError:
    resample_mode = Image.ANTIALIAS


class PlasmaIntro(tk.Tk):
    def __init__(self):
        super().__init__()

        # === Fenêtre principale ===
        self.title("HelixOne Boot")
        self.geometry("800x500+350+150")
        self.overrideredirect(True)
        self.configure(bg="#0A0A0A")

        # === Canvas pour animation ===
        self.canvas = tk.Canvas(self, width=800, height=500, bg="#0A0A0A", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # === Halo animé ===
        self.halo = self.canvas.create_oval(250, 100, 550, 400, outline="#00C9FF", width=4)

        # === Logo ===
        self.logo_img = None
        self.tk_logo = None
        self.logo_canvas_id = None
        self.load_logo()

        # === Texte de statut ===
        self.status_texts = [
            "Chargement moteur FXI...",
            "Connexion aux APIs...",
            "Analyse des sources...",
            "Initialisation interface..."
        ]
        self.status_index = 0
        self.status_label = tk.Label(self, text="", font=("Roboto", 14), fg="#00C9FF", bg="#0A0A0A")
        self.status_label.place(relx=0.5, rely=0.88, anchor="center")

        # === Son de démarrage ===
        threading.Thread(target=self.play_start_sound, daemon=True).start()

        # === Lancements des animations ===
        self.after(0, self.animate_text)
        self.after(0, self.animate_halo)
        threading.Thread(target=self.animate_logo_fade, daemon=True).start()

        # === Fermeture après délai ===
        self.after(5000, self.close_intro)

    def load_logo(self):
        try:
            img = Image.open(get_asset_path("logo.png")).convert("RGBA")
            # Redimensionner en gardant les proportions
            original_ratio = img.width / img.height
            new_width = 160
            new_height = int(new_width / original_ratio)
            img = img.resize((new_width, new_height), resample_mode)
            self.logo_img = img
            self.update_logo(alpha=1.0)
        except Exception as e:
            print(f"[⚠️] Erreur chargement logo : {e}")

    def update_logo(self, alpha=1.0):
        if not self.logo_img:
            return
        faded = self.logo_img.copy()
        alpha_layer = faded.getchannel("A").point(lambda p: int(p * alpha))
        faded.putalpha(alpha_layer)
        self.tk_logo = ImageTk.PhotoImage(faded)

        if self.logo_canvas_id:
            self.canvas.itemconfig(self.logo_canvas_id, image=self.tk_logo)
        else:
            self.logo_canvas_id = self.canvas.create_image(400, 250, image=self.tk_logo)

    def animate_logo_fade(self):
        alpha = 1.0
        delta = -0.03
        try:
            while self.winfo_exists():
                alpha += delta
                if alpha <= 0.5 or alpha >= 1.0:
                    delta *= -1
                self.update_logo(alpha)
                time.sleep(0.05)
        except tk.TclError:
            pass  # La fenêtre a été fermée pendant le thread

    def animate_text(self):
        if self.winfo_exists():
            self.status_label.config(text=self.status_texts[self.status_index])
            self.status_index = (self.status_index + 1) % len(self.status_texts)
            self.after(1200, self.animate_text)

    def animate_halo(self):
        if self.canvas.winfo_exists():
            try:
                current_width = int(float(self.canvas.itemcget(self.halo, "width")))
                next_width = 6 if current_width == 4 else 4
                next_color = "#00C9FF" if current_width == 4 else "#0277BD"
                self.canvas.itemconfig(self.halo, outline=next_color, width=next_width)
                self.after(300, self.animate_halo)
            except Exception:
                pass  # Widget supprimé, on ne relance pas

    def play_start_sound(self):
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(get_asset_path("son_start.mp3"))
            pygame.mixer.music.play()
        except Exception as e:
            print(f"[⚠️] Erreur audio : {e}")

    def close_intro(self):
        self.destroy()
        lancer_interface()


if __name__ == "__main__":
    PlasmaIntro().mainloop()
