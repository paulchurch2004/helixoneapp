import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import os
import sys
import cv2

from src.interface.main_window import launch_helixone_ui as lancer_interface
from src.asset_path import get_asset_path


class VideoIntro(tk.Tk):
    """Intro vidéo synchronisée avec le son"""

    def __init__(self):
        super().__init__()
        self.title("HelixOne Boot")
        self.configure(bg="black")

        # Forcer la fenêtre au premier plan sur macOS
        self.attributes("-topmost", True)
        self.lift()
        self.focus_force()

        self._closed = False
        self.tk_img = None
        self.canvas_image_id = None
        self.cap = None

        # Ouvrir la vidéo
        video_path = get_asset_path("intro.mp4")
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            self.after(100, self.close_intro)
            return

        # Propriétés vidéo
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24
        vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Adapter à la taille max tout en gardant les proportions
        max_w, max_h = 900, 900
        scale = min(max_w / vid_w, max_h / vid_h, 1.0)
        self.display_w = int(vid_w * scale)
        self.display_h = int(vid_h * scale)
        self.need_resize = (vid_w, vid_h) != (self.display_w, self.display_h)

        # Centrer la fenêtre
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w - self.display_w) // 2
        y = (screen_h - self.display_h) // 2
        self.geometry(f"{self.display_w}x{self.display_h}+{x}+{y}")

        # Canvas vidéo
        self.canvas = tk.Canvas(
            self, width=self.display_w, height=self.display_h,
            bg="black", highlightthickness=0
        )
        self.canvas.pack()

        # Clic ou Echap pour skip
        self.canvas.bind("<Button-1>", lambda e: self.close_intro())
        self.bind("<Escape>", lambda e: self.close_intro())

        # Temps de départ pour la synchro audio/vidéo
        self.start_time = None

        # Lancer l'audio en arrière-plan
        threading.Thread(target=self.play_audio, daemon=True).start()

        # Lancer la lecture vidéo
        self.after(0, self.play_frame)

    def play_audio(self):
        """Joue l'audio (mp3 séparé ou depuis le mp4)"""
        try:
            import pygame
            pygame.mixer.init()

            audio_path = get_asset_path("intro.mp3")
            video_path = get_asset_path("intro.mp4")

            if os.path.exists(audio_path):
                pygame.mixer.music.load(audio_path)
            elif os.path.exists(video_path):
                pygame.mixer.music.load(video_path)
            else:
                return

            pygame.mixer.music.play()
            # Marquer le temps de départ pour la synchro
            self.start_time = time.time()
        except Exception:
            pass

    def play_frame(self):
        """Lit et affiche la frame vidéo synchronisée avec l'audio"""
        if self._closed:
            return

        # Attendre que l'audio démarre pour synchroniser
        if self.start_time is None:
            self.start_time = time.time()

        # Calculer quelle frame on devrait afficher selon le temps écoulé
        elapsed = time.time() - self.start_time
        target_frame = int(elapsed * self.fps)
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Sauter les frames en retard pour rester synchronisé
        skip = target_frame - current_frame
        if skip > 1:
            for _ in range(skip - 1):
                if not self.cap.grab():
                    self.close_intro()
                    return

        # Lire la frame courante
        ret, frame = self.cap.read()
        if not ret:
            self.close_intro()
            return

        # Resize avec cv2 (interpolation NEAREST = plus rapide)
        if self.need_resize:
            frame = cv2.resize(frame, (self.display_w, self.display_h),
                                    interpolation=cv2.INTER_NEAREST)

        # BGR → RGB → PIL → PhotoImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self.tk_img = ImageTk.PhotoImage(img)

        # Mettre à jour le canvas (sans recréer l'objet)
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(
                0, 0, anchor="nw", image=self.tk_img
            )
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=self.tk_img)

        # Calculer le délai jusqu'à la prochaine frame
        next_frame_time = (target_frame + 1) / self.fps
        delay_ms = max(1, int((next_frame_time - elapsed) * 1000))
        self.after(delay_ms, self.play_frame)

    def close_intro(self):
        """Arrête tout et lance l'interface principale"""
        if self._closed:
            return
        self._closed = True

        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except Exception:
            pass

        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

        self.destroy()
        lancer_interface()


if __name__ == "__main__":
    VideoIntro().mainloop()
